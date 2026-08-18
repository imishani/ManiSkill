from pathlib import Path
import copy
from typing import List, Optional
import hashlib
import json
import os
import tempfile
from contextlib import nullcontext
from time import perf_counter


import mplib
import numpy as np
import sapien
import transforms3d
import trimesh
from mplib.sapien_utils import SapienPlanningWorld, SapienPlanner
from sympy import euler

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.pose import to_sapien_pose
import sapien.physx as physx
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh

from transforms3d import quaternions, euler

import open3d as o3d

try:
    from pymosaic.domains.maniskill.planning.skills.skill_timing import (
        record_motion_planner_call,
        record_motion_planner_cache_hit,
        record_motion_planner_cache_miss,
        skill_timer,
    )
except Exception:  # pragma: no cover - ManiSkill example may run standalone.
    def skill_timer(skill_name, section_name):
        return nullcontext()

    def record_motion_planner_call(*args, **kwargs):
        return None

    def record_motion_planner_cache_hit(*args, **kwargs):
        return None

    def record_motion_planner_cache_miss(*args, **kwargs):
        return None



def _motion_result_status(result):
    if isinstance(result, dict):
        return str(result.get("status", "unknown"))
    return "unknown"


def _motion_result_waypoints(result):
    if not isinstance(result, dict):
        return None
    position = result.get("position")
    try:
        return int(position.shape[0])
    except Exception:
        return None


def _round_list(values, decimals=3):
    try:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
        return []
    return [float(x) for x in np.round(arr, int(decimals)).tolist()]


def _motion_request_key(method, current_qpos, *, pose=None, qpose=None, extra=None):
    parts = [str(method)]
    parts.append("q=" + repr(_round_list(current_qpos, decimals=2)))
    if pose is not None:
        try:
            p = _round_list(pose.p, decimals=3)
            q = _round_list(pose.q, decimals=3)
            parts.append("pose=" + repr((p, q)))
        except Exception:
            parts.append("pose=<unavailable>")
    if qpose is not None:
        parts.append("goalq=" + repr(_round_list(qpose, decimals=2)))
    if extra is not None:
        parts.append("extra=" + str(extra))
    return "|".join(parts)


OPEN = 0
HALF_CLOSED=0.4
CLOSED = 0.8

class RidgebackUR10ePlanningSolver:
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        motion_backend: Optional[str] = None,
        srmp_srdf_path: Optional[str] = None,
        srmp_time_limit_s: Optional[float] = None,
        srmp_search_resolution_deg: Optional[float] = None,
        srmp_search_heuristic: Optional[str] = None,
        srmp_search_weight: Optional[float] = None,
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        if motion_backend is None:
            motion_backend = getattr(env, "motion_backend", "mplib")
        self.motion_backend = str(motion_backend).lower()
        if self.motion_backend not in {"mplib", "srmp"}:
            raise ValueError(f"Unsupported motion backend: {self.motion_backend!r}")
        if srmp_srdf_path is None:
            srmp_srdf_path = getattr(env, "srmp_srdf_path", None)
        self.srmp_srdf_path = srmp_srdf_path
        self.srmp_planner_id = "wAstar"
        if srmp_time_limit_s is None:
            srmp_time_limit_s = getattr(env, "srmp_time_limit_s", 1.0)
        self.srmp_time_limit_s = float(max(0.01, srmp_time_limit_s))
        if srmp_search_resolution_deg is None:
            srmp_search_resolution_deg = getattr(env, "srmp_search_resolution_deg", 1.0)
        if srmp_search_heuristic is None:
            srmp_search_heuristic = getattr(env, "srmp_search_heuristic", "bfs")
        if srmp_search_weight is None:
            srmp_search_weight = getattr(env, "srmp_search_weight", 50.0)
        self.srmp_search_resolution_deg = float(srmp_search_resolution_deg)
        self.srmp_search_heuristic = str(srmp_search_heuristic)
        self.srmp_search_weight = float(srmp_search_weight)
        if self.srmp_search_resolution_deg <= 0.0:
            raise ValueError("SRMP search resolution must be positive")
        if self.srmp_search_weight < 1.0:
            raise ValueError("SRMP weighted-A* weight must be at least 1")
        self.srmp_planner = None
        self._srmp_robot_name = "static_ridgebackur10e"
        self._last_srmp_failure_detail = None
        self._srmp_trace_path = os.environ.get("PYMOSAIC_SRMP_TRACE_PATH")
        self._srmp_attachment_state = {}
        self._srmp_collision_world_key = None
        self._srmp_scene_object_names = []
        self._srmp_mesh_paths = {}
        self._srmp_result_cache = {}
        self._srmp_result_cache_hits = 0
        self._srmp_search_ready = False
        self._srmp_planner_context = None
        self._srmp_batched_ik = None

        self.base_pose = to_sapien_pose(base_pose)

        self.planner = self.setup_planner()
        if self.motion_backend == "srmp":
            self.srmp_planner = self.setup_srmp_planner()

        self.control_mode = self.base_env.control_mode

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.gripper_state = OPEN
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            if "grasp_pose_visual" not in self.base_env.scene.actors:
                self.grasp_pose_visual = build_panda_gripper_grasp_pose_visual(
                    self.base_env.scene
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
            self.grasp_pose_visual.set_pose(self.base_env.agent.tcp.pose)
        self.elapsed_steps = 0

        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None

        # self.planner = None

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.base_env.render_human()
        while True:
            if viewer.window.key_down("c"):
                break
            self.base_env.render_human()

    def setup_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        dof = 9 if self.robot.name == "ridgebackur10e" else 6 if self.robot.name == "static_ridgebackur10e" else 7
        planner = mplib.Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="ur_arm_TCP",
            joint_vel_limits=np.ones(dof) * self.joint_vel_limits,
            joint_acc_limits=np.ones(dof) * self.joint_acc_limits,
            use_convex=False,
        )

        # planning_world = SapienPlanningWorld(
        #     self.base_env.scene.sub_scenes[0],
        #     [self.robot._objs[0]]
        # )
        # planner = SapienPlanner(planning_world,
        #                         "scene-0-ridgebackur10e_ur_arm_TCP",
        #                         joint_vel_limits=np.ones(9) * self.joint_vel_limits,
        #                         joint_acc_limits=np.ones(9) * self.joint_acc_limits)

        pose = mplib.pymp.Pose(p=self.base_pose.p, q=self.base_pose.q)
        # print(f"Setting base pose {pose}")
        planner.set_base_pose(pose)
        # planner.update_from_simulation()
        return planner

    def setup_srmp_planner(self):
        if not self.srmp_srdf_path:
            raise ValueError("SRMP backend requires an SRDF path")
        srdf_path = Path(self.srmp_srdf_path).expanduser().resolve()
        if not srdf_path.is_file():
            raise FileNotFoundError(f"SRMP SRDF not found: {srdf_path}")

        try:
            import srmp
        except ImportError as exc:
            raise ImportError(
                "SRMP backend requested but srmp is not installed in this environment"
            ) from exc

        # SRMP defaults to a [-1, 1] x/y occupancy grid, which excludes the
        # auto-shop shelves at x > 1 m. Keep the grid local to this UR10e
        # solver while covering the robot's actual single-arm workspace.
        from ims.pyims import GridConfig

        grid = GridConfig()
        grid.origin_x = -1.0
        grid.origin_y = -1.5
        grid.origin_z = 0.0
        grid.size_x = 3.0
        grid.size_y = 3.0
        grid.size_z = 2.5
        grid.resolution = 0.02

        urdf_path = Path(self.env_agent.urdf_path).expanduser().resolve()
        planner = srmp.PlannerInterface(grid)
        active_joint_names = [
            joint.get_name() for joint in self.robot.get_active_joints()
        ]
        non_gripper_names = set(self.env_agent.arm_joint_names) | set(
            getattr(self.env_agent, "base_joint_names", ())
        )
        self._srmp_gripper_joint_names = [
            name for name in active_joint_names if name not in non_gripper_names
        ]
        if not self._srmp_gripper_joint_names:
            raise RuntimeError("SRMP robot has no active gripper joints")

        self._srmp_robot_name = planner.add_robot(
            str(urdf_path),
            name=self._srmp_robot_name,
            srdf_path=str(srdf_path),
            end_effector=self.env_agent.ee_link_name,
            gripper_joint_names=self._srmp_gripper_joint_names,
        )

        expected_joints = list(self.env_agent.arm_joint_names)
        move_group_joints = planner.get_move_group_joint_names(self._srmp_robot_name)
        if move_group_joints != expected_joints:
            raise RuntimeError(
                "SRMP move-group mismatch: "
                f"expected {expected_joints}, got {move_group_joints}"
            )
        if planner.get_move_group_qpos_dim(self._srmp_robot_name) != len(expected_joints):
            raise RuntimeError("SRMP move group does not have six controllable arm DOFs")
        actual_gripper_joints = planner.get_gripper_joint_names(
            self._srmp_robot_name
        )
        if actual_gripper_joints != self._srmp_gripper_joint_names:
            raise RuntimeError(
                "SRMP gripper-joint mismatch: "
                f"expected {self._srmp_gripper_joint_names}, "
                f"got {actual_gripper_joints}"
            )

        planner.set_base_pose(
            self._srmp_robot_name,
            srmp.Pose(p=np.asarray(self.base_pose.p), q=np.asarray(self.base_pose.q)),
        )
        self._srmp_planner_context = {
            "planner_id": self.srmp_planner_id,
            "heuristic": self.srmp_search_heuristic,
            "weight": str(self.srmp_search_weight),
            "resolution": str(self.srmp_search_resolution_deg),
            "time_limit": str(self.srmp_time_limit_s),
        }
        return planner

    def _ensure_srmp_search_planner(self):
        if not self._srmp_search_ready:
            self.srmp_planner.make_planner(
                [self._srmp_robot_name], self._srmp_planner_context
            )
            self._srmp_search_ready = True

    def _move_group_qpos(self, qpos) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        indices = np.asarray(self.planner.move_group_joint_indices, dtype=np.int64)
        if qpos.size == indices.size:
            return qpos.copy()
        if indices.size == 0 or qpos.size <= int(indices.max()):
            raise ValueError(
                f"Cannot map qpos with {qpos.size} values to move-group indices {indices.tolist()}"
            )
        return qpos[indices]

    def _sync_srmp_gripper_qpos(self, qpos=None) -> np.ndarray:
        active_joint_names = [
            joint.get_name() for joint in self.robot.get_active_joints()
        ]
        values = np.asarray(
            self.robot.get_qpos().cpu().numpy()[0] if qpos is None else qpos,
            dtype=np.float64,
        ).reshape(-1)
        if values.size != len(active_joint_names):
            values = np.asarray(
                self.robot.get_qpos().cpu().numpy()[0], dtype=np.float64
            ).reshape(-1)
        if values.size != len(active_joint_names):
            raise RuntimeError(
                "Cannot map ManiSkill qpos to active joints for SRMP gripper"
            )
        qpos_by_name = dict(zip(active_joint_names, values))
        gripper_qpos = np.asarray(
            [qpos_by_name[name] for name in self._srmp_gripper_joint_names],
            dtype=np.float64,
        )
        self.srmp_planner.set_gripper_qpos(
            self._srmp_robot_name, gripper_qpos
        )
        return gripper_qpos

    def _time_parameterize_srmp_path(self, positions: np.ndarray) -> dict:
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 6 or positions.shape[0] == 0:
            return {"status": "SRMP returned an invalid trajectory"}

        max_step = max(
            1e-4,
            float(self.joint_vel_limits) * float(self.base_env.control_timestep),
        )
        dense = [positions[0]]
        for target in positions[1:]:
            start = dense[-1]
            steps = int(max(1, np.ceil(np.max(np.abs(target - start)) / max_step)))
            for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
                dense.append(start + float(alpha) * (target - start))
        dense_positions = np.asarray(dense, dtype=np.float64)
        velocities = np.zeros_like(dense_positions)
        if dense_positions.shape[0] > 1:
            velocities[1:] = np.diff(dense_positions, axis=0) / float(
                self.base_env.control_timestep
            )
            velocities[0] = velocities[1]
        return {
            "status": "Success",
            "position": dense_positions,
            "velocity": velocities,
            "acceleration": np.zeros_like(dense_positions),
            "duration": float(
                max(0, dense_positions.shape[0] - 1) * self.base_env.control_timestep
            ),
        }

    def _srmp_trajectory_result(self, trajectory) -> dict:
        """Sample SRMP's timed trajectory at the simulator control period."""
        positions = np.asarray(trajectory.positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 6 or positions.shape[0] == 0:
            return {"status": "SRMP returned an invalid trajectory"}

        times = np.asarray(getattr(trajectory, "times", []), dtype=np.float64)
        if (
            times.shape != (positions.shape[0],)
            or positions.shape[0] < 2
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
            or times[-1] <= times[0]
        ):
            return self._time_parameterize_srmp_path(positions)

        relative_times = times - times[0]
        duration = float(relative_times[-1])
        dt = float(self.base_env.control_timestep)
        sample_times = np.arange(0.0, duration, dt, dtype=np.float64)
        if sample_times.size == 0 or not np.isclose(sample_times[-1], duration):
            sample_times = np.append(sample_times, duration)

        def interpolate(values, fallback):
            array = np.asarray(values, dtype=np.float64)
            if array.shape != positions.shape or not np.all(np.isfinite(array)):
                return fallback
            return np.column_stack(
                [np.interp(sample_times, relative_times, array[:, j]) for j in range(6)]
            )

        dense_positions = np.column_stack(
            [np.interp(sample_times, relative_times, positions[:, j]) for j in range(6)]
        )
        finite_difference_velocity = np.zeros_like(dense_positions)
        if dense_positions.shape[0] > 1:
            delta_t = np.diff(sample_times)
            finite_difference_velocity[1:] = (
                np.diff(dense_positions, axis=0) / delta_t[:, None]
            )
            finite_difference_velocity[0] = finite_difference_velocity[1]
        velocities = interpolate(
            getattr(trajectory, "velocities", []),
            finite_difference_velocity,
        )
        accelerations = interpolate(
            getattr(trajectory, "accelerations", []),
            np.zeros_like(dense_positions),
        )
        return {
            "status": "Success",
            "position": dense_positions,
            "velocity": velocities,
            "acceleration": accelerations,
            "duration": duration,
        }

    def _srmp_distance_detail(self) -> dict:
        detail = {}
        for key, method_name in (
            ("self_distance", "distance_to_self_collision"),
            ("environment_distance", "distance_to_robot_collision"),
        ):
            try:
                detail[key] = float(getattr(self.srmp_planner, method_name)())
            except Exception as exc:
                detail[key + "_error"] = f"{type(exc).__name__}: {exc}"
        return detail

    def _mplib_collision_detail(self, qpos) -> dict:
        q = self.planner.pad_move_group_qpos(self._move_group_qpos(qpos))
        detail = {}
        for key, method_name in (
            ("environment", "check_for_env_collision"),
            ("self", "check_for_self_collision"),
        ):
            try:
                collisions = list(getattr(self.planner, method_name)(q) or [])
                detail[key + "_collision_count"] = len(collisions)
                if collisions:
                    detail[key + "_collisions"] = [str(item) for item in collisions[:8]]
            except Exception as exc:
                detail[key + "_collision_error"] = f"{type(exc).__name__}: {exc}"
        detail["colliding"] = bool(
            detail.get("environment_collision_count", 0)
            or detail.get("self_collision_count", 0)
        )
        return detail

    def _srmp_collision_world_summary(self) -> dict:
        mplib_attached = []
        try:
            world = self.planner.planning_world
            mplib_attached = sorted(
                name
                for name in world.get_object_names()
                if world.is_object_attached(name)
            )
        except Exception:
            pass
        return {
            "native_scene_object_count": len(self._srmp_scene_object_names),
            "collision_world_key": repr(self._srmp_collision_world_key),
            "mplib_attached_objects": mplib_attached,
            "srmp_attached_objects": dict(self._srmp_attachment_state),
        }

    def _write_srmp_trace(self, record: dict) -> None:
        if not self._srmp_trace_path:
            return
        path = Path(self._srmp_trace_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _srmp_request_cache_key(
        self, start, goals, *, goal_type="joints", gripper_qpos=None
    ):
        start_array = np.ascontiguousarray(start, dtype=np.float64)
        goal_arrays = tuple(
            np.ascontiguousarray(goal, dtype=np.float64).tobytes() for goal in goals
        )
        attachment_key = json.dumps(
            self._srmp_attachment_state, sort_keys=True, separators=(",", ":")
        )
        return (
            str(goal_type),
            start_array.tobytes(),
            goal_arrays,
            np.ascontiguousarray(gripper_qpos, dtype=np.float64).tobytes(),
            self._srmp_collision_world_key,
            attachment_key,
            self.srmp_planner_id,
            self.srmp_search_heuristic,
            float(self.srmp_search_weight),
            float(self.srmp_search_resolution_deg),
            float(self.srmp_time_limit_s),
        )

    def _srmp_plan_qpos(self, goals, current_qpos, *, target_pose=None) -> dict:
        import srmp

        started_at = perf_counter()
        self._last_srmp_failure_detail = None
        self._last_srmp_cache_hit = False
        start = self._move_group_qpos(current_qpos)
        gripper_qpos = self._sync_srmp_gripper_qpos(current_qpos)
        self._ensure_srmp_search_planner()
        goal_list = [self._move_group_qpos(goal) for goal in goals]
        cache_goals = list(goal_list)
        cache_goal_type = "joints"
        if target_pose is not None:
            target_pose = to_sapien_pose(target_pose)
            target_vector = np.concatenate(
                [
                    np.asarray(target_pose.p, dtype=np.float64).reshape(3),
                    np.asarray(target_pose.q, dtype=np.float64).reshape(4),
                ]
            )
            cache_goals.append(target_vector)
            cache_goal_type = "joints_for_pose"
        cache_key = self._srmp_request_cache_key(
            start,
            cache_goals,
            goal_type=cache_goal_type,
            gripper_qpos=gripper_qpos,
        )
        cached = self._srmp_result_cache.get(cache_key)
        if cached is not None:
            self._srmp_result_cache_hits += 1
            self._last_srmp_cache_hit = True
            result, failure_detail = copy.deepcopy(cached)
            self._last_srmp_failure_detail = failure_detail
            if self._last_srmp_failure_detail is None:
                self._last_srmp_failure_detail = {"reason": "exact_cache_hit", "cache_hit": True}
            else:
                self._last_srmp_failure_detail["cache_hit"] = True
            return result
        trace = None
        if self._srmp_trace_path:
            trace = {
                "backend": "srmp",
                "planner_id": self.srmp_planner_id,
                "time_limit_s": self.srmp_time_limit_s,
                "start": start.tolist(),
                "goals": [goal.tolist() for goal in goal_list],
                "target_pose": (
                    None
                    if target_pose is None
                    else {
                        "p": np.asarray(target_pose.p).tolist(),
                        "q_wxyz": np.asarray(target_pose.q).tolist(),
                    }
                ),
                "collision_world": self._srmp_collision_world_summary(),
                "mplib_start": self._mplib_collision_detail(start),
                "goal_checks": [],
            }

        def finish(result, failure_detail=None):
            if failure_detail is not None:
                self._last_srmp_failure_detail = failure_detail
            status = _motion_result_status(result)
            reason = (failure_detail or {}).get("reason")
            if status == "Success" or reason in {"invalid_start_state", "invalid_goal_state"}:
                self._srmp_result_cache[cache_key] = copy.deepcopy(
                    (result, self._last_srmp_failure_detail)
                )
            if trace is not None:
                trace["status"] = _motion_result_status(result)
                trace["duration_s"] = float(perf_counter() - started_at)
                trace["failure_detail"] = self._last_srmp_failure_detail
                trace["waypoint_count"] = _motion_result_waypoints(result)
                self._write_srmp_trace(trace)
            return result

        for goal in goal_list:
            # SRMP 0.1.3.6 segfaults on an already-satisfied joint goal.
            if np.allclose(start, goal, atol=1e-8, rtol=0.0):
                return finish(self._time_parameterize_srmp_path(start[None, :]))

        self.srmp_planner.set_qpos(self._srmp_robot_name, start)
        start_colliding = bool(
            self.srmp_planner.is_state_colliding(self._srmp_robot_name)
        )
        if trace is not None:
            trace["srmp_start"] = {
                "colliding": start_colliding,
                **self._srmp_distance_detail(),
            }
        if start_colliding:
            return finish(
                {"status": "Invalid start state"},
                {
                    "reason": "invalid_start_state",
                    "valid_goal_count": 0,
                    "invalid_goal_count": len(goal_list),
                },
            )

        # SRMP's lattice cannot expand configurations effectively on a joint
        # limit.  Enter the lattice through a short, collision-checked bridge.
        joint_limits = np.asarray(
            self.srmp_planner.get_move_group_joint_limits(self._srmp_robot_name),
            dtype=np.float64,
        )
        lattice_margin = 0.5 * np.deg2rad(self.srmp_search_resolution_deg)
        search_start = np.clip(
            start,
            joint_limits[:, 0] + lattice_margin,
            joint_limits[:, 1] - lattice_margin,
        )
        start_bridge = None
        if not np.allclose(search_start, start, atol=1e-12, rtol=0.0):
            bridge_steps = max(
                1, int(np.ceil(np.max(np.abs(search_start - start)) / 0.002))
            )
            start_bridge = np.linspace(start, search_start, bridge_steps + 1)
            for qpos in start_bridge[1:]:
                self.srmp_planner.set_qpos(self._srmp_robot_name, qpos)
                if bool(self.srmp_planner.is_state_colliding(self._srmp_robot_name)):
                    self.srmp_planner.set_qpos(self._srmp_robot_name, start)
                    return finish(
                        {"status": "Invalid start state"},
                        {"reason": "joint_limit_start_bridge_collision"},
                    )
            if trace is not None:
                trace["srmp_start"]["search_start"] = search_start.tolist()

        valid_goals = []
        invalid_details = []
        for goal in goal_list:
            self.srmp_planner.set_qpos(self._srmp_robot_name, goal)
            colliding = bool(
                self.srmp_planner.is_state_colliding(self._srmp_robot_name)
            )
            collision_detail = {
                "goal": np.round(goal, 4).tolist(),
                "colliding": colliding,
                **self._srmp_distance_detail(),
            }
            if trace is not None:
                trace["goal_checks"].append(
                    {
                        **collision_detail,
                        "mplib": self._mplib_collision_detail(goal),
                    }
                )
            if colliding:
                invalid_details.append(collision_detail)
            else:
                valid_goals.append(goal)
        self.srmp_planner.set_qpos(self._srmp_robot_name, start)
        if not valid_goals:
            return finish(
                {"status": "Invalid goal state"},
                {
                    "reason": "invalid_goal_state",
                    "valid_goal_count": 0,
                    "invalid_goal_count": len(invalid_details),
                    "goals": invalid_details,
                },
            )

        try:
            trajectory = self.srmp_planner.plan(
                search_start,
                srmp.GoalConstraint(srmp.GoalType.JOINTS, valid_goals),
            )
        except RuntimeError as exc:
            return finish(
                {"status": f"SRMP planning error: {exc}"},
                {
                    "reason": "planning_error",
                    "error": str(exc),
                    "valid_goal_count": len(valid_goals),
                    "invalid_goal_count": len(invalid_details),
                    "start": np.round(start, 4).tolist(),
                    "goals": [np.round(goal, 4).tolist() for goal in valid_goals],
                },
            )
        positions = np.asarray(trajectory.positions, dtype=np.float64)
        if positions.size == 0:
            return finish(
                {"status": "SRMP failed to find a path"},
                {
                    "reason": "search_failed",
                    "valid_goal_count": len(valid_goals),
                    "invalid_goal_count": len(invalid_details),
                },
            )
        if positions.ndim == 1:
            positions = positions[None, :]
        if start_bridge is not None:
            positions = np.concatenate([start_bridge[:-1], positions], axis=0)
        if target_pose is not None:
            endpoint_ok, endpoint_detail = self._srmp_pose_endpoint_ok(
                positions[-1],
                target_pose,
            )
            endpoint_errors = [
                float(np.max(np.abs(goal - positions[-1]))) for goal in valid_goals
            ]
            endpoint_goal = valid_goals[int(np.argmin(endpoint_errors))]
            endpoint_error = min(endpoint_errors)
            refinement_limit_rad = np.deg2rad(self.srmp_search_resolution_deg) + 1e-6
            endpoint_detail.update(
                {
                    "endpoint_error_rad": endpoint_error,
                    "refinement_limit_rad": refinement_limit_rad,
                }
            )
            if not endpoint_ok and endpoint_error <= refinement_limit_rad:
                correction = np.linspace(
                    positions[-1],
                    endpoint_goal,
                    max(1, int(np.ceil(endpoint_error / 0.01))) + 1,
                    dtype=np.float64,
                )[1:]
                for qpos in correction:
                    self.srmp_planner.set_qpos(self._srmp_robot_name, qpos)
                    if bool(self.srmp_planner.is_state_colliding(self._srmp_robot_name)):
                        self.srmp_planner.set_qpos(self._srmp_robot_name, start)
                        return finish(
                            {"status": "SRMP endpoint refinement collides"},
                            {
                                "reason": "endpoint_refinement_collision",
                                **endpoint_detail,
                            },
                        )
                positions = np.concatenate([positions, correction], axis=0)
                endpoint_ok, refined_detail = self._srmp_pose_endpoint_ok(
                    positions[-1], target_pose
                )
                endpoint_detail.update(refined_detail)
                endpoint_detail["refined"] = True
                self.srmp_planner.set_qpos(self._srmp_robot_name, start)
            if trace is not None:
                trace["endpoint_check"] = endpoint_detail
            if not endpoint_ok:
                return finish(
                    {"status": "SRMP pose goal not reached"},
                    {"reason": "pose_goal_not_reached", **endpoint_detail},
                )
        else:
            endpoint_errors = [
                float(np.max(np.abs(goal - positions[-1]))) for goal in valid_goals
            ]
            goal_index = int(np.argmin(endpoint_errors))
            endpoint_error = endpoint_errors[goal_index]
            endpoint_goal = valid_goals[goal_index]
            endpoint_tolerance_rad = 1e-4
            refinement_limit_rad = np.deg2rad(self.srmp_search_resolution_deg) + 1e-6
            endpoint_detail = {
                "endpoint_error_rad": endpoint_error,
                "endpoint_tolerance_rad": endpoint_tolerance_rad,
                "refinement_limit_rad": refinement_limit_rad,
                "requested_goal": endpoint_goal.tolist(),
                "returned_endpoint": positions[-1].tolist(),
            }
            if trace is not None:
                trace["endpoint_check"] = endpoint_detail
            if endpoint_error > refinement_limit_rad:
                return finish(
                    {"status": "SRMP joint goal not reached"},
                    {"reason": "joint_goal_not_reached", **endpoint_detail},
                )
            if endpoint_error > endpoint_tolerance_rad:
                correction_steps = max(1, int(np.ceil(endpoint_error / 0.01)))
                correction = np.linspace(
                    positions[-1],
                    endpoint_goal,
                    correction_steps + 1,
                    dtype=np.float64,
                )[1:]
                for qpos in correction:
                    self.srmp_planner.set_qpos(self._srmp_robot_name, qpos)
                    if bool(self.srmp_planner.is_state_colliding(self._srmp_robot_name)):
                        self.srmp_planner.set_qpos(self._srmp_robot_name, start)
                        return finish(
                            {"status": "SRMP endpoint refinement collides"},
                            {
                                "reason": "endpoint_refinement_collision",
                                **endpoint_detail,
                            },
                        )
                self.srmp_planner.set_qpos(self._srmp_robot_name, start)
                positions = np.concatenate([positions, correction], axis=0)

        result = self._time_parameterize_srmp_path(positions)
        if result.get("status") != "Success":
            return finish(result, {"reason": "invalid_trajectory"})
        return finish(result)

    def _srmp_pose_endpoint_ok(self, qpos, target_pose) -> tuple[bool, dict]:
        target_pose = to_sapien_pose(target_pose)
        target_p = np.asarray(target_pose.p, dtype=np.float64).reshape(3)
        target_q = np.asarray(target_pose.q, dtype=np.float64).reshape(4)
        target_q /= max(float(np.linalg.norm(target_q)), 1e-12)
        final_pose = self.srmp_planner.compute_fk(
            self._srmp_robot_name,
            np.asarray(qpos, dtype=np.float64),
        )
        final_p = np.asarray(final_pose.p, dtype=np.float64).reshape(3)
        final_q = np.asarray(final_pose.q, dtype=np.float64).reshape(4)
        final_q /= max(float(np.linalg.norm(final_q)), 1e-12)
        position_error_m = float(np.linalg.norm(final_p - target_p))
        quaternion_dot = float(np.clip(abs(np.dot(final_q, target_q)), 0.0, 1.0))
        orientation_error_rad = float(2.0 * np.arccos(quaternion_dot))
        position_tolerance_m = 0.005
        orientation_tolerance_rad = float(np.deg2rad(2.0))
        detail = {
            "position_error_m": position_error_m,
            "orientation_error_rad": orientation_error_rad,
            "position_tolerance_m": position_tolerance_m,
            "orientation_tolerance_rad": orientation_tolerance_rad,
            "final_pose": {
                "p": final_p.tolist(),
                "q_wxyz": final_q.tolist(),
            },
        }
        return (
            position_error_m <= position_tolerance_m
            and orientation_error_rad <= orientation_tolerance_rad,
            detail,
        )

    def _srmp_plan_pose(self, pose, current_qpos) -> dict:
        import pytorch_kinematics as pk
        import torch

        started_at = perf_counter()
        self._last_srmp_failure_detail = None
        pose = to_sapien_pose(pose)
        start = self._move_group_qpos(current_qpos)
        self._sync_srmp_gripper_qpos(current_qpos)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._srmp_batched_ik is None:
            with open(self.env_agent.urdf_path, "rb") as stream:
                chain = pk.build_serial_chain_from_urdf(
                    stream.read(),
                    end_link_name=self.env_agent.ee_link_name,
                    root_link_name="ur_arm_base_link",
                ).to(dtype=torch.float32, device=device)
            if chain.get_joint_parameter_names() != list(self.env_agent.arm_joint_names):
                raise RuntimeError("Batched IK joint order does not match the SRMP move group")
            limits = torch.tensor(
                np.asarray(chain.get_joint_limits()).T,
                dtype=torch.float32,
                device=device,
            )
            self._srmp_batched_ik = pk.PseudoInverseIK(
                chain,
                joint_limits=limits,
                num_retries=64,
                max_iterations=100,
                early_stopping_any_converged=False,
                early_stopping_no_improvement=None,
            )

        pk_start = self._srmp_batched_ik.chain.forward_kinematics(
            torch.tensor(start, dtype=torch.float32, device=device).unsqueeze(0)
        ).get_matrix()[0].detach().cpu().numpy()
        srmp_start_pose = self.srmp_planner.compute_fk(
            self._srmp_robot_name, start
        )
        srmp_start = np.eye(4, dtype=np.float64)
        srmp_start[:3, :3] = transforms3d.quaternions.quat2mat(
            np.asarray(srmp_start_pose.q, dtype=np.float64)
        )
        srmp_start[:3, 3] = np.asarray(srmp_start_pose.p, dtype=np.float64)
        world_from_pk = srmp_start @ np.linalg.inv(pk_start)
        target_world = pose.to_transformation_matrix()
        if hasattr(target_world, "detach"):
            target_world = target_world.detach().cpu().numpy()
        target_world = np.asarray(target_world).reshape(4, 4)
        target_matrix = torch.tensor(
            np.linalg.inv(world_from_pk) @ target_world,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        ik_started_at = perf_counter()
        solutions = self._srmp_batched_ik.solve(pk.Transform3d(matrix=target_matrix))
        converged = solutions.converged[0].detach().cpu().numpy().astype(bool)
        candidates = solutions.solutions[0].detach().cpu().numpy()[converged]
        ik_duration_s = float(perf_counter() - ik_started_at)

        valid_goals = []
        for candidate in candidates:
            self.srmp_planner.set_qpos(self._srmp_robot_name, candidate)
            if not self.srmp_planner.is_state_colliding(self._srmp_robot_name):
                valid_goals.append(np.asarray(candidate, dtype=np.float64))
        self.srmp_planner.set_qpos(self._srmp_robot_name, start)

        if self._srmp_trace_path:
            self._write_srmp_trace({
                "backend": "srmp",
                "goal_type": "batched_pytorch_ik",
                "start": start.tolist(),
                "target_pose": {
                    "p": np.asarray(pose.p).tolist(),
                    "q_wxyz": np.asarray(pose.q).tolist(),
                },
                "candidate_count": int(len(candidates)),
                "valid_goal_count": int(len(valid_goals)),
                "device": str(device),
                "ik_duration_s": ik_duration_s,
                "duration_s": float(perf_counter() - started_at),
            })
        if not valid_goals:
            self._last_srmp_failure_detail = {
                "reason": "no_collision_free_ik",
                "candidate_count": int(len(candidates)),
            }
            return {"status": "SRMP found no collision-free IK goal"}
        valid_goal = min(valid_goals, key=lambda goal: np.linalg.norm(goal - start))
        return self._srmp_plan_qpos([valid_goal], current_qpos, target_pose=pose)

    def _srmp_plan_screw(self, pose, current_qpos, *, wrt_world=True) -> dict:
        self._last_srmp_failure_detail = None
        pose = to_sapien_pose(pose)
        target_pose = pose if wrt_world else self.base_pose * pose
        start = self._move_group_qpos(current_qpos)
        self._sync_srmp_gripper_qpos(current_qpos)
        try:
            trajectory = self.srmp_planner.plan_screw(
                self._srmp_robot_name,
                self.env_agent.ee_link_name,
                start_qpos=start,
                end_pose=self._srmp_pose(target_pose),
                qpos_step=0.05,
            )
            result = self._srmp_trajectory_result(trajectory)
            if result.get("status") != "Success":
                return result
            endpoint_ok, detail = self._srmp_pose_endpoint_ok(
                np.asarray(result["position"])[-1], target_pose
            )
            if not endpoint_ok:
                self._last_srmp_failure_detail = {
                    "reason": "screw_pose_not_reached",
                    **detail,
                }
                return {"status": "SRMP screw pose not reached"}
            return result
        except RuntimeError as exc:
            self._last_srmp_failure_detail = {
                "reason": "screw_failed",
                "error": str(exc),
            }
            return {"status": f"SRMP screw failed: {exc}"}
        finally:
            self.srmp_planner.set_qpos(self._srmp_robot_name, start)

    def attach_mesh(self, mesh_path, relative_pose) -> Optional[str]:
        """Attach one carried collision mesh to the TCP in every active backend."""
        before_names = set(self.planner.planning_world.get_object_names())
        self.planner.update_attached_mesh(mesh_path, relative_pose)
        after_names = set(self.planner.planning_world.get_object_names())
        mplib_names = [
            name
            for name in after_names
            if self.planner.planning_world.is_object_attached(name)
        ]
        new_names = sorted(set(mplib_names) - before_names)
        mplib_name = (
            new_names[0]
            if new_names
            else (sorted(mplib_names)[0] if mplib_names else None)
        )

        if self.srmp_planner is not None:
            import srmp

            srmp_name = "mosaic_attached_geom"
            if self.srmp_planner.has_object(srmp_name):
                if self.srmp_planner.is_object_attached(srmp_name):
                    self.srmp_planner.detach_object(srmp_name, also_remove=True)
                else:
                    self.srmp_planner.remove_object(srmp_name)
            current_qpos = self.robot.get_qpos().cpu().numpy()[0]
            self.srmp_planner.set_qpos(
                self._srmp_robot_name,
                self._move_group_qpos(current_qpos),
            )
            # SRMP attaches geometry at the link origin and has no attachment-
            # offset argument. Bake object-in-link into the mesh itself.
            tcp_link_name = self.env_agent.ee_link_name
            tcp_pose = self.srmp_planner.get_link_pose(
                self._srmp_robot_name,
                tcp_link_name,
            )
            link_name = "robotiq_arg2f_base_link"
            link_id = self.srmp_planner.get_link_index(
                self._srmp_robot_name,
                link_name,
            )
            object_in_tcp = srmp.Pose(
                p=np.asarray(relative_pose.p, dtype=np.float64),
                q=np.asarray(relative_pose.q, dtype=np.float64),
            )
            object_world = tcp_pose * object_in_tcp
            link_pose = self.srmp_planner.get_link_pose(
                self._srmp_robot_name,
                link_name,
            )
            object_in_link = (
                np.linalg.inv(link_pose.to_transformation_matrix())
                @ object_world.to_transformation_matrix()
            )
            attached_mesh_path = self._srmp_transformed_mesh_path(
                mesh_path,
                object_in_link,
            )
            self.srmp_planner.add_mesh(
                srmp_name,
                attached_mesh_path,
                pose=link_pose,
            )
            self.srmp_planner.attach_object(
                srmp_name,
                self._srmp_robot_name,
                link_id,
            )
            self.srmp_planner.update_attached_bodies_pose()
            self._srmp_attachment_state = {
                srmp_name: {
                    "link_name": link_name,
                    "link_id": int(link_id),
                    "mesh_path": attached_mesh_path,
                    "relative_pose": {
                        "p": np.asarray(relative_pose.p, dtype=np.float64).tolist(),
                        "q": np.asarray(relative_pose.q, dtype=np.float64).tolist(),
                    },
                }
            }
        return mplib_name

    def _srmp_transformed_mesh_path(self, mesh_path, transform: np.ndarray) -> str:
        mesh = trimesh.load_mesh(str(mesh_path), process=False)
        transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        digest = hashlib.sha256(
            np.ascontiguousarray(mesh.vertices).tobytes()
            + np.ascontiguousarray(mesh.faces).tobytes()
            + transform.tobytes()
        ).hexdigest()
        cached = self._srmp_mesh_paths.get(digest)
        if cached is not None:
            return cached
        mesh.apply_transform(transform)
        fd, path = tempfile.mkstemp(prefix="mosaic_srmp_attached_", suffix=".stl")
        os.close(fd)
        mesh.export(path)
        self._srmp_mesh_paths[digest] = path
        return path

    def detach_attached_mesh(self, mplib_name: Optional[str] = None) -> None:
        """Detach carried geometry from every active backend."""
        world = self.planner.planning_world
        # Remove the tracked object and stale attachments left by interrupted solves.
        names = list(world.get_object_names())
        if mplib_name is not None and mplib_name not in names:
            names.insert(0, mplib_name)
        for name in names:
            try:
                if world.is_object_attached(name):
                    world.detach_object(name, also_remove=True)
            except Exception:
                continue

        if self.srmp_planner is not None:
            for name in list(self._srmp_attachment_state):
                try:
                    self.srmp_planner.detach_object(name, also_remove=True)
                except Exception:
                    pass
            self._srmp_attachment_state = {}

    @staticmethod
    def _srmp_pose(pose):
        import srmp

        return srmp.Pose(
            p=np.asarray(pose.p, dtype=np.float64),
            q=np.asarray(pose.q, dtype=np.float64),
        )

    def _clear_srmp_scene_collisions(self):
        if self.srmp_planner is not None:
            if self._srmp_search_ready:
                # SRMP bakes the collision world into its search planner. Rebuild
                # the native interface rather than reusing that stale state.
                self.srmp_planner = self.setup_srmp_planner()
                self._srmp_search_ready = False
                self._srmp_attachment_state = {}
            else:
                for name in self._srmp_scene_object_names:
                    if self.srmp_planner.has_object(name):
                        self.srmp_planner.remove_object(name)
        self._srmp_scene_object_names = []
        self._srmp_collision_world_key = None
        self._srmp_result_cache.clear()

    def set_srmp_collision_actors(self, actors, exclude_actor_ids=None):
        if self.srmp_planner is None:
            return
        excluded = {int(value) for value in (exclude_actor_ids or [])}
        actor_records = []
        for wrapper in actors:
            try:
                scene_id = int(wrapper.per_scene_id[0])
            except Exception:
                scene_id = -1
            if scene_id in excluded:
                continue
            try:
                if not wrapper.get_collision_meshes(to_world_frame=True):
                    continue
            except TypeError:
                if not wrapper.get_collision_meshes():
                    continue
            raw = wrapper._objs[0]
            pose = raw.pose
            actor_records.append(
                (
                    scene_id,
                    tuple(np.round(np.asarray(pose.p), 9)),
                    tuple(np.round(np.asarray(pose.q), 9)),
                    raw,
                )
            )
        world_key = tuple(record[:3] for record in actor_records)
        if world_key == self._srmp_collision_world_key:
            return

        self._clear_srmp_scene_collisions()
        raw_actors = []
        for _, _, _, raw in actor_records:
            shapes = [
                shape
                for component in raw.get_components()
                for shape in getattr(component, "collision_shapes", [])
            ]
            if any(type(shape).__name__ == "PhysxCollisionShapePlane" for shape in shapes):
                continue
            raw_actors.append(raw)
        before = set(self.srmp_planner.get_object_names())
        scene = type("SRMPActorScene", (), {"get_all_actors": lambda _self: raw_actors})()
        self.srmp_planner.read_sim(scene, "sapien")
        self._srmp_scene_object_names = sorted(
            set(self.srmp_planner.get_object_names()) - before
        )
        self._srmp_collision_world_key = world_key

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action = np.hstack([qpos, self.gripper_state])
            # delta_base = action[:3] - self.robot.get_qpos()[0, :3].cpu().numpy()
            # action[:3] = delta_base
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info


    def get_eef_z(self):
        """Helper function for constraint"""
        ee_idx = self.planner.link_name_2_idx[self.planner.move_group]
        ee_pose = self.planner.robot.get_pinocchio_model().get_link_pose(ee_idx)
        mat = transforms3d.quaternions.quat2mat(ee_pose.q)
        return mat[:, 2]

    def get_curr_qpose(self):
        qpose = self.robot.get_qpos().cpu().numpy()[0]
        return qpose

    def make_f(self):
        """
        Create a constraint function that takes in a qpos and outputs a scalar.
        A valid constraint function should evaluates to 0 when the constraint
        is satisfied.

        See [ompl constrained planning](https://ompl.kavrakilab.org/constrainedPlanning.html)
        for more details.
        """

        def f(x, out):
            self.planner.robot.set_qpos(x)
            diff = self.get_eef_z().dot(np.array([0, 0, 1])) - 0.966
            out[0] = (
                0 if diff < 0 else diff
            )  # maintain 15 degrees w.r.t. z axis

        # constraint function ankor end
        return f

    def make_j(self):
        """
        Create the jacobian of the constraint function w.r.t. qpos.
        This is needed because the planner uses the jacobian to project a random sample
        to the constraint manifold.
        """

        # constraint jacobian ankor
        def j(x, out):
            full_qpos = self.planner.pad_move_group_qpos(x)
            jac = self.planner.robot.get_pinocchio_model().compute_single_link_jacobian(
                full_qpos, len(self.planner.move_group_joint_indices) - 1
            )
            rot_jac = jac[3:, self.planner.move_group_joint_indices]
            for i in range(len(self.planner.move_group_joint_indices)):
                out[i] = np.cross(rot_jac[:, i], self.get_eef_z()).dot(
                    np.array([0, 0, 1])
                )

        # constraint jacobian ankor end
        return j


    def set_robot_pose(self, pose: sapien.Pose):
        pose = to_sapien_pose(pose)
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = sapien.Pose(p=pose.p, q=pose.q)
        # get the ik solution for the pose
        mplib_pose = mplib.pymp.Pose(p=pose.p, q=pose.q)
        # get current qpos
        qpos = self.robot.get_qpos().cpu().numpy()[0]

        current_qpos = np.clip(
            qpos, self.planner.joint_limits[:, 0], self.planner.joint_limits[:, 1]
        )
        current_qpos = self.planner.pad_move_group_qpos(current_qpos)
        # wrt world
        mplib_pose = self.planner._transform_goal_to_wrt_base(mplib_pose)

        status, ik_solution = self.planner.IK(mplib_pose, current_qpos, return_closest=True)
        if status == "Success":
            # add the gripper state
            self.robot.set_qpos(ik_solution)
        else:
            print("IK failed")
            return -1


    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, 
        dry_run: bool = False,
        refine_steps: int = 0,
        constrain=False,
        goal_qpos: Optional[np.ndarray] = None,
    ):
        with skill_timer("UR10eMotionPlanner", "motion_request_setup"):
            pose = to_sapien_pose(pose)
            if self.grasp_pose_visual is not None:
                self.grasp_pose_visual.set_pose(pose)
            pose = sapien.Pose(p=pose.p, q=pose.q)
            # result = self.planner.plan_qpos_to_pose(
            #     np.concatenate([pose.p, pose.q]),
            #     self.robot.get_qpos().cpu().numpy()[0],
            #     time_step=self.base_env.control_timestep,
            #     use_point_cloud=self.use_point_cloud,
            #     wrt_world=True,
            # )
            mplib_pose = mplib.pymp.Pose(p=pose.p, q=pose.q)
            current_qpos = self.robot.get_qpos().cpu().numpy()[0]
            request_key = _motion_request_key(
                "plan_pose",
                current_qpos,
                pose=pose,
                extra=(
                    f"backend={self.motion_backend},constrain={bool(constrain)}"
                ),
            )
            constraint_function = self.make_f() if constrain else None
            constraint_jacobian = self.make_j() if constrain else None

        planning_time = 5 if constrain else 10
        planner_method = "plan_pose"
        solve_t0 = perf_counter()
        with skill_timer("UR10eMotionPlanner", "motion_solve"):
            if self.motion_backend == "srmp" and not constrain:
                planner_method = "srmp_plan_pose"
                if goal_qpos is None:
                    result = self._srmp_plan_pose(pose, current_qpos)
                else:
                    result = self._srmp_plan_qpos(
                        [goal_qpos],
                        current_qpos,
                        target_pose=pose,
                    )
            else:
                result = self.planner.plan_pose(
                    mplib_pose,
                    current_qpos,
                    time_step=self.base_env.control_timestep,
                    wrt_world=True,
                    constraint_function=constraint_function,
                    constraint_jacobian=constraint_jacobian,
                    constraint_tolerance=0.05 if constrain else 0.001,
                    planning_time=planning_time,
                )
        configured_planning_time = (
            float(getattr(self, "srmp_time_limit_s", planning_time))
            if planner_method.startswith("srmp_")
            else planning_time
        )
        record_motion_planner_call(
            planner_method,
            _motion_result_status(result),
            perf_counter() - solve_t0,
            planning_time_s=configured_planning_time,
            waypoint_count=_motion_result_waypoints(result),
            request_key=request_key,
            diagnostic=(
                self._last_srmp_failure_detail
                if planner_method.startswith("srmp_")
                else None
            ),
        )
        if planner_method.startswith("srmp_"):
            if getattr(self, "_last_srmp_cache_hit", False):
                record_motion_planner_cache_hit(planner_method)
            else:
                record_motion_planner_cache_miss(planner_method)
        if result["status"] != "Success":
            print(result["status"])
            # self.render_wait()
            return -1
        # self.render_wait()
        if dry_run:
            return result
        with skill_timer("UR10eMotionPlanner", "motion_result_follow"):
            return self.follow_path(result, refine_steps=refine_steps)


    def move_to_qpose_withRRTConnect(self,
                                     qpose: List[np.ndarray],
                                     dry_run: bool = False,
                                     refine_steps: int = 0):
        with skill_timer("UR10eMotionPlanner", "motion_request_setup"):
            current_qpos = self.robot.get_qpos().cpu().numpy()[0]
            request_key = _motion_request_key(
                "plan_qpos",
                current_qpos,
                qpose=qpose,
                extra=f"backend={self.motion_backend}",
            )
        planning_time = self.srmp_time_limit_s if self.motion_backend == "srmp" else 1
        solve_t0 = perf_counter()
        with skill_timer("UR10eMotionPlanner", "motion_solve"):
            if self.motion_backend == "srmp":
                planner_method = "srmp_plan_qpos"
                result = self._srmp_plan_qpos(qpose, current_qpos)
            else:
                planner_method = "plan_qpos"
                result = self.planner.plan_qpos(
                    qpose,
                    current_qpos,
                    time_step=self.base_env.control_timestep,
                    planning_time=planning_time,
                )
        configured_planning_time = (
            float(getattr(self, "srmp_time_limit_s", planning_time))
            if planner_method.startswith("srmp_")
            else planning_time
        )
        record_motion_planner_call(
            planner_method,
            _motion_result_status(result),
            perf_counter() - solve_t0,
            planning_time_s=configured_planning_time,
            waypoint_count=_motion_result_waypoints(result),
            request_key=request_key,
            diagnostic=(
                self._last_srmp_failure_detail
                if planner_method.startswith("srmp_")
                else None
            ),
        )
        if planner_method.startswith("srmp_"):
            if getattr(self, "_last_srmp_cache_hit", False):
                record_motion_planner_cache_hit(planner_method)
            else:
                record_motion_planner_cache_miss(planner_method)
        if result["status"] != "Success":
            print(result["status"])
            # self.render_wait()
            return -1
        # self.render_wait()
        if dry_run:
            return result
        with skill_timer("UR10eMotionPlanner", "motion_result_follow"):
            return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0, wrt_world=True
    ):
        with skill_timer("UR10eMotionPlanner", "motion_request_setup"):
            pose = to_sapien_pose(pose)
            # try screw two times before giving up
            if self.grasp_pose_visual is not None:
                self.grasp_pose_visual.set_pose(pose)
            pose = sapien.Pose(p=pose.p , q=pose.q)
            mplib_pose = mplib.pymp.Pose(p=pose.p, q=pose.q)
            current_qpos = self.robot.get_qpos().cpu().numpy()[0]
            request_key = _motion_request_key(
                "plan_screw",
                current_qpos,
                pose=pose,
                extra=f"wrt_world={bool(wrt_world)}",
            )
        solve_t0 = perf_counter()
        with skill_timer("UR10eMotionPlanner", "motion_solve"):
            if self.motion_backend == "srmp":
                result = self._srmp_plan_screw(
                    pose, current_qpos, wrt_world=wrt_world
                )
                planner_method = "srmp_plan_screw"
            else:
                result = self.planner.plan_screw(
                    mplib_pose,
                    current_qpos,
                    time_step=self.base_env.control_timestep,
                    qpos_step=0.05,
                    wrt_world=wrt_world,
                )
                planner_method = "plan_screw"
        record_motion_planner_call(
            planner_method,
            _motion_result_status(result),
            perf_counter() - solve_t0,
            waypoint_count=_motion_result_waypoints(result),
            request_key=request_key,
            diagnostic=(
                self._last_srmp_failure_detail
                if planner_method.startswith("srmp_")
                else None
            ),
        )
        if result["status"] != "Success" and self.motion_backend != "srmp":
            solve_t0 = perf_counter()
            with skill_timer("UR10eMotionPlanner", "motion_solve"):
                result = self.planner.plan_screw(
                    mplib_pose,
                    current_qpos,
                    time_step=self.base_env.control_timestep,
                    # use_point_cloud=self.use_point_cloud,
                    qpos_step=0.05,
                    wrt_world=wrt_world,
                )
            record_motion_planner_call(
                "plan_screw_retry",
                _motion_result_status(result),
                perf_counter() - solve_t0,
                waypoint_count=_motion_result_waypoints(result),
            )
            if result["status"] != "Success":
                print(result["status"])
                # self.render_wait()
                return -1
        elif result["status"] != "Success":
            print(result["status"])
            return -1
        # self.render_wait()
        if dry_run:
            return result
        with skill_timer("UR10eMotionPlanner", "motion_result_follow"):
            return self.follow_path(result, refine_steps=refine_steps)

    def get_pc_from_env(self,
                        obj,
                        plot=False):
        # check if the object is a mesh or a box or a sphere
        if obj.px_body_type == "static":
            print("Object is not a kinematic or dynamic object")
            raise ValueError("Object is not a kinematic object or has no collision shapes")
        mesh: trimesh.Trimesh or None = get_component_mesh(
            obj._objs[0].find_component_by_type(physx.PhysxRigidDynamicComponent),
            to_world_frame=True
        )

        assert mesh is not None, "can not get actor mesh for {}".format(obj)

        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
        # pose = obj._objs[0].get_pose()
        # # transform the mesh
        # transform_mat = np.eye(4)
        # transform_mat[:3, :3] = quaternions.quat2mat(pose.q)
        # transform_mat[:3, 3] = pose.p
        # mesh_o3d.transform(transform_mat)

        aabb_o3d = mesh_o3d.get_axis_aligned_bounding_box()
        aabb = np.array([[aabb_o3d.min_bound[0], aabb_o3d.min_bound[1], aabb_o3d.min_bound[2]],
                         [aabb_o3d.max_bound[0], aabb_o3d.max_bound[1], aabb_o3d.max_bound[2]]])
        pcd = mesh_o3d.sample_points_uniformly(number_of_points=5000)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # pcd.orient_normals_consistent_tangent_plane(k=100)
        if plot:
            o3d.visualization.draw_geometries([pcd, aabb_o3d],
                                              point_show_normal=True,  # Show the normals as small lines
                                              width=800,
                                              height=600,
                                              mesh_show_back_face=True)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        return points, normals, aabb


    def sample_grasp_ee_poses(self,
                              aabb,
                              pc,
                              n_samples: int,
                              ik_solver,
                              collision_checker,
                              **kwargs):
        ee_samples = []
        obj_min, obj_max = aabb[0, :], aabb[1, :]
        for _ in range(n_samples):
            ee_position = np.random.uniform(obj_min, obj_max, size=3)
            # Keep z at the object's world-frame center so sampled grasp poses stay
            # on the object instead of collapsing near the global origin.
            ee_position[2] = 0.5 * (obj_min[2] + obj_max[2])
            # # if x or y are close to the center, go back towards the edge of the object
            # if np.abs(ee_position[0] - (obj_max[0] - obj_min[0]) / 2) < 0.08:
            #     ee_position[0] = np.random.choice([obj_min[0], obj_max[0]])
            #     ee_position[0] += 0.05 * ((obj_max[0] - obj_min[0]) / 2 - ee_position[0])
            # if np.abs(ee_position[1] - (obj_max[1] - obj_min[1]) / 2) < 0.08:
            #     ee_position[1] = np.random.choice([obj_min[1], obj_max[1]])
            #     ee_position[1] += 0.05 * ((obj_max[1] - obj_min[1]) / 2 - ee_position[1])

            # sample a point from the point cloud
            ee_orientation = np.random.uniform(-np.pi, np.pi, size=3)
            # 1.
            # ee_orientation[:2] *= 0.6
            # ee_pose = sapien.Pose(p=ee_position, q=euler.euler2quat(*ee_orientation))
            # 2.
            # choices = [np.array([0, -np.pi / 2., np.random.choice([-1., 1.]) * np.pi / 2.]),
            #            np.array([np.random.choice([-1., 1.]) * np.pi / 2., 0., 0.])]
            # ee_orientation = choices[np.random.choice([0, 1])]
            # ee_pose = sapien.Pose(p=ee_position, q=euler.euler2quat(*ee_orientation, axes="rxyz"))
            # 3.
            # ee_orientation[2] = 0
            # ee_orientation[1] = -np.pi / 2
            # ee_orientation[0] *= 0.6
            ee_pose = sapien.Pose(p=ee_position, q=euler.euler2quat(*ee_orientation, axes="szyx"))

            # # transform with respect to the base
            base_pose = self.robot.get_pose()
            base_pose = sapien.Pose(p=base_pose.p.squeeze().cpu().numpy(),
                                    q=base_pose.q.squeeze().cpu().numpy())
            ee_pose = base_pose.inv() * ee_pose
            mplib_ee_pose = mplib.pymp.Pose(p=ee_pose.p,
                                            q=ee_pose.q)

            # check if q_init is in kwargs
            if "q_init" not in kwargs:
                kwargs["start_qpos"] = self.robot.get_qpos().cpu().numpy()[0]
            current_qpos = np.clip(
                kwargs["start_qpos"],
                self.planner.joint_limits[:, 0],
                self.planner.joint_limits[:, 1]
            )
            current_qpos = self.planner.pad_move_group_qpos(current_qpos)
            kwargs["start_qpos"] = current_qpos
            kwargs["return_closest"] = True
            # kwargs["verbose"] = True
            with skill_timer("UR10eMotionPlanner", "ik"):
                ik_succ, ik_solution = ik_solver(mplib_ee_pose, **kwargs)
            if ik_succ == "Success": # and not collision_checker(ik_solution): alreasy checks for collisions!
                ee_samples.append((ee_pose, ik_solution))
        assert len(ee_samples) > 0, "\033[91mNo valid grasp poses found\033[0m"
        # print(f"Found {len(ee_samples)} valid grasp poses")
        return ee_samples


    def get_antipodal_score(self,
                            robot_joint_angles: np.ndarray,
                            pc,
                            normals):
        self.planner.pinocchio_model.compute_forward_kinematics(robot_joint_angles)
        tcp_pose = self.planner.pinocchio_model.get_link_pose(self.planner.link_name_2_idx[self.planner.move_group])
        tcp_pose = sapien.Pose(tcp_pose.get_p(), tcp_pose.get_q())
        base_pose = self.robot.get_pose()
        base_pose = sapien.Pose(p=base_pose.p.squeeze().cpu().numpy(),
                                q=base_pose.q.squeeze().cpu().numpy())
        return self.get_antipodal_pose_score(base_pose * tcp_pose, pc, normals)

    def get_antipodal_pose_score(self, tcp_pose: sapien.Pose, pc, normals):
        """Score a proposed world-frame TCP pose without running IK first."""
        tcp_position = tcp_pose.get_p()
        tcp_orientation = tcp_pose.get_q()
        score = 0
        gripper_line_vector = np.array([0.0, 0.2, 0.0])
        tcp_ori_mat = quaternions.quat2mat(tcp_orientation)
        gripper_line_vector = tcp_ori_mat @ gripper_line_vector
        gripper_line_vector /= np.linalg.norm(gripper_line_vector)
        box = o3d.geometry.OrientedBoundingBox(tcp_position,
                                               tcp_ori_mat,
                                               np.array([0.05, 0.4, 0.05]))
        indices = box.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(pc))
        if isinstance(normals, np.ndarray):
            normals = [x for x in normals]
        elif not isinstance(normals, list):
            raise ValueError(f"Invalid type {type(normals)}")
        normals_in_grasp = np.array([normals[i] for i in indices])
        if len(normals_in_grasp) > 0:
            score = np.mean(np.abs(np.dot(normals_in_grasp, gripper_line_vector)))
        return score


    def find_best_grasp(self,
                        obj,
                        n_samples: int = 50
                        ):

        pc, normals, aabb = self.get_pc_from_env(obj)
        poses_conf = self.sample_grasp_ee_poses(aabb,
                                                pc,
                                                n_samples,
                                                self.planner.IK,
                                                self.planner.check_for_collision)
        best_score = -np.inf
        best_grasp = None
        for p in poses_conf:
            score = self.get_antipodal_score(p[-1], pc, normals)
            if score > best_score:
                best_score = score
                best_grasp = p
        if best_grasp is None:
            return None
        base_pose = self.robot.get_pose()
        base_pose = sapien.Pose(p=base_pose.p.squeeze().cpu().numpy(),
                                q=base_pose.q.squeeze().cpu().numpy())
        return base_pose * best_grasp[0], best_grasp[1]

    def find_n_best_grasps(self,
                           obj,
                           n_samples: int = 50,
                           n_grasps: int = 10):
        pc, normals, aabb = self.get_pc_from_env(obj)
        base_pose = self.robot.get_pose()
        base_pose = sapien.Pose(p=base_pose.p.squeeze().cpu().numpy(),
                                q=base_pose.q.squeeze().cpu().numpy())
        obj_min, obj_max = aabb[0, :], aabb[1, :]

        # Preserve the random pose distribution, but rank poses before invoking
        # MPLib IK. Previously all samples ran IK before all but ten were dropped.
        scored_poses = []
        for _ in range(int(max(0, n_samples))):
            ee_position = np.random.uniform(obj_min, obj_max, size=3)
            ee_position[2] = 0.5 * (obj_min[2] + obj_max[2])
            ee_orientation = np.random.uniform(-np.pi, np.pi, size=3)
            world_pose = sapien.Pose(
                p=ee_position,
                q=euler.euler2quat(*ee_orientation, axes="szyx"),
            )
            score = self.get_antipodal_pose_score(world_pose, pc, normals)
            scored_poses.append((float(score), base_pose.inv() * world_pose))
        scored_poses.sort(key=lambda item: item[0], reverse=True)

        start_qpos = self.robot.get_qpos().cpu().numpy()[0]
        start_qpos = np.clip(
            start_qpos,
            self.planner.joint_limits[:, 0],
            self.planner.joint_limits[:, 1],
        )
        start_qpos = self.planner.pad_move_group_qpos(start_qpos)

        grasps = []
        for _score, ee_pose in scored_poses:
            mplib_ee_pose = mplib.pymp.Pose(p=ee_pose.p, q=ee_pose.q)
            with skill_timer("UR10eMotionPlanner", "ik"):
                ik_succ, ik_solution = self.planner.IK(
                    mplib_ee_pose,
                    start_qpos=start_qpos,
                    return_closest=True,
                )
            if ik_succ != "Success":
                continue
            grasps.append((base_pose * ee_pose, ik_solution))
            if len(grasps) >= int(max(1, n_grasps)):
                break
        return grasps


    def open_gripper(self):
        self.gripper_state = OPEN
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()

        for i in range(6):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def control_gripper(self, t=6, gripper_state = CLOSED):
        self.gripper_state = gripper_state
        # self.render_wait()
        qpos = self.robot.get_qpos()[0, :-6].cpu().numpy()
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            elif self.control_mode == "pd_joint_delta_pos":
                action = np.hstack([qpos * 0, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        # self.render_wait()
        return obs, reward, terminated, truncated, info

    def add_box_collision(self, extents: np.ndarray, pose: sapien.Pose):
        self.use_point_cloud = True
        box = trimesh.creation.box(extents, transform=pose.to_transformation_matrix())
        pts, _ = trimesh.sample.sample_surface(box, 256)
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def add_collision_pts(self, pts: np.ndarray):
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.remove_point_cloud()
        self.planner.update_point_cloud(self.all_collision_pts)

    def clear_collisions(self):
        self.all_collision_pts = None
        self.use_point_cloud = False
        self.planner.remove_point_cloud()
        self._clear_srmp_scene_collisions()

    def close(self):
        # Closing discards the native interface, so rebuilding it is unnecessary.
        self._srmp_search_ready = False
        self._clear_srmp_scene_collisions()
        self.srmp_planner = None
        for path in self._srmp_mesh_paths.values():
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        self._srmp_mesh_paths.clear()



def build_panda_gripper_grasp_pose_visual(scene: ManiSkillScene):
    builder = scene.create_actor_builder()
    grasp_pose_visual_width = 0.01
    grasp_width = 0.05

    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0.0]),
        radius=grasp_pose_visual_width,
        material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
    )

    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, 0.08]),
        half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, 0.05]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                grasp_width + grasp_pose_visual_width,
                0.05 - 0.03,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                -grasp_width - grasp_pose_visual_width,
                0.05 - 0.03,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
    )
    grasp_pose_visual = builder.build_kinematic(name="grasp_pose_visual")
    return grasp_pose_visual
