from pathlib import Path
from typing import List, Optional
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
        skill_timer,
    )
except Exception:  # pragma: no cover - ManiSkill example may run standalone.
    def skill_timer(skill_name, section_name):
        return nullcontext()

    def record_motion_planner_call(*args, **kwargs):
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
        srmp_planner_id: str = "wAstar",
        srmp_time_limit_s: float = 10.0,
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
        self.srmp_planner_id = str(srmp_planner_id)
        self.srmp_time_limit_s = float(max(0.01, srmp_time_limit_s))
        self.srmp_planner = None
        self._srmp_robot_name = "static_ridgebackur10e"
        self._last_srmp_failure_detail = None

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
        planner.add_robot(
            str(urdf_path),
            name=self._srmp_robot_name,
            srdf_path=str(srdf_path),
            end_effector=self.env_agent.ee_link_name,
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

        planner.set_base_pose(
            self._srmp_robot_name,
            srmp.Pose(p=np.asarray(self.base_pose.p), q=np.asarray(self.base_pose.q)),
        )
        planner.make_planner(
            [self._srmp_robot_name],
            {
                "planner_id": self.srmp_planner_id,
                "heuristic": "bfs",
                "weight": "10.0",
                "resolution": "0.1",
                "time_limit": str(self.srmp_time_limit_s),
            },
        )
        return planner

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

    def _srmp_plan_qpos(self, goals, current_qpos) -> dict:
        import srmp

        self._last_srmp_failure_detail = None
        start = self._move_group_qpos(current_qpos)
        goal_list = [self._move_group_qpos(goal) for goal in goals]
        for goal in goal_list:
            # SRMP 0.1.3.6 segfaults on an already-satisfied joint goal.
            if np.allclose(start, goal, atol=1e-8, rtol=0.0):
                return self._time_parameterize_srmp_path(start[None, :])

        self.srmp_planner.set_qpos(self._srmp_robot_name, start)
        if self.srmp_planner.is_state_colliding(self._srmp_robot_name):
            return {"status": "Invalid start state"}

        valid_goals = []
        invalid_details = []
        for goal in goal_list:
            self.srmp_planner.set_qpos(self._srmp_robot_name, goal)
            if self.srmp_planner.is_state_colliding(self._srmp_robot_name):
                invalid_details.append(
                    {
                        "goal": np.round(goal, 4).tolist(),
                        "self_distance": float(
                            self.srmp_planner.distance_to_self_collision()
                        ),
                        "environment_distance": float(
                            self.srmp_planner.distance_to_robot_collision()
                        ),
                    }
                )
            else:
                valid_goals.append(goal)
        self.srmp_planner.set_qpos(self._srmp_robot_name, start)
        if not valid_goals:
            self._last_srmp_failure_detail = {
                "reason": "invalid_goal_state",
                "goals": invalid_details,
            }
            return {"status": "Invalid goal state"}

        try:
            trajectory = self.srmp_planner.plan(
                start,
                srmp.GoalConstraint(srmp.GoalType.JOINTS, valid_goals),
            )
        except RuntimeError as exc:
            self._last_srmp_failure_detail = {
                "reason": "planning_error",
                "error": str(exc),
                "start": np.round(start, 4).tolist(),
                "goals": [np.round(goal, 4).tolist() for goal in valid_goals],
            }
            return {"status": f"SRMP planning error: {exc}"}
        positions = np.asarray(trajectory.positions, dtype=np.float64)
        if positions.size == 0:
            return {"status": "SRMP failed to find a path"}
        return self._time_parameterize_srmp_path(positions)

    def _srmp_plan_pose(self, pose, current_qpos) -> dict:
        mplib_pose = mplib.pymp.Pose(p=pose.p, q=pose.q)
        mplib_pose = self.planner._transform_goal_to_wrt_base(mplib_pose)
        padded_start = self.planner.pad_move_group_qpos(current_qpos)
        # SRMP 0.1.3.6 currently accepts one JOINTS goal. Use the closest
        # collision-free IK solution until MULTI_GOAL_JOINTS is supported.
        status, solutions = self.planner.IK(
            mplib_pose,
            padded_start,
            return_closest=True,
        )
        if status != "Success":
            return {"status": status}
        solutions = np.asarray(solutions, dtype=np.float64)
        if solutions.ndim == 1:
            solutions = solutions[None, :]
        return self._srmp_plan_qpos(solutions, current_qpos)

    def _sync_srmp_point_cloud(self):
        if self.srmp_planner is None:
            return
        name = "mosaic_collision_cloud"
        if self.srmp_planner.has_object(name):
            self.srmp_planner.remove_object(name)
        if self.all_collision_pts is not None and len(self.all_collision_pts) > 0:
            self.srmp_planner.add_point_cloud(
                name,
                np.asarray(self.all_collision_pts, dtype=np.float64),
                resolution=0.001,
            )

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
        constrain=False
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
                result = self._srmp_plan_pose(pose, current_qpos)
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
        record_motion_planner_call(
            planner_method,
            _motion_result_status(result),
            perf_counter() - solve_t0,
            planning_time_s=planning_time,
            waypoint_count=_motion_result_waypoints(result),
            request_key=request_key,
        )
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
        record_motion_planner_call(
            planner_method,
            _motion_result_status(result),
            perf_counter() - solve_t0,
            planning_time_s=planning_time,
            waypoint_count=_motion_result_waypoints(result),
            request_key=request_key,
        )
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
            result = self.planner.plan_screw(
                mplib_pose,
                current_qpos,
                time_step=self.base_env.control_timestep,
                qpos_step=0.05,
                # use_point_cloud=self.use_point_cloud,
                wrt_world=wrt_world,
            )
        record_motion_planner_call(
            "plan_screw",
            _motion_result_status(result),
            perf_counter() - solve_t0,
            waypoint_count=_motion_result_waypoints(result),
            request_key=request_key,
        )
        if result["status"] != "Success":
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
        self._sync_srmp_point_cloud()

    def add_collision_pts(self, pts: np.ndarray):
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.remove_point_cloud()
        self.planner.update_point_cloud(self.all_collision_pts)
        self._sync_srmp_point_cloud()

    def clear_collisions(self):
        self.all_collision_pts = None
        self.use_point_cloud = False
        self.planner.remove_point_cloud()
        self._sync_srmp_point_cloud()

    def close(self):
        pass



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
