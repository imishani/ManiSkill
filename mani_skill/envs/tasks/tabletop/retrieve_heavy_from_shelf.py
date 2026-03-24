"""
RetrieveHeavyFromShelf-v1
=========================
A large/heavy object sits at the back of a shelf level surrounded by smaller
clutter objects.  The robot must push aside the clutter, grasp the heavy
target, and carry it to a delivery cart beside the shelf.

Designed for the RidgebackUR10e (mobile single-arm), with the architecture
kept flexible for future bimanual or multi-robot extensions.
"""

from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.ridgeback_ur10e.ridgeback_ur10e import RidgebackUR10e
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.actors.ycb import get_ycb_builder
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.shelf import ShelfSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

# ---------------------------------------------------------------------------
# Curated object lists
# ---------------------------------------------------------------------------

# Large, heavy, or bulky YCB objects that serve as the retrieval target.
# These have large footprints or high mass, making them non-trivial to grasp.
HEAVY_OBJECT_IDS: List[str] = [
    "021_bleach_cleanser",   # tall bottle, ~1.9 kg
    "003_cracker_box",       # large box
    "004_sugar_box",         # large box
    "006_mustard_bottle",    # tall bottle, ~1.2 kg
    "019_pitcher_base",      # tall pitcher, bulky
    "036_wood_block",        # dense wooden block
    "035_power_drill",       # heavy tool, ~2.1 kg
    "048_hammer",            # heavy tool, ~1.5 kg
]

# Smaller objects that act as shelf clutter.  They must be pushed aside or
# navigated around before the target can be grasped.
CLUTTER_OBJECT_IDS: List[str] = [
    "005_tomato_soup_can",
    "007_tuna_fish_can",
    "010_potted_meat_can",
    "002_master_chef_can",
    "025_mug",
    "065-a_cups",
    "065-b_cups",
    "062_dice",
    "008_pudding_box",
    "009_gelatin_box",
    "013_apple",
    "014_lemon",
    "016_pear",
    "011_banana",
]


@register_env("RetrieveHeavyFromShelf-v1", max_episode_steps=300, asset_download_ids=["ycb"])
class RetrieveHeavyFromShelfEnv(BaseEnv):
    """
    **Task Description:**
    Retrieve a large/heavy YCB object from a shelf that is cluttered with
    smaller objects and place it on a nearby delivery cart.

    **Randomizations:**
    - Target object identity sampled from ``HEAVY_OBJECT_IDS`` (per reconfiguration).
    - Clutter object identities sampled from ``CLUTTER_OBJECT_IDS`` (per reconfiguration).
    - Clutter object positions randomized on the shelf in front of the target.
    - Goal position randomized on the delivery cart surface.

    **Success Condition:**
    - Target object centroid is within ``goal_thresh`` of the goal site.

    **Extensibility notes:**
    - ``num_clutter`` controls how many clutter objects are placed (default 4).
    - ``add_back_panel`` on the ShelfSceneBuilder toggles the back wall.
    - The success condition and goal can be swapped for a simpler "lift out of
      shelf" condition by changing ``evaluate()``.
    - Bimanual support: add a second robot uid to ``SUPPORTED_ROBOTS`` and
      update ``_initialize_episode`` / ``evaluate`` accordingly.
    """

    SUPPORTED_REWARD_MODES = ["none", "dense"]
    SUPPORTED_ROBOTS = ["ridgebackur10e"]
    agent: RidgebackUR10e

    goal_thresh: float = 0.12  # success distance threshold (m)

    def __init__(
        self,
        *args,
        robot_uids: str = "ridgebackur10e",
        robot_init_qpos_noise: float = 0.02,
        num_clutter: int = 4,
        reconfiguration_freq=None,
        num_envs: int = 1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.num_clutter = num_clutter

        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0

        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Sim / sensor config
    # ------------------------------------------------------------------

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21, max_rigid_patch_count=2**19
            )
        )

    @property
    def _default_sensor_configs(self):
        # Side-on camera giving a view into the shelf
        pose = sapien_utils.look_at(
            eye=[-0.2, -1.5, 1.2], target=[1.0, 0.0, 0.65]
        )
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 3,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[-0.5, -2.0, 1.5], target=[1.0, 0.0, 0.65])
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=1280,
            height=720,
            fov=np.pi / 3,
            near=0.01,
            far=100,
        )

    # ------------------------------------------------------------------
    # Agent loading
    # ------------------------------------------------------------------

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0.0, 0.0, 0.0]))

    # ------------------------------------------------------------------
    # Scene building
    # ------------------------------------------------------------------

    def _load_scene(self, options: dict):
        self.scene_builder = ShelfSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build(add_back_panel=False)

        # --- Target (heavy) objects, one per sub-scene -------------------
        heavy_ids = self._batched_episode_rng.choice(HEAVY_OBJECT_IDS, replace=True)
        self._target_objs: List[Actor] = []
        for i, model_id in enumerate(heavy_ids):
            builder = get_ycb_builder(self.scene, id=model_id)
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.set_scene_idxs([i])
            self._target_objs.append(
                builder.build(name=f"target_{model_id}_{i}")
            )
        self.target_obj = Actor.merge(self._target_objs, name="target_object")

        # --- Clutter objects, num_clutter per sub-scene ------------------
        self._clutter_objs: List[Actor] = []
        for i in range(self.num_envs):
            clutter_ids = self._batched_episode_rng[i].choice(
                CLUTTER_OBJECT_IDS,
                size=self.num_clutter,
                replace=len(CLUTTER_OBJECT_IDS) < self.num_clutter,
            )
            for j, model_id in enumerate(clutter_ids):
                builder = get_ycb_builder(self.scene, id=model_id)
                builder.initial_pose = sapien.Pose(p=[0, 0, 0])
                builder.set_scene_idxs([i])
                self._clutter_objs.append(
                    builder.build(name=f"clutter_{i}_{j}_{model_id}")
                )
        if self._clutter_objs:
            self.clutter_objs = Actor.merge(self._clutter_objs, name="clutter_objects")
        else:
            self.clutter_objs = None

        # --- Goal site (invisible sphere over delivery cart) -------------
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 0.3],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _after_reconfigure(self, options: dict):
        """Cache per-object z-offsets so each object rests flat on the shelf."""
        self._target_obj_zs: List[float] = []
        for obj in self._target_objs:
            mesh = obj.get_first_collision_mesh()
            self._target_obj_zs.append(float(-mesh.bounding_box.bounds[0, 2]))
        self._target_obj_zs_t = common.to_tensor(self._target_obj_zs, device=self.device)

        self._clutter_obj_zs: List[float] = []
        for obj in self._clutter_objs:
            mesh = obj.get_first_collision_mesh()
            self._clutter_obj_zs.append(float(-mesh.bounding_box.bounds[0, 2]))
        self._clutter_obj_zs_t = (
            common.to_tensor(self._clutter_obj_zs, device=self.device)
            if self._clutter_obj_zs
            else None
        )

    # ------------------------------------------------------------------
    # Episode initialization
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)

            sb = self.scene_builder  # shorthand
            shelf_z = sb.target_shelf_surface_z
            sfx = sb.SHELF_FRONT_X
            sd = sb.SHELF_DEPTH
            pt = sb.PANEL_THICKNESS
            inner_hw = sb.shelf_inner_half_width   # half the inner y-span

            # -----------------------------------------------------------------
            # Place target object near the back-center of the shelf
            # -----------------------------------------------------------------
            # x: back third of shelf interior, small random offset
            # y: small random offset from shelf center
            # z: shelf surface + object z-offset (so it rests flat)
            target_x_range = sd * 0.25  # ±range around back-center
            target_back_cx = sfx + sd * 0.75  # 3/4 back

            for local_i, global_i in enumerate(env_idx.tolist()):
                target_z_off = float(self._target_obj_zs_t[global_i])
                target_xyz = torch.zeros(3)
                target_xyz[0] = (
                    target_back_cx
                    + (torch.rand(1).item() * 2 - 1) * target_x_range * 0.5
                )
                target_xyz[1] = (torch.rand(1).item() * 2 - 1) * inner_hw * 0.3
                target_xyz[2] = shelf_z + target_z_off + 1e-3
                q = random_quaternions(1, lock_x=True, lock_y=True)
                self._target_objs[global_i].set_pose(
                    Pose.create_from_pq(target_xyz.unsqueeze(0), q)
                )

            # -----------------------------------------------------------------
            # Place clutter objects in front of the target
            # -----------------------------------------------------------------
            # Distribute evenly in x (front portion) and y (spread across width)
            if self._clutter_objs:
                clutter_per_env = self.num_clutter
                for local_i, global_i in enumerate(env_idx.tolist()):
                    clutter_start = global_i * clutter_per_env
                    for j in range(clutter_per_env):
                        obj_idx = clutter_start + j
                        if obj_idx >= len(self._clutter_objs):
                            break
                        clutter_z_off = float(self._clutter_obj_zs_t[obj_idx])

                        # x: front half of shelf
                        frac_x = (j + 0.5) / clutter_per_env  # 0..1 in front half
                        cx = sfx + pt + (sd * 0.5) * frac_x
                        cx += (torch.rand(1).item() * 2 - 1) * 0.03

                        # y: spread across inner width with small jitter
                        frac_y = (j / max(clutter_per_env - 1, 1)) * 2 - 1
                        cy = frac_y * inner_hw * 0.8
                        cy += (torch.rand(1).item() * 2 - 1) * 0.04

                        cz = shelf_z + clutter_z_off + 1e-3
                        pos = torch.tensor([[cx, cy, cz]])
                        q = random_quaternions(1, lock_x=True, lock_y=True)
                        self._clutter_objs[obj_idx].set_pose(
                            Pose.create_from_pq(pos, q)
                        )

            # -----------------------------------------------------------------
            # Goal site: random position on delivery cart surface
            # -----------------------------------------------------------------
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = (
                sb.SHELF_FRONT_X
                + (torch.rand(b) * 2 - 1) * (sb.CART_WIDTH / 2 - 0.05)
            )
            goal_xyz[:, 1] = (
                sb.SHELF_WIDTH / 2 + sb.CART_Y_GAP + sb.CART_DEPTH / 2
                + (torch.rand(b) * 2 - 1) * (sb.CART_DEPTH / 2 - 0.05)
            )
            goal_xyz[:, 2] = sb.cart_surface_z + 0.05
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            # -----------------------------------------------------------------
            # Robot: position in front of shelf, arm in a neutral reaching pose
            # -----------------------------------------------------------------
            if self.robot_uids == "ridgebackur10e":
                # Base joints: x, y, theta (holonomic base)
                # Arms joints: UR10e 6-DOF
                # Gripper: 2 active + 4 passive
                qpos = np.array(
                    [
                        0.0, 0.0, 0.0,               # base: at world origin, facing +x
                        0.0, -1.0472, -2.0, 0.0, 1.5708, 0.0,  # arm (ready pose)
                        0.0, 0.0,                    # gripper open
                        0.0, 0.0, 0.0, 0.0,          # passive joints
                    ]
                )
                self.agent.reset(qpos)
                # Place robot on the ground (z=0) in front of the shelf
                self.agent.robot.set_root_pose(sapien.Pose(p=[0.0, 0.0, 0.0]))
            else:
                raise NotImplementedError(f"Robot uid '{self.robot_uids}' not supported.")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self):
        obj_to_goal = self.goal_site.pose.p - self.target_obj.pose.p
        dist = torch.linalg.norm(obj_to_goal, dim=-1)
        is_obj_placed = dist <= self.goal_thresh
        is_grasped = self.agent.is_grasping(self.target_obj)
        return dict(
            is_grasped=is_grasped,
            obj_to_goal_dist=dist,
            is_obj_placed=is_obj_placed,
            success=is_obj_placed,
        )

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_goal_pos=self.goal_site.pose.p - self.agent.tcp.pose.p,
                obj_pose=self.target_obj.pose.raw_pose,
                tcp_to_obj_pos=self.target_obj.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.target_obj.pose.p,
            )
            # clutter_poses is intentionally excluded from the flattened obs
            # because the merged actor spans num_clutter × num_envs entries which
            # is incompatible with the per-env tensor stacking.  Access via
            # self.clutter_objs.pose directly if needed by a custom policy.
        return obs

    # ------------------------------------------------------------------
    # Reward  (minimal – task is intended for planning/skill-based methods)
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.target_obj.pose.p - self.agent.tcp.pose.p, dim=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped.float()

        obj_to_goal_dist = info["obj_to_goal_dist"]
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped.float()

        reward += info["is_obj_placed"].float() * is_grasped.float()

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], dim=1)
        )
        reward += static_reward * info["is_obj_placed"].float() * is_grasped.float()

        reward[info["success"]] = 6.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.0
