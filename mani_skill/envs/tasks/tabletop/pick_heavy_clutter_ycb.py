import os
from typing import Dict, List, Union, Any

import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Fetch, RidgebackUR10e
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.building import actors
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.envs.utils.randomization.pose import random_quaternions


#
class PickHeavyClutterEnv(BaseEnv):
    """Base environment picking items out of clutter type of tasks. Flexibly supports using different configurations and object datasets"""

    SUPPORTED_REWARD_MODES = ["none", "dense"]
    SUPPORTED_ROBOTS = ["ridgebackur10e", "fetch"]
    agent: Union[RidgebackUR10e, Fetch]

    DEFAULT_EPISODE_JSON: str
    DEFAULT_ASSET_ROOT: str
    DEFAULT_MODEL_JSON: str

    def __init__(
        self,
        *args,
        robot_uids="ridgebackur10e",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        reconfiguration_freq=None,
        episode_json: str = None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise

        if episode_json is None:
            episode_json = self.DEFAULT_EPISODE_JSON
        if not os.path.exists(episode_json):
            raise FileNotFoundError(
                f"Episode json ({episode_json}) is not found."
                "To download default json:"
                "`python -m mani_skill.utils.download_asset pick_clutter_ycb`."
            )
        self._episodes: List[Dict] = load_json(episode_json)
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21, max_rigid_patch_count=2**19
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 2.0, 2.0], [-0.12, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=2560, height=2560, fov=1, near=0.01, far=100
        )

    def _load_model(self, model_id: str) -> ActorBuilder:
        raise NotImplementedError()

    def _load_agent(self, options: dict):
        # super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))
        super()._load_agent(options, sapien.Pose(p=[0., 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build(scale=1.5)
        # self.scene_builder.build()

        # sample some clutter configurations
        eps_idxs = self._batched_episode_rng.randint(0, len(self._episodes))

        self.selectable_target_objects: List[List[Actor]] = []
        """for each sub-scene, a list of objects that can be selected as targets"""
        self._objs = []

        for i, eps_idx in enumerate(eps_idxs):
            self.selectable_target_objects.append([])
            episode = self._episodes[eps_idx]
            for actor_config in episode["actors"]:
                builder = self._load_model(actor_config["model_id"])
                init_pose = actor_config["pose"]
                builder.initial_pose = sapien.Pose(p=init_pose[:3], q=init_pose[3:])
                builder.set_scene_idxs([i])
                obj = builder.build(name=f"set_{i}_{actor_config['model_id']}")
                self._objs.append(obj)
                if actor_config["rep_pts"] is not None:
                    # rep_pts is representative points, representing visible points
                    # we only permit selecting target objects that are visible
                    self.selectable_target_objects[-1].append(obj)

        self.all_objects = Actor.merge(self._objs, name="all_objects")

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=0.08,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

        self._sample_target_objects()

    def _sample_target_objects(self):
        # note this samples new target objects for every sub-scene
        target_objects = []
        for i in range(self.num_envs):
            selected_obj_idxs = torch.randint(low=0, high=99999, size=(self.num_envs,))
            selected_obj_idxs[i] = selected_obj_idxs[i] % len(
                self.selectable_target_objects[-1]
            )
            target_objects.append(
                self.selectable_target_objects[-1][selected_obj_idxs[i]]
            )
        self.target_object = Actor.merge(target_objects, name="target_object")
        self.model_id = self.target_object.name

    # def _after_reconfigure(self, options: dict):
    #     self.object_zs = []
    #     self.object_meshes = []
    #     for obj in self._objs:
    #         collision_mesh = obj.get_first_collision_mesh()
    #         # this value is used to set object pose so the bottom is at z=0
    #         self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
    #         self.object_meshes.append(collision_mesh)
    #     self.object_zs = common.to_tensor(self.object_zs, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)
            goal_pos = torch.rand(size=(b, 3)) * torch.tensor(
                [-0.6, 0.5, 0.1]
            ) + torch.tensor([-0.15, -0.25, 0.35])
            self.goal_pos = goal_pos
            ori = random_quaternions(b, lock_x=True, lock_y=True)
            self.goal_site.set_pose(Pose.create_from_pq(self.goal_pos, ori))

            # reset objects to original poses
            if b == self.num_envs:
                # if all envs reset
                self.all_objects.pose = self.all_objects.initial_pose
            else:
                # if only some envs reset, we unfortunately still have to do some mask wrangling
                mask = torch.isin(self.all_objects._scene_idxs, env_idx)
                self.all_objects.pose = self.all_objects.initial_pose[mask]

            # Initialize robot arm to a higher position above the table than the default typically used for other table top tasks
            if self.robot_uids == "panda" or self.robot_uids == "panda_wristcam":
                # fmt: off
                qpos = np.array(
                    [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
                )
                # fmt: on
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))
            elif self.robot_uids == "xmate3_robotiq":
                qpos = np.array([0, 0.6, 0, 1.3, 0, 1.3, -1.57, 0, 0])
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.562, 0, 0]))
            elif self.robot_uids == "ridgebackur10e":
                qpos = np.array(
                    [-1.5, 0, 0,
                     0., -1.0472, -2., 0., 1.5708, 0.,
                     0., 0.,
                     0, 0, 0, 0  # Passive joints for the gripper
                     ]
                )
                # # self.agent.reset(qpos)
                # # randomize the initial base pose of the robot, around the center of the table
                # radius = np.sqrt(self.table_scene.table_length ** 2 + self.table_scene.table_width ** 2) / 2
                # angle = self._episode_rng.uniform(0, 2 * np.pi)
                # x_pos = self.table_scene.table.pose.p[0][0] + radius * np.cos(angle)
                # y_pos = self.table_scene.table.pose.p[0][1] + radius * np.sin(angle)
                # # look at the center of the table
                # theta = np.arctan2(
                #     self.table_scene.table.pose.p[0][1] - y_pos,
                #     self.table_scene.table.pose.p[0][0] - x_pos
                # )
                # qpos[0] = x_pos
                # qpos[1] = y_pos
                # qpos[2] = theta

                self.agent.reset(qpos)

                # self.agent.robot.set_root_pose(sapien.Pose([x_pos, y_pos, -self.table_scene.table_height],
                #                                            euler2quat(0, 0, theta)))
            elif self.robot_uids == "static_ridgebackur10e":
                qpos = np.array(
                    [0., -1.0472, -2., 0., 1.5708, 0.,
                     0., 0.,
                     0, 0, 0, 0  # Passive joints for the gripper
                     ]
                )

                # # randomize the initial base pose of the robot, around the center of the table
                # radius = np.sqrt(self.table_scene.table_length ** 2 + self.table_scene.table_width ** 2) / 2
                # angle = self._episode_rng.uniform(0, 2 * np.pi)
                # x_pos = self.table_scene.table.pose.p[0][0] + radius * np.cos(angle)
                # y_pos = self.table_scene.table.pose.p[0][1] + radius * np.sin(angle)
                # # look at the center of the table
                # theta = np.arctan2(
                #     self.table_scene.table.pose.p[0][1] - y_pos,
                #     self.table_scene.table.pose.p[0][0] - x_pos
                # )
                # qpos[0] = x_pos
                # qpos[1] = y_pos
                # qpos[2] = theta

                self.agent.reset(qpos)
                # self.agent.robot.set_root_pose(sapien.Pose([x_pos, y_pos, -self.table_scene.table_height],
                #                                            euler2quat(0, 0, theta)))
            else:
                raise NotImplementedError(self.robot_uids)

    def evaluate(self):
        obj_to_goal_pos = self.goal_site.pose.p - self.target_object.pose.p
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, dim=-1) <= 0.08
        is_grasped = self.agent.is_grasping(self.target_object)
        return dict(
            is_grasped=is_grasped,
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            # is_robot_static=is_robot_static,
            is_grasping=self.agent.is_grasping(self.target_object),
            # success=torch.logical_and(is_obj_placed, is_robot_static),
            success=is_obj_placed,
        )

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_goal_pos=self.goal_site.pose.p - self.agent.tcp.pose.p,
                obj_pose=self.target_object.pose.raw_pose,
                tcp_to_obj_pos=self.target_object.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.target_object.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.target_object.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.target_object.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        reward += info["is_obj_placed"] * is_grasped

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward += static_reward * info["is_obj_placed"] * is_grasped

        reward[info["success"]] = 6
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6


@register_env(
    "PickHeavyClutterYCB-v1",
    asset_download_ids=["ycb", "pick_clutter_ycb_configs"],
    max_episode_steps=100,
)
class PickHeavyClutterYCBEnv(PickHeavyClutterEnv):
    # DEFAULT_EPISODE_JSON = f"{ASSET_DIR}/tasks/pick_clutter/ycb_train_5k.json.gz"
    MANISKILL_PATH = os.path.dirname(os.path.abspath(__file__)) + "/../../../.."
    DEFAULT_EPISODE_JSON = f"{MANISKILL_PATH}/tests/kitchen_item_permutations.json"
    # _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickClutterYCB-v1_rt.mp4"

    def _load_model(self, model_id):
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
        return builder
