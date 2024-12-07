from typing import Any, Dict, List, Optional, Union, Sequence

import numpy as np
import sapien
import sapien.physx as physx
import torch
import trimesh
import os.path as osp
from pathlib import Path

from transforms3d.euler import euler2quat
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch, RidgebackUR10e
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors import BaseSensorConfig
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.scene_builder.kitchen_counter import KitchenCounterSceneBuilder
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Articulation, Link, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

import inspect


class BarTablesSceneBuilder(SceneBuilder):

    def build(self):
        N_bar_tables = 9
        self.tables = {}
        # get the path to where TableSceneBuilder is

        path_to_table_scene_builder = osp.dirname(inspect.getfile(TableSceneBuilder))
        model_dir = Path(path_to_table_scene_builder) / "assets"
        table_model_file = str(model_dir / "table.glb")
        scale = 1.75
        # table_position = [-0.12, 0, -0.9196429]
        table_position = [-2.4178784, 0, -0.9196429]
        # U shape -- table 1, 2  are on the center, table 3, 4 are on the left, table 5, 6 are on the right
        for i in range(N_bar_tables):
            builder = self.scene.create_actor_builder()
            if i < 3:
                table_pose = sapien.Pose(q=euler2quat(0, 0, 0))
            # elif i < 4:
            #     table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
            else:
                table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))

            builder.add_box_collision(
                pose=sapien.Pose(p=[0, 0, 0.9196429 / 2]),
                half_size=(1.209 / 2, 2.418 / 2, 0.9196429 / 2),
            )
            builder.add_visual_from_file(
                filename=table_model_file, scale=[scale] * 3, pose=sapien.Pose(q=euler2quat(0, 0, 0))#table_pose
            )
            position = table_position.copy()
            if i == 0:
                position[1] += 0
            elif i == 1:
                position[1] += table_width
            elif i == 2:
                position[1] -= table_width
            elif i == 3:
                position[0] += table_length / 2.
                position[1] += table_width + table_length
            elif i == 4:
                position[0] += ((table_length / 2.) + table_width)
                position[1] += table_width + table_length
            elif i == 5:
                position[0] += ((table_length / 2.) + 2 * table_width)
                position[1] += table_width + table_length
            elif i == 6:
                position[0] += table_length / 2.
                position[1] -= table_width + table_length
            elif i == 7:
                position[0] += ((table_length / 2.) + table_width)
                position[1] -= table_width + table_length
            elif i == 8:
                position[0] += ((table_length / 2.) + 2 * table_width)
                position[1] -= table_width + table_length
            else:
                raise NotImplementedError
            builder.initial_pose = sapien.Pose(p=position, q=table_pose.q)
            table = builder.build_kinematic(name=f"table-{i}")
            self.tables[f"table-{i}"] = table
            if i == 0:
                aabb = (
                    table._objs[0]
                    .find_component_by_type(sapien.render.RenderBodyComponent)
                    .compute_global_aabb_tight()
                )
                table_length = aabb[1, 0] - aabb[0, 0]
                table_width = aabb[1, 1] - aabb[0, 1]
                table_height = aabb[1, 2] - aabb[0, 2]

        self.table_length = table_length
        self.table_width = table_width
        self.table_height = table_height

    def initialize(self, env_idx: torch.Tensor):
        pass


@register_env("BarWaiter-v1")
class BarWaiterEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["fetch", "ridgebackur10e"]
    agent = Union[Fetch, RidgebackUR10e]

    cube_half_size = 0.02

    def __init__(self,
                 *args,
                 robot_uids="ridgebackur10e",
                 robot_init_qpos_noise=0.02,
                 reconfiguration_freq=None,
                 num_envs=1,
                 **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if reconfiguration_freq is None:
            # if not user set, we pick a number
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(),
            spacing=15
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
        pose = sapien_utils.look_at([3.0, -7.5, 2.5], [3.0, 0.0, 1.0])
        return CameraConfig(
            "render_camera", pose, 2048, 2048, 60 * np.pi / 180, 0.01, 100
        )

    @property
    def _default_viewer_camera_config(self):
        return CameraConfig(
            uid="viewer",
            pose=sapien.Pose([0, 0, 1]),
            width=1920,
            height=1080,
            shader_pack="default",
            near=0.0,
            far=1000,
            fov=60 * np.pi / 180,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, sapien.Pose(p=[4, 4, 0.])
        )

    def _load_scene(self, options: dict):
        # self.ground = build_ground(self.scene, floor_width=400)
        self.bar_tables_scene = BarTablesSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.bar_tables_scene.build()
        floor_width = 400
        if self.scene.parallel_in_single_scene:
            floor_width = 1000

        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-self.bar_tables_scene.table_height
        )

        self._load_kitchen_counter()

        self._load_shelf()

        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[-2.4178784, 0, self.cube_half_size]),
        )


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        pass

    def evaluate(self):
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=bool),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

    def _get_obs_extra(self, info: Dict):
        return dict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


    def _load_kitchen_counter(self):
        path_to_kitchen_scene_builder = osp.dirname(inspect.getfile(KitchenCounterSceneBuilder))
        model_dir = Path(path_to_kitchen_scene_builder) / "assets"
        table_model_file = str(model_dir / "kitchen_counter.glb")
        table_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, np.pi))
        builder = self.scene.create_actor_builder()
        builder.add_nonconvex_collision_from_file(
            filename=table_model_file, pose=table_pose, scale=[1.0] * 3
        )
        builder.add_visual_from_file(
            filename=table_model_file, scale=[1.0] * 3, pose=table_pose
        )
        builder.initial_pose = sapien.Pose(p=[1.5*2.4178784, 0, -0.9])
        table = builder.build_static(name="kitchen-counter")
        aabb = (
            table._objs[0]
            .find_component_by_type(sapien.render.RenderBodyComponent)
            .compute_global_aabb_tight()
        )
        self.kitchen_table_length = aabb[1, 0] - aabb[0, 0]
        self.kitchen_table_width = aabb[1, 1] - aabb[0, 1]
        self.kitchen_table_height = aabb[1, 2] - aabb[0, 2]


    def _load_shelf(self, offset=None):
        if offset is None:
            offset = [1., 0., 0.]
        shelf_size_he = [0.3/2, 2./2, 0.02/3]
        shelf_rod_he = [0.05/2, 0.05/2, 1.6/2]
        self.shelf_components = []
        for i in range(5):
            builder = self.scene.create_actor_builder()
            pos = [0 + offset[0], 0 + offset[1], -0.74 + 0.32 * i + offset[2]]
            builder.add_box_collision(
                pose=sapien.Pose(p=[0, 0, 0.01]),
                half_size=shelf_size_he,
            )
            builder.add_box_visual(
                pose=sapien.Pose(p=[0, 0, 0.01]),
                half_size=shelf_size_he,
            )
            builder.initial_pose = sapien.Pose(p=pos, q=euler2quat(0, 0, np.pi / 2))
            shelf = builder.build_static(
                name=f"shelf_{i}"
            )
            self.shelf_components.append(shelf)

        positions_rods = [
            [1. + offset[0], 0.16 + offset[1], -0.1 + offset[2]],
            [1. + offset[0], -0.16 + offset[1], -0.1 + offset[2]],
            [-1. + offset[0], 0.16 + offset[1], -0.1 + offset[2]],
            [-1. + offset[0], -0.16 + offset[1], -0.1 + offset[2]],
        ]
        for i in range(4):
            builder = self.scene.create_actor_builder()
            pos = positions_rods[i]
            builder.add_box_collision(
                pose=sapien.Pose(p=[0, 0, 0.01]),
                half_size=shelf_rod_he,
            )
            builder.add_box_visual(
                pose=sapien.Pose(p=[0, 0, 0.01]),
                half_size=shelf_rod_he,
            )
            builder.initial_pose = sapien.Pose(p=pos, q=euler2quat(0, 0, 0))
            shelf_rod = builder.build_static(
                name=f"shelf_rod_{i}"
            )
            self.shelf_components.append(shelf_rod)



