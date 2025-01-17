from typing import Any, Dict, Union, List

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda, RidgebackUR10e, StaticRidgebackUR10e
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.io_utils import load_json

from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.building.actors.ycb import get_ycb_builder

from mani_skill import ASSET_DIR


@register_env("PickCube-v1", max_episode_steps=50)
class PickCubeEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to grasp a red cube and move it to a target goal position.

    **Randomizations:**
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the cube's z-axis rotation is randomized to a random angle
    - the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

    **Success Conditions:**
    - the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "fetch", "ridgebackur10e", "static_ridgebackur10e"]
    agent: Union[Panda, Fetch, RidgebackUR10e, StaticRidgebackUR10e]
    cube_half_size = 0.02
    # goal_thresh = 0.025
    goal_thresh = 0.05

    def __init__(self, *args, robot_uids="static_ridgebackur10e", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 1., 1.1], [-0.12, 0.0, 0.1])
        return CameraConfig("render_camera", pose, 1280, 1280, 1, 0.01, 100)

    def _load_agent(self, options: dict,
                    init_pose: sapien.Pose = sapien.Pose(p=[-0.615, 0, 0])):
        super()._load_agent(options, init_pose)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))


    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

@register_env("PickBlock-v1", max_episode_steps=50, asset_download_ids=["ycb"])
class PickBlockEnv(PickCubeEnv):

    goal_thresh = 0.15

    kitchen_keywords = [
        'pitcher',
        'mug',
        'cup',
        'plate',
        'bowl',
        'spatula',
        'knife',
        'fork',
        'spoon',
    ]

    def __init__(self, additional_objs=True, *args, **kwargs):
        self.all_model_ids = np.array(
            list(
                load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json").keys()
            )
        )
        self.additional_objs = additional_objs
        super().__init__(*args, **kwargs)


    def _load_agent(self, options: dict,
                    init_pose: sapien.Pose = sapien.Pose(p=[-1.2, 0, 0])):
        super()._load_agent(options, init_pose)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build(scale=1.2)
        # self.cube = actors.build_box(
        #     self.scene,
        #     half_sizes=(self.cube_half_size*4, self.cube_half_size*4, self.cube_half_size/1.5),
        #     color=[1, 0, 0, 1],
        #     name="cube",
        #     initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size/1.5]),
        # )
        # self.cube = actors.build_cylinder(
        #     self.scene,
        #     radius=self.cube_half_size * 6,
        #     half_length=self.cube_half_size / 1.5,
        #     color=[1, 0, 0, 1],
        #     name="cube",
        #     initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size / 1.5]),
        # )

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:029_plate",
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, self.cube_half_size * 0.5 * 1.8 + 1e-3])

        self.cube = builder.build(name=f"cube")

        self.cube.set_mass(self.cube.get_mass() * 0.1)

        builder = self.scene.create_actor_builder()
        q = euler2quat(np.pi/2., 0, 0)

        builder.add_nonconvex_collision_from_file(
            filename="/home/imishani/crate.obj",
            scale=[1.5, 1.5, 1.5]
        )

        builder.add_visual_from_file(filename="/home/imishani/crate.glb",
                                     scale=[1.5, 1.5, 1.5])

        builder.initial_pose = sapien.Pose(p=[-1.2, -0.8, -self.table_scene.table_height + 1e-3], q=q)
        self.crate = builder.build_kinematic(name="crate")
        # self.crate.set_pose(sapien.Pose(p=[-1.2, -1., 0.0], q=q))

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 0.2],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self.goal_site.set_pose(Pose.create_from_pq(p=torch.tensor([-1.2, -0.8, -self.table_scene.table_height + 0.5])))
        self._hidden_objects.append(self.goal_site)

        if not self.additional_objs:
            return

        kitchen_model_ids = [
            model_id for model_id in self.all_model_ids if any(keyword in model_id for keyword in self.kitchen_keywords)
        ]

        # randomize the list of all possible models in the YCB dataset
        # then sub-scene i will load model model_ids[i % number_of_ycb_objects]
        model_ids = self._batched_episode_rng.choice(kitchen_model_ids, replace=True)
        # if (
        #         self.num_envs > 1
        #         and self.num_envs < len(self.kitchen_model_ids)
        #         and self.reconfiguration_freq <= 0
        #         and not WARNED_ONCE
        # ):
        #     WARNED_ONCE = True
        #     print(
        #         """There are less parallel environments than total available models to sample.
        #         Not all models will be used during interaction even after resets unless you call env.reset(options=dict(reconfigure=True))
        #         or set reconfiguration_freq to be >= 1."""
        #     )
        self.model_id = model_ids[0]
        self._objs: List[Actor] = []
        self.obj_heights = []
        for i, model_id in enumerate(model_ids):
            # TODO: before official release we will finalize a metadata dataclass that these build functions should return.
            builder = get_ycb_builder(self.scene,
                                      id=model_id,
                                      add_collision=True,
                                      add_visual=True)

            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.set_scene_idxs([i])
            self._objs.append(builder.build(name=f"{model_id}-{i}"))
            self.remove_from_state_dict_registry(self._objs[-1])
        self.obj = Actor.merge(self._objs, name="ycb_objects")
        # self.add_to_state_dict_registry(self.obj)

    def _after_reconfigure(self, options: dict):
        collision_mesh = self.cube.get_first_collision_mesh()
        self.cube_z = -collision_mesh.bounding_box.bounds[0, 2]
        if not self.additional_objs:
            return
        self.object_zs = []
        self.object_meshes = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            # this value is used to set object pose so the bottom is at z=0
            self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
            self.object_meshes.append(collision_mesh)
        self.object_zs = common.to_tensor(self.object_zs, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with (torch.device(self.device)):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            # put it on the edge of the table
            x_table_center = self.table_scene.table.pose.p[:, 0]
            y_table_center = self.table_scene.table.pose.p[:, 1]
            dim = self.table_scene.table_length, self.table_scene.table_width

            noise = (torch.rand((b, 2)) * 0.01 + 0.01).to(self.device)

            # we have for options for the position of the object -- 4 edges of the table with some noise
            # regions_for_sampling = [
            #     # [x_table_center + dim[0]/2, 0],
            #     [x_table_center - dim[0] / 2, 0],
            #     [0, y_table_center + dim[1]/2],
            #     [0, y_table_center - dim[1]/2]
            # ]
            # region = regions_for_sampling[torch.randint(4, (b,))]
            # region = regions_for_sampling[torch.randint(3, (b,))]

            region = [(x_table_center - dim[0] / 2).unsqueeze(-1),
                      torch.zeros((b, 1), device=self.device)]
            region = torch.cat(region, dim=-1)

            idx = torch.where(region == 0)[-1].unsqueeze(-1)
            center = torch.stack([x_table_center, y_table_center], dim=-1)
            xyz[:, 1 - idx] = torch.gather(region, 1, 1 - idx) - torch.sign(torch.gather(region, 1, 1 - idx) -
                                                              torch.gather(center, 1, 1 - idx)) * torch.gather(noise, 1, 1 - idx)

            dim = torch.tensor(dim).repeat(b, 1).to(self.device)
            xyz[:, idx] = torch.gather(center, 1, 1 - idx) + (torch.rand((b,)) - 0.5).unsqueeze(-1) * torch.gather(dim, 1, idx)

            # xyz[..., 2] = self.cube_half_size * 0.5 * 1.8 + 1e-3
            # xyz[:, 2] = self.cube_half_size
            xyz[:, 2] = self.cube_z

            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            # qs = euler2quat(0., np.pi/2., 0)
            # qs = torch.tensor(qs).repeat(b, 1).to(self.device)
            # rotate the object 90 deg around x
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            # goal_xyz = torch.zeros((b, 3))
            # goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            # goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            # self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            ####### used for data collection #######
            # goal_xyz = xyz.clone()
            # goal_xyz[:, 0] -= 0.15
            # self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
            #########################################

            # goal_xyz = torch.zeros((b, 3))
            # goal_xyz[:, 2] += 0.2
            # self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
            self.goal_site.set_pose(
                Pose.create_from_pq(p=torch.tensor([-1.2, -0.8, -self.table_scene.table_height + 0.4])))

            if not self.additional_objs:
                return
            xyz[:, :2] = torch.rand((b, 2)) * torch.tensor([[self.table_scene.table_length, self.table_scene.table_width]]) + self.table_scene.table.pose.p[:, :2] - torch.tensor([[self.table_scene.table_length, self.table_scene.table_width]]) / 2
            xyz[:, 2] = self.object_zs[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))





    def evaluate(self):
        is_obj_placed = (
                torch.linalg.norm(self.goal_site.pose.p[:, :2] - self.cube.pose.p[:, :2], axis=1)
                <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }


