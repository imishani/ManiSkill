from typing import Any, Dict, Union, List

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import RidgebackUR10e, StaticRidgebackUR10e
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.common import quat_diff_rad
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from pytorch_kinematics import quaternion_close
from transforms3d import quaternions
from mani_skill.utils.building.actors.ycb import get_ycb_builder
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.structs.actor import Actor
from mani_skill import ASSET_DIR


@register_env("PushBlock-v1", max_episode_steps=50, asset_download_ids=["ycb"])
class PushBlockEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to push and move a block to a goal region in front of it

    **Randomizations:**
    - the cube's xy position is randomized on top of a table. It is placed flat on the table
    - the target goal region is marked by a red/white circular target. The position of the target is randomized but in distance range [0.1, 0.1] to [0.3, 0.3] from the cube's xy position

    **Success Conditions:**
    - the cube's xy position is within goal_radius (default 0.05) of the target's xy position by euclidean distance.
    """

    SUPPORTED_ROBOTS = ["ridgebackur10e", "static_ridgebackur10e"]

    # Specify some supported robot types
    agent: Union[RidgebackUR10e,StaticRidgebackUR10e]

    # set some commonly used values
    goal_radius = 0.03
    block_half_size = (0.08, 0.08, 0.01)

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

    def __init__(self, *args, robot_uids="static_ridgebackur10e",
                 robot_init_qpos_noise=0.02,
                 additional_objs=False,
                 **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.all_model_ids = np.array(
            list(
                load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json").keys()
            )
        )
        self.additional_objs = additional_objs
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
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
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = sapien_utils.look_at(eye=[1., 0, 1.5], target=[-0.12, 0, 0.1])
        return CameraConfig(
            "render_camera", pose=pose, width=1280, height=1280, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict,
                    init_pose: sapien.Pose = sapien.Pose(p=[-1.2, 0, 0])):
        super()._load_agent(options, init_pose)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build(scale=1.2)

        # self.obj = actors.build_box(
        #     self.scene,
        #     half_sizes=self.block_half_size,
        #     color=[1, 0, 0, 1],
        #     name="block",
        #     initial_pose=sapien.Pose(p=[0, 0, self.block_half_size[2]]),
        # )

        # self.obj = actors.build_cylinder(
        #     self.scene,
        #     radius=self.block_half_size[0] / np.sqrt(2),
        #     half_length=self.block_half_size[2],
        #     color=[1, 0, 0, 1],
        #     name="block",
        #     initial_pose=sapien.Pose(p=[0, 0, self.block_half_size[2]]),
        # )

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:029_plate",
        )

        builder.initial_pose = sapien.Pose(
            # p=[0, 0, -self.block_half_size[2]]
            p=[0, 0, -0.0123]
        )

        self.obj = builder.build(name=f"block")
        # self.obj.set_mass(self.obj.get_mass() * 2.)

        # we also add in red/white target to visualize where we want the cube to be pushed to
        # we specify add_collisions=False as we only use this as a visual for videos and do not want it to affect the actual physics
        # we finally specify the body_type to be "kinematic" so that the object stays in place
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
        )

        # optionally you can automatically hide some Actors from view by appending to the self._hidden_objects list. When visual observations
        # are generated or env.render_sensors() is called or env.render() is called with render_mode="sensors", the actor will not show up.
        # This is useful if you intend to add some visual goal sites as e.g. done in PickCube that aren't actually part of the task
        # and are there just for generating evaluation videos.
        self._hidden_objects.append(self.goal_region)


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

            # builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.initial_pose = sapien.Pose(p=[-0.35, -0.2, -0.0123])
            builder.set_scene_idxs([i])
            builder.set_physx_body_type("static")
            self._objs.append(builder.build(name=f"{model_id}-{i}"))
            self.remove_from_state_dict_registry(self._objs[-1])
        self.obj_extra = Actor.merge(self._objs, name="ycb_objects")
        # self.add_to_state_dict_registry(self.obj)

    def _after_reconfigure(self, options: dict):

        collision_mesh = self.obj.get_first_collision_mesh()
        self.obj_z = -collision_mesh.bounding_box.bounds[0, 2]
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
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            b = len(env_idx)
            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.
            self.table_scene.initialize(env_idx)

            # here we write some randomization code that randomizes the x, y position of the cube we are pushing
            xyz = torch.zeros((b, 3))
            xyz[..., 0] = ((torch.rand((b, 1)) * self.table_scene.table_length - self.table_scene.table_length / 2.) * 0.8) - 0.05
            xyz[..., 1] = (torch.rand((b, 1)) * self.table_scene.table_width - self.table_scene.table_width / 2.) * 0.8
            xyz[..., 0] += self.table_scene.table.pose.p[env_idx, 0]
            xyz[..., 1] += self.table_scene.table.pose.p[env_idx, 1]
            # xyz[..., 2] = self.block_half_size[2] + 1e-3
            # xyz[..., 2] = self.block_half_size[2] * 1.8 + 1e-3
            xyz[..., 2] = self.obj_z / 2. + 1e-3

            q = randomization.random_quaternions(b, lock_x=True, lock_y=True)

            # we can then create a pose object using Pose.create_from_pq to then set the cube pose with. Note that even though our quaternion
            # is not batched, Pose.create_from_pq will automatically batch p or q accordingly
            # furthermore, notice how here we do not even using env_idx as a variable to say set the pose for objects in desired
            # environments. This is because internally any calls to set data on the GPU buffer (e.g. set_pose, set_linear_velocity etc.)
            # automatically are masked so that you can only set data on objects in environments that are meant to be initialized
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            # here we set the location of that red/white target (the goal region).
            # In particular, we randomize the xy position of the target to be in distance range [0.1, 0.1] to [0.3, 0.3]
            # from the cube's xy position and we further rotate 90 degrees on the
            # y-axis to make the target object face up
            target_region_xyz = xyz
            target_region_xyz[..., :2] += (torch.rand((b, 2)) - 0.5) * 0.4  - torch.tensor([0.1, 0.])
            # target_region_xyz[..., :2] += (torch.rand((b, 2)) - 0.5) * 0.6  - torch.tensor([0.1, 0.])

            # if target positions are outside the table, clip them to the edge
            target_region_xyz[..., 1] = torch.clamp(target_region_xyz[..., 1],
                                                    - (self.table_scene.table_width / 2.) + self.table_scene.table.pose.p[env_idx, 1] + 1e-2,
                                                    (self.table_scene.table_width / 2.) + self.table_scene.table.pose.p[env_idx, 1] - 1e-2) # TODO: not sure that is true for batch
            target_region_xyz[..., 0] = torch.clamp(target_region_xyz[..., 0],
                                                    - (self.table_scene.table_length / 2.) + self.table_scene.table.pose.p[env_idx, 0] + 1e-2,
                                                    (self.table_scene.table_length / 2.) + self.table_scene.table.pose.p[env_idx, 0] - 1e-2) # TODO: not sure that is true for batch

            # ########### edge of table
            # target_region_xyz[..., 0] = -0.5691 + 0.01*torch.randn((b,))


            # set a little bit above 0 so the target is sitting on the table
            target_region_xyz[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

            if not self.additional_objs:
                return

            # xyz[:, :2] = torch.rand((b, 2)) * torch.tensor([[self.table_scene.table_length, self.table_scene.table_width]]) + self.table_scene.table.pose.p[:, :2] - torch.tensor([[self.table_scene.table_length, self.table_scene.table_width]]) / 2
            xyz[..., :2] = torch.tensor([-0.35, -0.2]).repeat(b, 1).to(self.device)
            xyz[:, 2] = self.object_zs[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.obj_extra.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def set_target_pose(self, options: dict):
        target_region_xyz = self.obj.pose.p
        target_region_xyz[..., :2] += (torch.rand((target_region_xyz.shape[0], 2)) - 0.5) * 0.4  - torch.tensor([0.1, 0.])
        # target_region_xyz[..., :2] += (torch.rand((b, 2)) - 0.5) * 0.6 - torch.tensor([0.1, 0.])

        # if target positions are outside the table, clip them to the edge
        target_region_xyz[..., 1] = torch.clamp(target_region_xyz[..., 1],
                                                - (self.table_scene.table_width / 2.) + self.table_scene.table.pose.p[0, 1] + 1e-2,
                                                (self.table_scene.table_width / 2.) + self.table_scene.table.pose.p[
                                                    0, 1] - 1e-2)  # TODO: not sure that is true for batch
        target_region_xyz[..., 0] = torch.clamp(target_region_xyz[..., 0],
                                                - (self.table_scene.table_length / 2.) + self.table_scene.table.pose.p[
                                                    0, 0] + 1e-2,
                                                (self.table_scene.table_length / 2.) + self.table_scene.table.pose.p[
                                                    0, 0] - 1e-2)  # TODO: not sure that is true for batch

        # ########### edge of table
        # target_region_xyz[..., 0] = -0.5691 + 0.01*torch.randn((b,))

        # set a little bit above 0 so the target is sitting on the table
        target_region_xyz[..., 2] = 1e-3
        self.goal_region.set_pose(
            Pose.create_from_pq(
                p=target_region_xyz,
                q=euler2quat(0, np.pi / 2, 0),
            )
        )

    def evaluate(self):
        # success is achieved when the cube's xy position on the table is within the
        # goal region's area (a circle centered at the goal region's xy position)
        is_obj_placed = (
            torch.linalg.norm(
                self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        # and quaternion_close(self.obj.pose.q, self.goal_region.pose.q)
        )

        return {
            "success": is_obj_placed,
        }

    def _get_obs_extra(self, info: Dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self._obs_mode in ["state", "state_dict"]:
            # if the observation mode is state/state_dict, we provide ground truth information about where the cube is.
            # for visual observation modes one should rely on the sensed visual data to determine where the cube is
            obs.update(
                goal_pos=self.goal_region.pose.p,
                obj_pose=self.obj.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # We also create a pose marking where the robot should push the cube from that is easiest (pushing from behind the cube)
        # get the direction of the push (target pose - block pose)
        push_direction = (self.goal_region.pose.p - self.obj.pose.p) / torch.norm(self.goal_region.pose.p - self.obj.pose.p)
        push_direction[..., 2] = 0  # add 0 to the z component to make it a 2D push direction
        
        tcp_push_pose = Pose.create_from_pq(
            p=self.obj.pose.p - (self.block_half_size[0] * 2.0 * push_direction),

        )
        tcp_push_pose.set_p(tcp_push_pose.p + torch.tensor([0, 0, 0.03]).to(self.device))
        # q_rot = quaternions.axangle2quat([0, 0, 1], np.arctan2(push_direction[1], push_direction[0]))
        # q = quaternions.qmult(q_rot, tcp_push_pose.q)
        # tcp_push_pose.set_q(q)

        tcp_to_push_position = tcp_push_pose.p - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_position, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)
        reward = reaching_reward

        # add orientation reward
        # tcp_to_push_quat_diff = quat_diff_rad(tcp_push_pose.q, self.agent.tcp.pose.q)
        # reward += 1 - torch.tanh(5 * tcp_to_push_quat_diff)

        # compute a placement reward to encourage robot to move the cube to the center of the goal region
        # we further multiply the place_reward by a mask reached so we only add the place reward if the robot has reached the desired push pose
        # This reward design helps train RL agents faster by staging the reward out.
        reached = tcp_to_push_pose_dist < 0.03
        obj_to_goal_dist = torch.linalg.norm(
            self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * reached

        # assign rewards to parallel environments that achieved success to the maximum of 3.
        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
