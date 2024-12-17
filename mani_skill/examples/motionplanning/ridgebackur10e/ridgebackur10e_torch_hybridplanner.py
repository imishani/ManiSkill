import sys
import os
from pathlib import Path
from typing import Tuple

import einops
import sapien
import torch
import numpy as np
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
from matplotlib import pyplot as plt

from mp_baselines.planners.gpmp2 import GPMP2
from mp_baselines.planners.hybrid_planner import HybridPlanner
from mp_baselines.planners.multi_sample_based_planner import MultiSampleBasedPlanner
from mp_baselines.planners.rrt_connect import RRTConnect

from planning.domains.maniskill.envs.maniskill_env import ManiSkillEnv
from planning.domains.maniskill.robots.ridgebackur10e import RobotRidgebackUR10e, RobotStaticRidgebackUR10e
from pyplan.utils.utils_torch import get_device

# from torch_robotics.environments.env_spheres_3d import EnvSpheres3D
# from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

from domains.maniskill.utils.manip_utils import *

import gymnasium as gym
from transforms3d.euler import quat2euler

allow_ops_in_compiled_graph()

# enum
# from enum import Enum
# class GripperState(Enum):
OPEN = 0
CLOSED = 0.8

class GraspTask(PlanningTask):
    def __init__(self,
                 env: ManiSkillEnv,
                 robot: RobotRidgebackUR10e,
                 ooi: sapien.Entity = None,
                 ws_limits=None,
                 cell_size=0.01,
                 obstacle_cutoff_margin=0.01,
                 base_max_dist_from_ooi=1.5,
                 **kwargs):
        """

        :param env:
        :param robot:
        :param ooi: Object of Interest
        :param ws_limits:
        :param cell_size:
        :param obstacle_cutoff_margin:
        :param kwargs:
        """
        super(GraspTask, self).__init__(env=env,
                                        robot=robot,
                                        ws_limits=ws_limits,
                                        cell_size=cell_size,
                                        obstacle_cutoff_margin=obstacle_cutoff_margin, **kwargs)

        self.ooi = ooi
        self.base_max_dist_from_ooi = base_max_dist_from_ooi

    def find_grasp(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return find_best_grasp(self.robot,
                               self.compute_collision,
                               pk_inverse_kinematics,
                               env.scene.obj if self.ooi is None else self.ooi,
                               1000,
                               tensor_args,
                               debug=False)

    def coll_free_grasp_q(self, n_samples: int) -> torch.Tensor:
        return self.find_grasp()[2]

    def random_coll_free_q_around_ooi(self,
                                      n_samples: int = 1,
                                      max_samples: int = 1000,
                                      max_tries: int = 1000) -> torch.Tensor:
        return self.env.scene.agent.robot.get_qpos()[0, :9]
        # Get the object's position
        obj = self.ooi if self.ooi is not None else self.env.scene.obj
        obj_position = obj.pose.p
        obj_position = torch.tensor([obj_position[:, 0], obj_position[:, 1], obj_position[:, 2]],
                                    **self.tensor_args)
        arm_home_joint_angles = self.env.scene.agent.robot.get_qpos()[0, 3:9] # TODO: remove the hardcoded 9 and use the arm joints
        reject = True
        samples = torch.zeros((n_samples, self.robot.q_dim), **self.tensor_args)
        idx_begin = 0
        for i in range(max_tries):
            qs = self.robot.random_q(max_samples)
            qs[:, :2] = obj_position[:2] + torch.randn_like(qs[:, :2], **self.tensor_args) * self.base_max_dist_from_ooi
            qs[:, 0] = obj_position[0] - torch.rand(qs[:, 0].shape, **self.tensor_args) * 0.8
            qs[:, 3:] = arm_home_joint_angles + torch.randn_like(qs[:, 3:], **self.tensor_args) * 0.1
            # make sure within limits
            qs = torch.clamp(qs, self.robot.q_limits[0, :], self.robot.q_limits[1, :])

            in_collision = self.compute_collision(qs).squeeze()
            idxs_not_in_collision = torch.argwhere(in_collision == False).squeeze()
            if idxs_not_in_collision.nelement() == 0:
                # all points are in collision
                continue
            if idxs_not_in_collision.nelement() == 1:
                idxs_not_in_collision = [idxs_not_in_collision]
            idx_random = torch.randperm(len(idxs_not_in_collision))[:n_samples]
            free_qs = qs[idxs_not_in_collision[idx_random]]
            idx_end = min(idx_begin + free_qs.shape[0], samples.shape[0])
            samples[idx_begin:idx_end] = free_qs[:idx_end - idx_begin]
            idx_begin = idx_end
            if idx_end >= n_samples:
                reject = False
                break

        if reject:
            sys.exit("Could not find a collision free configuration")

        return samples.squeeze()

def interpolate_path(path, resolution=0.1):
    path_to_return = torch.tensor([], device=path.device, dtype=path.dtype)
    for i in range(path.shape[1] - 1):
        start = path[:, i].squeeze(0)
        end = path[:, i + 1].squeeze(0)
        n_points = int(torch.linalg.norm(end - start) / resolution)
        # add the start point
        path_to_return = torch.cat((path_to_return, start.unsqueeze(1)), dim=1)
        path_to_return = torch.cat((path_to_return,
                                    torch.stack([torch.linspace(start[j], end[j], n_points, device=path.device)
                                                 for j in range(start.shape[0])], dim=0)), dim=1)
        # add the last point
        path_to_return = torch.cat((path_to_return, end.unsqueeze(1)), dim=1)
    return path_to_return


def follow_path(gym_env,
                result,
                gripper_state,
                refine_steps: int = 0,
                vis = True):
    n_step = result["position"].shape[0]
    for i in range(n_step + refine_steps):
        qpos = result["position"][min(i, n_step - 1)]
        if gym_env.unwrapped.control_mode == "pd_joint_pos_vel":
            qvel = result["velocity"][min(i, n_step - 1)]
            action = np.hstack([qpos, qvel, gripper_state])
        else:
            action = np.hstack([qpos, gripper_state])
        # delta_base = action[:3] - self.robot.get_qpos()[0, :3].cpu().numpy()
        # action[:3] = delta_base
        obs, reward, terminated, truncated, info = gym_env.step(action)
        if vis:
            gym_env.unwrapped.render_human()
    return obs, reward, terminated, truncated, info

def render_wait(gym_env,
                vis=True):
    if not vis:
        return
    print("Press [c] to continue")
    viewer = gym_env.unwrapped.render_human()
    while True:
        if viewer.window.key_down("c"):
            break
        gym_env.unwrapped.render_human()

if __name__ == "__main__":
    base_file_name = Path(os.path.basename(__file__)).stem

    seed = 14
    fix_random_seed(seed)

    device = get_device()
    tensor_args = {'device': 'cpu', 'dtype': torch.float32}

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    gym_env = gym.make(
        "PickSingleKitchenYCB-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="human",
        reward_mode="dense",
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
        sim_backend="auto",
        # robot_uids="ridgebackur10e"
        robot_uids="static_ridgebackur10e"

    )
    gym_env.reset(seed=None)

    env = ManiSkillEnv(maniskill_scene=gym_env.unwrapped,
                       precompute_sdf_obj_fixed=True,
                       sdf_cell_size=0.02,
                       tensor_args=tensor_args)

    rpy = torch.tensor(quat2euler(gym_env.unwrapped.agent.robot.root_pose.q.squeeze()))
    xyz = gym_env.unwrapped.agent.robot.root_pose.p.squeeze()
    if gym_env.unwrapped.robot_uids == "ridgebackur10e":
        robot = RobotRidgebackUR10e(gym_env.unwrapped.agent.urdf_path,
                                    tensor_args=tensor_args,
                                    root_link_pose=torch.cat([xyz, rpy]))
    elif gym_env.unwrapped.robot_uids == "static_ridgebackur10e":
        robot = RobotStaticRidgebackUR10e(gym_env.unwrapped.agent.urdf_path,
                                          tensor_args=tensor_args,
                                          root_link_pose=torch.cat([xyz, rpy]))
    else:
        raise ValueError(f"Invalid robot_uid: {gym_env.unwrapped.agent.robot.robot_uid}")

    # robot = RobotPanda(
    #     # use_self_collision_storm=True,
    #     # grasped_object=GraspedObjectPandaBox(tensor_args=tensor_args),
    #     tensor_args=tensor_args
    # )


    # task = PlanningTask(
    #     env=env,
    #     robot=robot,
    #     ws_limits=torch.tensor([[-2., -2., -1.0], [2., 2., 2.0]], **tensor_args),  # workspace limits
    #     obstacle_cutoff_margin=0.05,
    #     tensor_args=tensor_args
    # )
    task = GraspTask(
        env=env,
        robot=robot,
        cell_size=0.01,
        obstacle_cutoff_margin=0.01,
        tensor_args=tensor_args
    )

    # -------------------------------- Planner ---------------------------------
    # # for _ in range(100):
    # q_free = task.random_coll_free_q(n_samples=2)
    # start_state = q_free[0]
    # goal_state = q_free[1]

    q_free = task.random_coll_free_q_around_ooi(n_samples=1)

    q_free_goal = task.coll_free_grasp_q(n_samples=1)
    start_state = q_free[:q_free_goal.shape[0]]
    # if a joint is more than 2 radians away from its corresponding joint in the start state, bring it back by 2 radians
    for i in range(start_state.shape[0]):
        if abs(start_state[i] - q_free_goal[i]) > 2 * torch.pi:
            if start_state[i] > q_free_goal[i]:
                q_free_goal[i] += 2 * torch.pi
            else:
                q_free_goal[i] -= 2 * torch.pi
    goal_state = q_free_goal

    # check if the EE positions are "enough" far apart
    start_state_ee_pos = robot.get_EE_position(start_state).squeeze()
    goal_state_ee_pos = robot.get_EE_position(goal_state).squeeze()

    n_trajectories = 1

    print(start_state)
    print(goal_state)
    ############### Sample-based planner
    rrt_connect_default_params_env = env.get_rrt_connect_params(robot=robot)
    # rrt_connect_default_params_env['max_time'] = 300

    rrt_connect_params = dict(
        **rrt_connect_default_params_env,
        task=task,
        start_state_pos=start_state,
        goal_state_pos=goal_state,
        tensor_args=tensor_args,
    )
    sample_based_planner_base = RRTConnect(**rrt_connect_params)
    # sample_based_planner_base = MoveItSamplingBasedPlanner(start_state_pos=start_state,
    #                                                        goal_state_pos=goal_state,
    #                                                        max_time=rrt_connect_default_params_env.get('max_time', 15),
    #                                                        tensor_args=tensor_args)
    sample_based_planner = MultiSampleBasedPlanner(
        sample_based_planner_base,
        n_trajectories=n_trajectories,
        max_processes=10,
        optimize_sequentially=True
    )
    pre_optimization_planners = [sample_based_planner]

    ############### Optimization-based planner
    n_support_points = 64
    dt = 0.04
    gpmp_opt_iters=500
    duration=5.0
    gpmp_default_params_env = env.get_gpmp2_params(robot=robot)
    gpmp_default_params_env['opt_iters'] = gpmp_opt_iters
    gpmp_default_params_env['n_support_points'] = n_support_points
    gpmp_default_params_env['dt'] = duration / n_support_points

    # Construct planner
    planner_params = dict(
        **gpmp_default_params_env,
        robot=robot,
        n_dof=robot.q_dim,
        num_particles_per_goal=n_trajectories,
        start_state=start_state,
        multi_goal_states=goal_state.unsqueeze(0),  # add batch dim for interface,
        collision_fields=task.get_collision_fields(),
        tensor_args=tensor_args,
    )
    opt_based_planner = GPMP2(**planner_params)

    ############### Hybrid planner
    opt_iters = planner_params['opt_iters']
    planner = HybridPlanner(
        sample_based_planner,
        opt_based_planner,
        tensor_args=tensor_args
    )

    trajs_iters = planner.optimize(debug=True, return_iterations=True)

    # save trajectories
    torch.cuda.empty_cache()
    trajs_iters_coll, trajs_iters_free = task.get_trajs_collision_and_free(trajs_iters[-1])
    if trajs_iters_coll is not None:
        torch.save(trajs_iters_coll.unsqueeze(0), f'trajs_iters_coll_{base_file_name}.pt')
    if trajs_iters_free is not None:
        torch.save(trajs_iters_free.unsqueeze(0), f'trajs_iters_free_{base_file_name}.pt')

    # -------------------------------- Visualize ---------------------------------
    planner_visualizer = PlanningVisualizer(
        task=task,
        planner=planner
    )

    print(f'----------------STATISTICS----------------')
    print(f'percentage free trajs: {task.compute_fraction_free_trajs(trajs_iters[-1])*100:.2f}')
    print(f'percentage collision intensity {task.compute_collision_intensity_trajs(trajs_iters[-1])*100:.2f}')
    print(f'success {task.compute_success_free_trajs(trajs_iters[-1])}')

    pos_trajs_iters = robot.get_position(trajs_iters)
    results = {
        'position': pos_trajs_iters[-1, 0, ...].cpu().numpy(),
        'velocity': trajs_iters[-1, 0, :, -9:].cpu().numpy()
    }

    follow_path(gym_env, results, OPEN, refine_steps=50)
    render_wait(gym_env)

    # planner_visualizer.plot_joint_space_state_trajectories(
    #     trajs=trajs_iters[-1],
    #     pos_start_state=start_state, pos_goal_state=goal_state,
    #     vel_start_state=torch.zeros_like(start_state), vel_goal_state=torch.zeros_like(goal_state),
    # )
    #
    # planner_visualizer.animate_opt_iters_joint_space_state(
    #     trajs=trajs_iters,
    #     pos_start_state=start_state, pos_goal_state=goal_state,
    #     vel_start_state=torch.zeros_like(start_state), vel_goal_state=torch.zeros_like(goal_state),
    #     video_filepath=f'{base_file_name}-joint-space-opt-iters.mp4',
    #     n_frames=max((2, opt_iters // 10)),
    #     anim_time=5
    # )

    planner_visualizer.render_robot_trajectories(
        trajs=pos_trajs_iters[-1, 0][None, ...], start_state=start_state, goal_state=goal_state,
        render_planner=False,
    )
    #
    # planner_visualizer.animate_robot_trajectories(
    #     trajs=pos_trajs_iters[-1, 0][None, ...], start_state=start_state, goal_state=goal_state,
    #     plot_trajs=False,
    #     video_filepath=f'{base_file_name}-robot-traj.mp4',
    #     # n_frames=max((2, pos_trajs_iters[-1].shape[1]//10)),
    #     n_frames=pos_trajs_iters[-1].shape[1],
    #     anim_time=n_support_points*dt
    # )
    #
    plt.show()

