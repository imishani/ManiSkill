import numpy as np
import sapien

from mani_skill.envs.tasks import PickSingleKitchenYCBEnv
from mani_skill.examples.motionplanning.ridgebackur10e.motionplanner import \
    RidgebackUR10ePlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: PickSingleKitchenYCBEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = RidgebackUR10ePlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    # torch_planer =

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    grasp_pose = planner.find_best_grasp(env.obj)
    qpose = grasp_pose[1]
    grasp_pose = sapien.Pose(grasp_pose[0].p, grasp_pose[0].q)

    # # retrieves the object oriented bounding box (trimesh box object)
    # # get the object name from the env
    # obb = get_actor_obb(env.obj)
    #
    # approaching = np.array([0, 0, 1])
    # # get transformation matrix of the tcp pose, is default batched and on torch
    # target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # # we can build a simple grasp pose using this information for Panda
    # grasp_info = compute_grasp_info_by_obb(
    #     obb,
    #     approaching=approaching,
    #     target_closing=target_closing,
    #     depth=FINGER_LENGTH,
    # )
    # closing, center = grasp_info["closing"], grasp_info["center"]
    # grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.obj.pose.sp.p)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0., 0, 0.1])
    planner.move_to_pose_with_RRTConnect(reach_pose, refine_steps=10)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    res = planner.close_gripper()
    planner.move_to_pose_with_screw(reach_pose)
    # planner.move_to_qpose_withRRTConnect([qpose])
    # planner.close_gripper()
    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p, reach_pose.q)
    res = planner.move_to_pose_with_RRTConnect(goal_pose)
    # res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res
