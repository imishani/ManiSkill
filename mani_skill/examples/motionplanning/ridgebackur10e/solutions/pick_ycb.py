import mplib
import numpy as np
import sapien
import torch
import trimesh

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from mani_skill.envs.tasks import PickSingleKitchenYCBEnv, PickHeavyClutterYCBEnv, PickCubeEnv, PickBlockEnv
from mani_skill.examples.motionplanning.ridgebackur10e.motionplanner import \
    RidgebackUR10ePlanningSolver, CLOSED, OPEN
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.common import quat_diff_rad
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix


def solve(env: PickSingleKitchenYCBEnv | PickHeavyClutterYCBEnv | PickCubeEnv | PickBlockEnv,
          seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = RidgebackUR10ePlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    ind_env = 0 # Not parallelized on gpu
    # torch_planer =

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    if isinstance(env, PickCubeEnv):
        target = env.cube
    elif isinstance(env, PickSingleKitchenYCBEnv):
        target = env.obj
    elif isinstance(env, PickHeavyClutterYCBEnv):
        target = env.target_object
    else:
        raise ValueError("Unknown environment type")

    # loop through the objects in the scene, and if they are not the object of interest, add them to the planner

    ptc_array = None
    for actor in env.scene.actors.values():
        if actor.per_scene_id[ind_env] != target.per_scene_id[ind_env]:
        # if True:
            # get collision meshes
            meshes = actor.get_collision_meshes()
            for mesh in meshes:
                ptc = trimesh.sample.sample_surface(mesh, 2000)[0]
                # to numpy
                if ptc_array is None:
                    ptc_array = np.array(ptc)
                else:
                    ptc_array = np.concatenate([ptc_array, np.array(ptc)])
    planner.add_collision_pts(ptc_array)

    # get_ooi = lambda x: next(y for y in x if y.name.startswith(env.model_id))
    get_ooi = lambda x: next(y for y in x if y.per_scene_id[ind_env] == target.per_scene_id[ind_env])
    # get_ooi = lambda x: next(y for y in x if y.per_scene_id[ind_env] == env.obj.per_scene_id[ind_env])
    ooi = get_ooi(env.scene.actors.values())

    grasp_pose = planner.find_best_grasp(ooi, n_samples=100)

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
    grasp_pose = grasp_pose * sapien.Pose([0., 0, 0.05])
    reach_pose = grasp_pose * sapien.Pose([0., 0, 0.15])

    meshes = ooi.get_collision_meshes()

    ptc_array = None
    for mesh in meshes:
        # add the mesh to the planner
        # get pointcloud from mesh
        ptc = trimesh.sample.sample_surface(mesh, 2000)[0]
        # to numpy
        if ptc_array is None:
            ptc_array = np.array(ptc)
        else:
            ptc_array = np.concatenate([ptc_array, np.array(ptc)])
    planner.add_collision_pts(ptc_array)


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(ptc[:, 0], ptc[:, 1], ptc[:, 2], s=1)
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-2, 2)
    # ax.set_zlim(-1, 2)
    # plt.show()

    planner.move_to_pose_with_RRTConnect(reach_pose, refine_steps=20)
    # planner.move_to_pose_with_screw(reach_pose, refine_steps=20)

    planner.clear_collisions()
    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose, refine_steps=0)
    res = planner.control_gripper()
    reach_pose = reach_pose * sapien.Pose(p=[0, 0, 0.1])
    planner.move_to_pose_with_screw(reach_pose, refine_steps=10)

    ptc_array = None
    for actor in env.scene.actors.values():
        if actor.per_scene_id[ind_env] != target.per_scene_id[ind_env]:
        # if actor.per_scene_id[ind_env] != env.obj.per_scene_id[ind_env]:
        # if not actor.name.startswith(env.model_id):
            # get collision meshes
            meshes = actor.get_collision_meshes()
            for mesh in meshes:
                # add the mesh to the planner
                # get pointcloud from mesh
                ptc = trimesh.sample.sample_surface(mesh, 2000)[0]
                if ptc_array is None:
                    ptc_array = np.array(ptc)
                else:
                    ptc_array = np.concatenate([ptc_array, np.array(ptc)])
    planner.add_collision_pts(ptc_array)

    ooi_relative_pose = ooi.pose.inv() * planner.robot.get_links()[-1].pose
    mplib_rel_pose = mplib.Pose(ooi_relative_pose.p.squeeze().cpu().numpy(),
                                ooi_relative_pose.q.squeeze().cpu().numpy())
    planner.planner.update_attached_box(size=(2*0.1, 2*0.1, 0.04),
    # planner.planner.update_attached_box(size=2*ooi._bodies[0].collision_shapes[0].half_size,
    # planner.planner.update_attached_box(size=(2*ooi._bodies[0].collision_shapes[0].radius,
    #                                           2*ooi._bodies[0].collision_shapes[0].radius,
    #                                           2*ooi._bodies[0].collision_shapes[0].half_length),
                                        pose=mplib_rel_pose)



    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    # goal_pose = sapien.Pose(env.goal_site.pose.sp.p, reach_pose.q)
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p)


    # res = planner.move_to_pose_with_RRTConnect(goal_pose, constrain=True)
    res = planner.move_to_pose_with_RRTConnect(goal_pose, constrain=False)
    # res = planner.move_to_pose_with_screw(goal_pose)
    if res == -1:
        planner.close()
        return res

    if res[4]['is_obj_placed']:
        res[4]['success'] = torch.tensor([True])

    planner.control_gripper(gripper_state=OPEN)

    planner.close()
    return res
