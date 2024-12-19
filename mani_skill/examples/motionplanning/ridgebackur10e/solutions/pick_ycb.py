import numpy as np
import sapien
import torch
import trimesh

from mani_skill.envs.tasks import PickSingleKitchenYCBEnv, PickHeavyClutterYCBEnv, PickCubeEnv, PickBlockEnv
from mani_skill.examples.motionplanning.ridgebackur10e.motionplanner import \
    RidgebackUR10ePlanningSolver
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
    for actor in env.scene.actors.values():
        if actor.per_scene_id[ind_env] != target.per_scene_id[ind_env]:
        # if True:
            # get collision meshes
            meshes = actor.get_collision_meshes()
            for mesh in meshes:
                ptc = trimesh.sample.sample_surface(mesh, 2000)[0]
                # to numpy
                ptc = np.array(ptc)
                planner.add_collision_pts(ptc)
    # get_ooi = lambda x: next(y for y in x if y.name.startswith(env.model_id))
    get_ooi = lambda x: next(y for y in x if y.per_scene_id[ind_env] == target.per_scene_id[ind_env])
    # get_ooi = lambda x: next(y for y in x if y.per_scene_id[ind_env] == env.obj.per_scene_id[ind_env])
    ooi = get_ooi(env.scene.actors.values())
    planner.render_wait()
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
    reach_pose = grasp_pose * sapien.Pose([0., 0, 0.1])

    meshes = ooi.get_collision_meshes()
    for mesh in meshes:
        # add the mesh to the planner
        # get pointcloud from mesh
        ptc = trimesh.sample.sample_surface(mesh, 2000)[0]
        # to numpy
        ptc = np.array(ptc)
        planner.add_collision_pts(ptc)

    planner.move_to_pose_with_RRTConnect(reach_pose, refine_steps=0)

    planner.clear_collisions()

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
                # to numpy
                ptc = np.array(ptc)
                planner.add_collision_pts(ptc)


    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    res = planner.close_gripper()
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p, reach_pose.q)


    # res = planner.move_to_pose_with_RRTConnect(goal_pose, constrain=True)
    res = planner.move_to_pose_with_RRTConnect(goal_pose, constrain=False)
    # res = planner.move_to_pose_with_screw(goal_pose)
    if res == -1:
        planner.close()
        return res
    print(res[4]['is_obj_placed'])
    if res[4]['is_obj_placed']:
        res[4]['success'] = torch.tensor([True])
    planner.close()
    return res
