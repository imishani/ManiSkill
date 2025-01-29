import mplib
import numpy as np
import sapien
import torch
import trimesh

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import PickBlockEnv
from mani_skill.examples.motionplanning.ridgebackur10e.motionplanner import \
    RidgebackUR10ePlanningSolver, CLOSED, OPEN
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.common import quat_diff_rad
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix

def solve(env: PickBlockEnv,
          seed=None, debug=False, vis=False, reset=True):
    if reset:
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
    assert len(env._objs) > 0
    target = env._objs[0]

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
    aabb_size = ooi.get_collision_meshes()[0].bounding_cylinder.extents[0]
    grasp_position = ooi.pose.p
    grasp_pose = sapien.Pose(p=grasp_position.cpu().numpy().squeeze() + np.array([0, 0, 0.1]),
                             q=euler2quat(0.0, -np.pi/2, 0))

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    # grasp_pose = grasp_pose * sapien.Pose([0., 0, -0.005])
    reach_pose = grasp_pose * sapien.Pose([0., 0, 0.1])

    meshes = ooi.get_collision_meshes()
    ooi_bb = ooi.get_first_collision_mesh().bounding_box.bounds

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

    if planner.move_to_pose_with_RRTConnect(reach_pose, refine_steps=10) == -1:
        # try to rotate by 180 degrees around z
        reach_pose = reach_pose * sapien.Pose(q=[0, 0, np.sin(np.pi), np.cos(np.pi)])
        if planner.move_to_pose_with_RRTConnect(reach_pose, refine_steps=10) == -1:
            planner.close()
            return -1
        else:
            grasp_pose = grasp_pose * sapien.Pose(q=[0, 0, np.sin(np.pi), np.cos(np.pi)])
    # planner.move_to_pose_with_screw(reach_pose, refine_steps=20)

    planner.clear_collisions()
    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    if planner.move_to_pose_with_screw(grasp_pose, refine_steps=0) == -1:
        print("Failed screw to grasp")
        planner.close()
        return -1

    res = planner.control_gripper(gripper_state=CLOSED)

    reach_pose = reach_pose * sapien.Pose(p=[0.05, 0, 0.0])

    if planner.move_to_pose_with_screw(reach_pose, refine_steps=10) == -1:
        print("Failed screw grasp to reach")
        planner.close()
        return -1
    # reach_pose = reach_pose * sapien.Pose(p=[0, 0, 0.1])
    #
    # if planner.move_to_pose_with_screw(reach_pose, refine_steps=0) == -1:
    #     planner.close()
    #     return -1
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
    planner.planner.update_attached_box(size=(ooi_bb[1, 2] - ooi_bb[0, 2],
                                              ooi_bb[1, 0] - ooi_bb[0, 0],
                                              ooi_bb[1, 1] - ooi_bb[0, 1]),
                                        pose=mplib.Pose())

    # #generate pointcloud from cylinder
    # planner.planner.update_attached_sphere(radius=0.254/2.,
    #                                        pose=mplib.Pose())



    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p + np.array([0, 0, 0.15]), reach_pose.q)
    # goal_pose = sapien.Pose(env.goal_site.pose.sp.p)


    # res = planner.move_to_pose_with_RRTConnect(goal_pose, constrain=True)
    # res = planner.move_to_pose_with_RRTConnect(goal_pose, constrain=False)
    res = planner.move_to_pose_with_screw(goal_pose)
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p + np.array([0, 0, 0.1]), reach_pose.q)
    res = planner.move_to_pose_with_screw(goal_pose)
    if res == -1:
        planner.close()
        return res

    if res[4]['is_obj_placed']:
        res[4]['success'] = torch.tensor([True])

    planner.control_gripper(gripper_state=OPEN)

    planner.close()
    return res
