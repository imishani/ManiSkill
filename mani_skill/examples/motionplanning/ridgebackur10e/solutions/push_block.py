import numpy as np
import sapien
import trimesh

from mani_skill.envs.tasks import PushBlockEnv
from mani_skill.examples.motionplanning.ridgebackur10e.motionplanner import RidgebackUR10ePlanningSolver

def solve(env: PushBlockEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = RidgebackUR10ePlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped
    ind_env = 0  # Not parallelized on gpu

    planner.control_gripper()
    push_direction = (env.goal_region.pose.sp.p - env.obj.pose.sp.p) / np.linalg.norm(env.goal_region.pose.sp.p - env.obj.pose.sp.p)
    push_direction[-1] = 0.
    # reach_pose = sapien.Pose(p=env.obj.pose.sp.p - (env.block_half_size[0] * 2.0* push_direction))
    # # a little ontop of the table
    # reach_pose = reach_pose * sapien.Pose(p=np.array([0, 0, 0.03]))

    ptc_array = None
    for actor in env.scene.actors.values():
        if actor.per_scene_id[ind_env] != env.obj.per_scene_id[ind_env]:
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
        else:
            aabb_size = actor.get_collision_meshes()[0].bounding_box.extents[0]
            reach_pose = sapien.Pose(p=env.obj.pose.sp.p - (aabb_size * 0.7 * push_direction))
            # a little ontop of the table
            reach_pose = reach_pose * sapien.Pose(p=np.array([0, 0, 0.03]))

    planner.add_collision_pts(ptc_array)

    # planner.set_robot_pose(reach_pose)
    # planner.move_to_pose_with_RRTConnect(reach_pose)
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(p=env.goal_region.pose.sp.p - (aabb_size * 0.5 * push_direction))
    goal_pose.set_p(np.array([*goal_pose.get_p()[:2], reach_pose.p[2]]))
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res
