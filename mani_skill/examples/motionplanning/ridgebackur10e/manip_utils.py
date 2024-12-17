import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from transforms3d import quaternions, euler

# # torch euler angles to quaternion
# def euler_to_quaternion(euler_angles: torch.Tensor,
#                         order='zyx',
#                         tensor_args: dict = None):
#     """
#     Convert euler angles to quaternion
#     :param euler_angles: euler angles in radians
#     :param order: euler angles order
#     :param tensor_args: tensor arguments
#     :return: quaternion
#     """
#     if tensor_args is None:
#         tensor_args = {"dtype": torch.float32, "device": 'cpu'}
#     r = R.from_euler(order, euler_angles.cpu().numpy())
#     return torch.tensor(r.as_quat(), **tensor_args)

def sample_grasp_ee_poses(obj_aabb: np.ndarray or torch.Tensor,
                          n_samples: int,
                          robot,
                          ik_func,
                          is_coll_func,
                          tensor_args: dict,
                          parallel=False,
                          **kwargs):
    """
    Sample grasp end-effector poses for a given object AABB
    :param obj_aabb: AABB of the object
    :param n_samples: number of samples
    :param robot: robot model
    :param ik_func: inverse kinematics function
    :param is_coll_func: collision checker function
    :param tensor_args: tensor arguments
    :param parallel: whether to sample in parallel
    :return: grasp end-effector poses and joint configurations
    """
    ee_samples = []
    obj_min, obj_max = obj_aabb[0, :], obj_aabb[1, :]
    # to tensor
    obj_min = torch.tensor(obj_min, **tensor_args)
    obj_max = torch.tensor(obj_max, **tensor_args)
    if parallel:
        # sample in batch
        ee_positions = torch.rand(n_samples, 3, **tensor_args) * (obj_max - obj_min) + obj_min
        ee_orientations = (torch.rand(n_samples, 3, **tensor_args) * 2 * torch.pi) - torch.pi
        # ee_orientations = torch.tensor([0, torch.pi / 2, 0], **tensor_args).repeat(n_samples, 1)
        ee_orientations = euler_to_quaternion(ee_orientations, tensor_args=tensor_args)
        ik_solutions, idx = ik_func(robot, ee_positions, ee_orientations, tensor_args, **kwargs)
        if ik_solutions is not None:
            for i, j in zip(range(ik_solutions.shape[0]), idx):
                if ik_solutions[i] is not None and is_coll_func(ik_solutions[i]) == False:
                    ee_samples.append((ee_positions[j], ee_orientations[j], ik_solutions[i]))
    else:
        for _ in range(n_samples):
            # Sample a random point within the AABB
            ee_position = torch.rand(3) * (obj_max - obj_min) + obj_min
            # Sample a random orientation
            ee_orientation = torch.rand(3) * 2 * np.pi - np.pi
            # quaternion
            ee_orientation = euler.euler2quat(ee_orientation[0].item(), ee_orientation[1].item(), ee_orientation[2].item())
            # position_for_ik = ee_position + torch.tensor([0.0, 0.0, 0.1], **tensor_args) # TODO:REMOVE THIS
            ik_solution = ik_func(robot, ee_position, ee_orientation, tensor_args, **kwargs)
            if ik_solution is not None and is_coll_func(ik_solution) == False:
                ee_samples.append((ee_position, ee_orientation, ik_solution))
    return ee_samples


def get_antipodal_score(robot_joint_angles: torch.Tensor or np.ndarray,
                        pc,
                        normals,
                        robot,
                        tensor_args: dict):
    """
    Get antipodal score
    :param robot_joint_angles:
    :param pc:
    :param normals:
    :param robot:
    :param tensor_args:
    :return:
    """
    pose_matrix = robot.get_EE_pose(robot_joint_angles)
    if pose_matrix.shape[0] != 1:
        return 0

    score = 0

    pose_matrix = pose_matrix.squeeze()
    # gripper_line_vector = pose_matrix[:3, 3]
    gripper_line_vector = torch.tensor([0, 0.2, 0], **tensor_args)
    #local to global frame
    gripper_line_vector = torch.matmul(pose_matrix, torch.cat((gripper_line_vector, torch.tensor([1.0], **tensor_args))))
    gripper_line_vector = gripper_line_vector[:3]
    gripper_line_vector -= pose_matrix[:3, 3]
    gripper_line_vector = gripper_line_vector / torch.norm(gripper_line_vector)

    box = o3d.geometry.OrientedBoundingBox(pose_matrix.cpu().numpy()[:3, 3],
                                           pose_matrix.cpu().numpy()[:3, :3],
                                           np.array([0.02, 0.08, 0.02]))

    # box.rotate(pose_matrix.cpu().numpy()[:3, :3])
    # box.translate(pose_matrix.cpu().numpy()[:3, 3])
    # box.extent = torch.tensor([0.02, 0.08, 0.02])

    # get indices of points within the box
    if type(pc) == torch.Tensor:
        pc = [x.cpu().numpy() for x in pc]
    elif type(pc) == np.ndarray:
        pc = [x for x in pc]
    elif type(pc) == list:
        pc = pc
    else:
        raise ValueError(f"Invalid type {type(pc)}")
    pc = o3d.utility.Vector3dVector(pc)
    indices = box.get_point_indices_within_bounding_box(pc)
    # get the normals of the points within the box
    if type(normals) == torch.Tensor:
        normals = [x.cpu().numpy() for x in normals]
    elif type(normals) == np.ndarray:
        normals = [x for x in normals]
    elif type(normals) == list:
        normals = normals
    else:
        raise ValueError(f"Invalid type {type(normals)}")
    normals_in_gripper = [normals[i] for i in indices]
    normals_in_gripper = np.array(normals_in_gripper)
    if len(normals_in_gripper) > 0:
        score = np.mean(np.abs(np.dot(normals_in_gripper, gripper_line_vector.cpu().numpy())))

        # # show the normals
        # pcd_vis = o3d.geometry.PointCloud()
        # pcd_vis.points = o3d.utility.Vector3dVector([pc[i] for i in indices])
        # pcd_vis.normals = o3d.utility.Vector3dVector(normals)
        # pcd_full = o3d.geometry.PointCloud()
        # pcd_full.points = o3d.utility.Vector3dVector(pc)
        # pcd_full.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # o3d.visualization.draw_geometries([pcd_full, box, pcd_vis])

    return score

def get_point_cloud_from_mesh(mesh: o3d.geometry.TriangleMesh,
                              n_points: int,
                              **kwargs):
    """
    Get point cloud from mesh
    :param mesh: mesh
    :param n_points: number of points
    :return: point cloud
    """
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    # pcd = torch.tensor(np.asarray(pcd.points), **tensor_args)
    return pcd


def get_point_cloud_from_moveit(object_name, ns="", plot=False, **kwargs):
    scene = moveit_commander.PlanningSceneInterface(ns)
    obj = scene.get_objects([object_name])[object_name]
    if len(obj.meshes) == 0 and len(obj.primitives) == 0:
        raise ValueError("Object has no meshes or primitives")
    elif len(obj.meshes) > 0:

        mesh = obj.meshes[0]
        mesh_o3d = o3d.geometry.TriangleMesh()
        # get aabb
        vertices = np.array([[mesh.vertices[i].x, mesh.vertices[i].y, mesh.vertices[i].z] for i in range(len(mesh.vertices))])
        # Transform the vertices to the correct frame
        obj_position = obj.pose.position
        obj_orientation = obj.pose.orientation
        obj_orientation = [obj_orientation.x, obj_orientation.y, obj_orientation.z, obj_orientation.w]
        obj_orientation = R.from_quat(obj_orientation).as_matrix()
        vertices = np.dot(vertices, obj_orientation.T) + np.array([obj_position.x, obj_position.y, obj_position.z])
        triangles = np.array([[mesh.triangles[i].vertex_indices[0], mesh.triangles[i].vertex_indices[1], mesh.triangles[i].vertex_indices[2]] for i in range(len(mesh.triangles))])
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(triangles)
        aabb_o3d = mesh_o3d.get_axis_aligned_bounding_box()
        # to numpy
        aabb = np.array([[aabb_o3d.min_bound[0], aabb_o3d.min_bound[1], aabb_o3d.min_bound[2]],
                         [aabb_o3d.max_bound[0], aabb_o3d.max_bound[1], aabb_o3d.max_bound[2]]])
        # # add margin to aabb
        # margin = 0.05
        # aabb[0, :] -= margin * np.ones(3)
        # aabb[1, :] += margin * np.ones(3)
        # # get the enveloping point cloud
        # convex_hull = mesh_o3d.compute_convex_hull()
        # pcd = o3d.geometry.PointCloud()
        # vertices = np.asarray(convex_hull[0].vertices)
        # pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd = get_point_cloud_from_mesh(mesh_o3d, 1500)

    else:
        obj_position = obj.pose.position
        obj_orientation = obj.pose.orientation
        prim = obj.primitives[0]
        import shape_msgs.msg
        if prim.type == shape_msgs.msg.SolidPrimitive.BOX:
            pcd = o3d.geometry.TriangleMesh.create_box(width=prim.dimensions[0], height=prim.dimensions[1], depth=prim.dimensions[2])
            obj_orientation = [obj_orientation.x, obj_orientation.y, obj_orientation.z, obj_orientation.w]
            obj_orientation = R.from_quat(obj_orientation).as_matrix()
            pcd.rotate(obj_orientation)
            pcd.translate([obj_position.x, obj_position.y, obj_position.z])
            aabb_o3d = pcd.get_axis_aligned_bounding_box()
        elif prim.type == shape_msgs.msg.SolidPrimitive.SPHERE:
            pcd = o3d.geometry.TriangleMesh.create_sphere(radius=prim.dimensions[0])
            obj_orientation = [obj_orientation.x, obj_orientation.y, obj_orientation.z, obj_orientation.w]
            obj_orientation = R.from_quat(obj_orientation).as_matrix()
            pcd.rotate(obj_orientation)
            pcd.translate([obj_position.x, obj_position.y, obj_position.z])
            aabb_o3d = pcd.get_axis_aligned_bounding_box()
        elif prim.type == shape_msgs.msg.SolidPrimitive.CYLINDER:
            pcd = o3d.geometry.TriangleMesh.create_cylinder(radius=prim.dimensions[1], height=prim.dimensions[0])
            obj_orientation = [obj_orientation.x, obj_orientation.y, obj_orientation.z, obj_orientation.w]
            obj_orientation = R.from_quat(obj_orientation).as_matrix()
            pcd.rotate(obj_orientation)
            pcd.translate([obj_position.x, obj_position.y, obj_position.z])
            aabb_o3d = pcd.get_axis_aligned_bounding_box()
        else:
            raise ValueError(f"Primitive type {prim.type} not supported.")
        aabb = np.array([[aabb_o3d.min_bound[0], aabb_o3d.min_bound[1], aabb_o3d.min_bound[2]],
                         [aabb_o3d.max_bound[0], aabb_o3d.max_bound[1], aabb_o3d.max_bound[2]]])
        # # add margin to aabb
        # margin = 0.05
        # aabb[0, :] -= margin * np.ones(3)
        # aabb[1, :] += margin * np.ones(3)
        pcd = get_point_cloud_from_mesh(pcd, 1500)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=50)

    if plot:
        # o3d.visualization.draw_geometries([pcd])
        # plot bnounding box
        o3d.visualization.draw_geometries([pcd, aabb_o3d],
                                          point_show_normal=True,  # Show the normals as small lines
                                          width=800,
                                          height=600,
                                          mesh_show_back_face=True)
        # o3d.visualization.draw_plotly([pcd, aabb_obj])

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    return points, normals, aabb

def pk_inverse_kinematics(robot: RobotRidgebackUR10e,
                          ee_position, ee_orientation,
                          tensor_args,
                          **kwargs):
    """
    Inverse kinematics function
    :param robot: robot model
    :param ee_position: end-effector position
    :param ee_orientation: end-effector orientation
    :param tensor_args:
    :return: joint configuration
    """
    import pytorch_kinematics as pk
    from pytorch_kinematics import Transform3d

    serial_chain = pk.SerialChain(robot.diff_robot, robot.link_name_ee, "world_link", **tensor_args)

    pose_world_frame = Transform3d(pos=ee_position, rot=ee_orientation, **tensor_args)
    # get robot joint limits
    lim = torch.tensor(serial_chain.get_joint_limits(), device=tensor_args['device'])
    lim[0, 2] = 0
    lim[1, 2] = 2 * torch.pi
    lim[0, 3] = -0.47
    lim[1, 3] = 3.641
    lim[0, 4] = -1.5*torch.pi
    lim[1, 4] = 0
    ik = pk.PseudoInverseIK(serial_chain,
                            max_iterations=kwargs.get("max_iterations", 100),
                            num_retries=kwargs.get("num_retries", 10),
                            joint_limits=lim.T,
                            early_stopping_any_converged=kwargs.get("early_stopping_any_converged", True),
                            early_stopping_no_improvement=kwargs.get("early_stopping_no_improvement", "all"),
                            debug=kwargs.get("debug", False),
                            lr=kwargs.get("lr", 0.2))
    sol = ik.solve(pose_world_frame)
    # # num goals x num retries x DOF tensor of joint angles; if not converged, best solution found so far
    # print(sol.solutions)
    # # num goals x num retries can check for the convergence of each run
    # print(sol.converged)
    # # num goals x num retries can look at errors directly
    # print(sol.err_pos)
    # print(sol.err_rot)
    # # return the best solution
    if (sol.converged == False).all():
        return None, None
    # elif more than one solution is converged, return the best one
    elif (sol.converged == True).any(axis=-1).sum() > 1:
        idx = torch.argmin(sol.err_pos, axis=1)
        solut = torch.tensor([], **tensor_args)
        indices = (sol.converged == True).any(axis=-1).nonzero().squeeze()
        for i, b in enumerate((sol.converged == True).any(axis=-1)):
            if b:
                # solut = torch.cat((solut, sol.solutions[i, idx[i], :].unsqueeze(0)), dim=0)
                # check which index is converged and pick one
                converged = sol.converged[i]
                converged_idx = torch.argwhere(converged == True).squeeze()
                if converged_idx.nelement() == 0:
                    continue
                # elif it is single number
                elif converged_idx.nelement() == 1:
                    solut = torch.cat((solut, sol.solutions[i, converged_idx, :].unsqueeze(0)), dim=0)
                else:
                    solut = torch.cat((solut, sol.solutions[i, converged_idx[0], :].unsqueeze(0)), dim=0)
                # solut = torch.cat((solut, sol.solutions[i, 0, :].unsqueeze(0)), dim=0)
        return solut, indices
    else:
        return sol.solutions[sol.converged, :], (sol.converged == True).any(axis=-1).nonzero().squeeze()


def find_best_grasp(robot: RobotRidgebackUR10e,
                    collision_checker,
                    object_name: str,
                    ns: str,
                    n_samples: int,
                    tensor_args: dict,
                    **kwargs):
    """
    Find the best grasp
    :param robot: robot model
    :param collision_checker:
    :param object_name: object name
    :param ns: namespace
    :param n_samples: number of samples
    :param tensor_args: tensor arguments
    :return: best grasp. Tuple of end-effector position, orientation and joint configuration
    """
    pc, normals, aabb = get_point_cloud_from_moveit(object_name, ns, **kwargs)
    poses_conf = sample_grasp_ee_poses(aabb, n_samples, robot, pk_inverse_kinematics, collision_checker,
                                       tensor_args, parallel=True, **kwargs)
    best_score = 0
    best_grasp = None
    for p in poses_conf:
        score = get_antipodal_score(p[2], pc, normals, robot, tensor_args)
        if score > best_score:
            best_score = score
            best_grasp = p
    return best_grasp


def get_pregrasp_joint_state(robot: RobotRidgebackUR10e,
                             joint_state,
                             relative_position,
                             ik_func,
                             tensor_args):
    """

    :param robot:
    :param grasp_pose:
    :param joint_state:
    :param relative_position:
    :param ik_func:
    :param tensor_args:
    :return:
    """
    ee_pose = robot.get_EE_pose(joint_state)
    ee_position = ee_pose[:, :3, 3]
    ee_orientation = ee_pose[:, :3, :3]
    # to quaternion
    ee_orientation = R.from_matrix(ee_orientation.cpu().numpy()).as_quat()
    ee_orientation = torch.tensor(ee_orientation, **tensor_args)
    ee_position = ee_position + relative_position
    ik_solution = ik_func(robot, ee_position, ee_orientation, tensor_args)
    return ik_solution

def test_ik():
    # from robot_ridgebackur10e import RobotRidgebackUR10e

    tensor_args = {'device': get_device(), 'dtype': torch.float32}
    robot = RobotRidgebackUR10e(tensor_args=tensor_args)
    ee_position = torch.tensor([0.5, 0.5, 1.3], **tensor_args)
    ee_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], **tensor_args)
    q = pk_inverse_kinematics(robot, ee_position, ee_orientation, tensor_args)
    print(f"Checking if the IK was successful. If True, the joint angles are: {q}")
    pose = robot.get_EE_pose(q)
    print(f"End-effector pose: {pose}")
    print(f"End-effector position: {pose[:3, 3]}")
    print(f"End-effector orientation: {pose[:3, :3]}")
    quat = pose[0, :3, :3]
    quat = R.from_matrix(quat).as_quat()
    quat = torch.tensor(quat, **tensor_args)
    quat = quat[[3, 0, 1, 2]]
    assert torch.allclose(ee_position, pose[0, :3, 3], atol=1e-2), "Position is not close enough"
    # get the quaternion distance
    quat = quat / torch.norm(quat)
    ee_orientation = ee_orientation / torch.norm(ee_orientation)

    dot = torch.abs(torch.sum(quat * ee_orientation))
    # Clamp to handle numerical errors
    dot = torch.clamp(dot, -1.0, 1.0)
    # Return normalized distance between 0 and 1
    assert torch.acos(2 * dot * dot - 1) < 1e-2, "Orientation is not close enough"

def test_grasp():
    # from robot_ridgebackur10e import RobotRidgebackUR10e
    # from mpd_moveit_envs import MoveItEnv
    # from torch_robotics.tasks.tasks import PlanningTask

    tensor_args = {'device': get_device(), 'dtype': torch.float32}
    env = MoveItEnv(group_name="manipulator",
                    precompute_sdf_obj_fixed=True,
                    sdf_cell_size=0.02,
                    tensor_args=tensor_args)
    robot = RobotRidgebackUR10e(tensor_args=tensor_args)

    task = PlanningTask(
        env=env,
        robot=robot,
        ws_limits=torch.tensor([[-3.5, -2., 0.0], [2.5, 3.5, 3.0]], **tensor_args),  # workspace limits
        obstacle_cutoff_margin=0.01,
        tensor_args=tensor_args
    )

    best_grasp = find_best_grasp(robot, task.compute_collision, "Cylinder_1", "",
                                 500, tensor_args, plot=True, debug=False)
    # best_grasp = find_best_grasp(robot, task.compute_collision, "bowl1", "", 500, tensor_args)
    # best_grasp = find_best_grasp(robot, task.compute_collision, "mug2", "",
    #                              500, tensor_args, plot=True)
    if best_grasp is None:
        print("No valid grasp found")
        return
    q = best_grasp[2].unsqueeze(0)
    fk = robot.fk_map_collision_impl(q)
    from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
    import matplotlib.pyplot as plt
    fig, ax = create_fig_and_axes(dim=3)
    robot.render(ax, q, draw_links_spheres=True)
    env.render(ax)
    plt.show()

    # move the robot to the grasp
    import sys
    import rospy
    from moveit_commander import RobotCommander, PlanningSceneInterface
    from moveit_msgs.msg import Grasp
    rospy.init_node('ridgebackur10e_spheres_HybridPlanner', anonymous=True)

    moveit_commander.roscpp_initialize(sys.argv)
    scene = PlanningSceneInterface()
    rospy.sleep(2)

    move_group = moveit_commander.MoveGroupCommander("full_body")
    start_state_target = move_group.get_current_joint_values()
    # allow collision with the object
    # robot_model.
    # robot_model.allow_collision("bowl1", "world_link", True)

    # q = get_pregrasp_joint_state(robot, q, torch.tensor([0.0, 0.0, 0.1], **tensor_args), pk_inverse_kinematics, tensor_args)
    start_state_target = q.squeeze().cpu().numpy().astype(np.float64).tolist()
    move_group.set_start_state_to_current_state()
    #### TEMP: TODO: REMOVE THIS
    object = scene.get_objects(["Cylinder_1"])["Cylinder_1"]
    object.operation = object.REMOVE
    scene.add_object(object)
    ####
    move_group.set_joint_value_target(start_state_target)
    move_group.set_max_velocity_scaling_factor(0.8)
    move_group.set_max_acceleration_scaling_factor(0.8)
    move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    #### TEMP: TODO: REMOVE THIS
    object.operation = object.ADD
    scene.add_object(object)
    ####

    print("Grasp done")


if __name__ == "__main__":
    test_ik()
    test_grasp()
