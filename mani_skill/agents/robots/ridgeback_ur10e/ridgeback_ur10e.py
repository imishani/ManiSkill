from copy import deepcopy
from mani_skill.agents.controllers import PDBaseVelController, PDBaseVelControllerConfig, PDJointPosControllerConfig
from typing import Dict, Tuple

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import Array

RIDGEBACK_WHEELS_COLLISION_BIT = 30
"""Collision bit of the fetch robot wheel links"""
RIDGEBACK_BASE_COLLISION_BIT = 31
"""Collision bit of the fetch base"""


@register_agent()
class RidgebackUR10e(BaseAgent):
    uid = "ridgebackur10e"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/ridgeback_ur10e/ridgeback_ur10e.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [-1., 0, 0,
                 0., -1.0472, -2., 0., 1.5708, 0.,
                 0., 0.,
                 0, 0, 0, 0 # Passive joints for the gripper
                ]
            ),
            pose=sapien.Pose(),
        )
    )

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="front_camera_base_link",
                pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]), # TODO: fix pose
                width=128,
                height=128,
                fov=2,
                near=0.01,
                far=100,
                entity_uid="front_camera_mount",
            )
        ]

    def __init__(self, *args, **kwargs):
        self.arm_joint_names = [
            'ur_arm_shoulder_pan_joint',
            'ur_arm_shoulder_lift_joint',
            'ur_arm_elbow_joint',
            'ur_arm_wrist_1_joint',
            'ur_arm_wrist_2_joint',
            'ur_arm_wrist_3_joint'
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100

        self.gripper_joint_names = [
            "finger_joint",
            "left_inner_knuckle_joint",
            "left_inner_finger_joint",
            "right_inner_knuckle_joint",
            "right_inner_finger_joint",
        ]
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100

        self.ee_link_name = "ur_arm_TCP"

        self.base_joint_names = [
            "x",
            "y",
            "theta",
        ]

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness*10,
            self.arm_damping*10,
            self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD ee position (for human-interaction/teleoperation)
        arm_pd_ee_delta_pose_align = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_delta_pose_align.frame = "ee_align"

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=True,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #

        # define a passive controller config to simply "turn off" other joints from being controlled and set their properties (damping/friction) to 0.
        # these joints are controlled passively by the mimic controller later on.
        passive_finger_joint_names = [
            "left_inner_knuckle_joint",
            "right_inner_knuckle_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ]
        passive_finger_joints = PassiveControllerConfig(
            joint_names=passive_finger_joint_names,
            damping=0,
            friction=0,
        )

        finger_joint_names = ["left_outer_knuckle_joint", "right_outer_knuckle_joint"]
        # use a mimic controller config to define one action to control both fingers
        finger_mimic_pd_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=finger_joint_names,
            lower=0.0,
            upper=0.8,
            stiffness=1e5,
            damping=1e3,
            force_limit=0.1,
            friction=0.05,
            normalize_action=False,
        )
        finger_mimic_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            joint_names=finger_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=1e3,
            damping=1e3,
            force_limit=0.1,
            normalize_action=True,
            friction=0.05,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Base
        # -------------------------------------------------------------------------- #
        base_pd_joint_vel = PDBaseVelControllerConfig(
            self.base_joint_names,
            lower=[-1, -3.14, -3.14],
            upper=[1, 3.14, 3.14],
            damping=1000,
            force_limit=500,
        )

        base_pd_joint_pos = PDJointPosControllerConfig(
            self.base_joint_names,
            lower=[-3, -3.14, -3.14],
            upper=[3, 3.14, 3.14],
            stiffness=10e5,
            damping=1000,
            force_limit=500,
            normalize_action=False,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                base=base_pd_joint_vel,
                arm=arm_pd_joint_delta_pos,
                finger=finger_mimic_pd_joint_delta_pos,
                passive_finger_joints=passive_finger_joints
                # body=body_pd_joint_delta_pos,
            ),
            pd_joint_pos=dict(
                base=base_pd_joint_pos,
                arm=arm_pd_joint_pos,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
#                 body=body_pd_joint_delta_pos,

            ),
            pd_ee_delta_pos=dict(
                base=base_pd_joint_vel,
                arm=arm_pd_ee_delta_pos,
                finger=finger_mimic_pd_joint_delta_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
#                 body=body_pd_joint_delta_pos,

            ),
            pd_ee_delta_pose=dict(
                base=base_pd_joint_vel,
                arm=arm_pd_ee_delta_pose,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
#                 body=body_pd_joint_delta_pos,

            ),
            pd_ee_delta_pose_align=dict(
                base=base_pd_joint_vel,
                arm=arm_pd_ee_delta_pose_align,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
#                 body=body_pd_joint_delta_pos,

            ),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                base=base_pd_joint_vel,
                arm=arm_pd_joint_target_delta_pos,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
#                 body=body_pd_joint_delta_pos,

            ),
            pd_ee_target_delta_pos=dict(
                base=base_pd_joint_vel,
                arm=arm_pd_ee_target_delta_pos,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
#                 body=body_pd_joint_delta_pos,

            ),
            pd_ee_target_delta_pose=dict(
                base=base_pd_joint_vel,
                arm=arm_pd_ee_target_delta_pose,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
#                 body=body_pd_joint_delta_pos,

            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(
                base=base_pd_joint_vel,
                arm=arm_pd_joint_vel,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
#                 body=body_pd_joint_delta_pos,

            ),
            pd_joint_pos_vel=dict(
                base=base_pd_joint_vel,
                arm=arm_pd_joint_pos_vel,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
#                 body=body_pd_joint_delta_pos,

            ),
            pd_joint_delta_pos_vel=dict(
                base=base_pd_joint_vel,
                arm=arm_pd_joint_delta_pos_vel,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
#                 body=body_pd_joint_delta_pos,

            ),
            pd_joint_delta_pos_stiff_body=dict(
                base=base_pd_joint_vel,
                arm=arm_pd_joint_delta_pos,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
                # body=stiff_body_pd_joint_pos,

            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.finger1_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_inner_finger"
        )
        self.finger2_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_inner_finger"
        )
        self.tcp: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        self.base_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "ur_arm_base_link"
        )
        self.front_left_wheel_link: Link = self.robot.links_map["front_left_wheel_link"]
        self.front_right_wheel_link: Link = self.robot.links_map["front_right_wheel_link"]
        self.rear_left_wheel_link: Link = self.robot.links_map["rear_left_wheel_link"]
        self.rear_right_wheel_link: Link = self.robot.links_map["rear_right_wheel_link"]
        for link in [self.front_left_wheel_link, self.front_right_wheel_link,
            self.rear_left_wheel_link, self.rear_right_wheel_link]:
            link.set_collision_group_bit(
                group=2, bit_idx=RIDGEBACK_WHEELS_COLLISION_BIT, bit=1
            )
        self.base_link.set_collision_group_bit(
            group=2, bit_idx=RIDGEBACK_BASE_COLLISION_BIT, bit=1
        )

        self.queries: Dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
        ] = dict()

    def _after_loading_articulation(self):
        outer_finger = self.robot.active_joints_map["right_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["right_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        # the next 4 magic arrays come from https://github.com/haosulab/cvpr-tutorial-2022/blob/master/debug/robotiq.py which was
        # used to precompute these poses for drive creation
        p_f_right = [-1.6048949e-08, 3.7600022e-02, 4.3000020e-02]
        p_p_right = [1.3578170e-09, -1.7901104e-02, 6.5159947e-03]
        p_f_left = [-1.8080145e-08, 3.7600014e-02, 4.2999994e-02]
        p_p_left = [-1.4041154e-08, -1.7901093e-02, 6.5159872e-03]

        right_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_right), pad, sapien.Pose(p_p_right)
        )
        right_drive.set_limit_x(0, 0)
        right_drive.set_limit_y(0, 0)
        right_drive.set_limit_z(0, 0)

        outer_finger = self.robot.active_joints_map["left_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["left_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        left_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_left), pad, sapien.Pose(p_p_left)
        )
        left_drive.set_limit_x(0, 0)
        left_drive.set_limit_y(0, 0)
        left_drive.set_limit_z(0, 0)

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = -self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2, base_threshold: float = 0.05):
        body_qvel = self.robot.get_qvel()[..., 3:-2]
        base_qvel = self.robot.get_qvel()[..., :3]
        return torch.all(body_qvel <= threshold, dim=1) & torch.all(
            base_qvel <= base_threshold, dim=1
        )

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (panda_hand_tcp)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

    @property
    def tcp_pose(self) -> Pose: #TODO: fix this to use the tcp link
        p = (self.finger1_link.pose.p + self.finger2_link.pose.p) / 2
        q = (self.finger1_link.pose.q + self.finger2_link.pose.q) / 2
        return Pose.create_from_pq(p=p, q=q)




@register_agent()
class StaticRidgebackUR10e(BaseAgent):
    uid = "static_ridgebackur10e"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/ridgeback_ur10e/static_ridgeback_ur10e.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=4.0, dynamic_friction=4.0, restitution=0.0)
        ),
        link=dict(
            left_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [0., -1.0472, -2., 0., 1.5708, 0.,
                 0., 0.,
                 0, 0, 0, 0 # Passive joints for the gripper
                ]
            ),
            pose=sapien.Pose(),
        )
    )

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="front_camera_base_link",
                pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]), # TODO: fix pose
                width=128,
                height=128,
                fov=2,
                near=0.01,
                far=100,
                entity_uid="front_camera_mount",
            )
        ]

    def __init__(self, *args, **kwargs):
        self.arm_joint_names = [
            'ur_arm_shoulder_pan_joint',
            'ur_arm_shoulder_lift_joint',
            'ur_arm_elbow_joint',
            'ur_arm_wrist_1_joint',
            'ur_arm_wrist_2_joint',
            'ur_arm_wrist_3_joint'
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100

        self.gripper_joint_names = [
            "finger_joint",
            "left_inner_knuckle_joint",
            "left_inner_finger_joint",
            "right_inner_knuckle_joint",
            "right_inner_finger_joint",
        ]
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100

        self.ee_link_name = "ur_arm_TCP"

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness*10,
            self.arm_damping*10,
            self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness*10,
            damping=self.arm_damping*10,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness*10,
            damping=self.arm_damping*10,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path
        )

        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD ee position (for human-interaction/teleoperation)
        arm_pd_ee_delta_pose_align = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_delta_pose_align.frame = "ee_align"

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=True,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #

        # define a passive controller config to simply "turn off" other joints from being controlled and set their properties (damping/friction) to 0.
        # these joints are controlled passively by the mimic controller later on.
        passive_finger_joint_names = [
            "left_inner_knuckle_joint",
            "right_inner_knuckle_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ]
        passive_finger_joints = PassiveControllerConfig(
            joint_names=passive_finger_joint_names,
            damping=0,
            friction=0,
        )

        finger_joint_names = ["left_outer_knuckle_joint", "right_outer_knuckle_joint"]
        # use a mimic controller config to define one action to control both fingers
        finger_mimic_pd_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=finger_joint_names,
            lower=0.0,
            upper=0.8,
            stiffness=1e5,
            damping=1e3,
            force_limit=0.1,
            friction=0.05,
            normalize_action=False,
        )

        finger_mimic_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            joint_names=finger_joint_names,
            lower=0.0,
            upper=0.8,
            stiffness=1e5,
            damping=1e3,
            force_limit=0.1,
            normalize_action=True,
            friction=0.05,
            use_delta=False,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                finger=finger_mimic_pd_joint_delta_pos,
                passive_finger_joints=passive_finger_joints
                # body=body_pd_joint_delta_pos,
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
#                 body=body_pd_joint_delta_pos,

            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                finger=finger_mimic_pd_joint_delta_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
#                 body=body_pd_joint_delta_pos,

            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                # finger=finger_mimic_pd_joint_pos,
                finger=finger_mimic_pd_joint_delta_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
#                 body=body_pd_joint_delta_pos,

            ),
            pd_ee_delta_pose_align=dict(
                arm=arm_pd_ee_delta_pose_align,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
#                 body=body_pd_joint_delta_pos,

            ),
            pd_ee_pose=dict(arm=arm_pd_ee_pose,
                            finger=finger_mimic_pd_joint_pos,
                            passive_finger_joints=passive_finger_joints
                            ),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
#                 body=body_pd_joint_delta_pos,

            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
#                 body=body_pd_joint_delta_pos,

            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
#                 body=body_pd_joint_delta_pos,

            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(
                arm=arm_pd_joint_vel,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
#                 body=body_pd_joint_delta_pos,

            ),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
#                 body=body_pd_joint_delta_pos,

            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
#                 body=body_pd_joint_delta_pos,

            ),
            pd_joint_delta_pos_stiff_body=dict(
                arm=arm_pd_joint_delta_pos,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
                # gripper=gripper_pd_joint_pos,
                # body=stiff_body_pd_joint_pos,

            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.finger1_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_inner_finger"
        )
        self.finger2_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_inner_finger"
        )
        self.tcp: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        self.base_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "ur_arm_base_link"
        )
        self.front_left_wheel_link: Link = self.robot.links_map["front_left_wheel_link"]
        self.front_right_wheel_link: Link = self.robot.links_map["front_right_wheel_link"]
        self.rear_left_wheel_link: Link = self.robot.links_map["rear_left_wheel_link"]
        self.rear_right_wheel_link: Link = self.robot.links_map["rear_right_wheel_link"]
        for link in [self.front_left_wheel_link, self.front_right_wheel_link,
            self.rear_left_wheel_link, self.rear_right_wheel_link]:
            link.set_collision_group_bit(
                group=2, bit_idx=RIDGEBACK_WHEELS_COLLISION_BIT, bit=1
            )
        self.base_link.set_collision_group_bit(
            group=2, bit_idx=RIDGEBACK_BASE_COLLISION_BIT, bit=1
        )

        self.queries: Dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
        ] = dict()

    def _after_loading_articulation(self):
        outer_finger = self.robot.active_joints_map["right_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["right_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        # the next 4 magic arrays come from https://github.com/haosulab/cvpr-tutorial-2022/blob/master/debug/robotiq.py which was
        # used to precompute these poses for drive creation
        p_f_right = [-1.6048949e-08, 3.7600022e-02, 4.3000020e-02]
        p_p_right = [1.3578170e-09, -1.7901104e-02, 6.5159947e-03]
        p_f_left = [-1.8080145e-08, 3.7600014e-02, 4.2999994e-02]
        p_p_left = [-1.4041154e-08, -1.7901093e-02, 6.5159872e-03]

        right_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_right), pad, sapien.Pose(p_p_right)
        )
        right_drive.set_limit_x(0, 0)
        right_drive.set_limit_y(0, 0)
        right_drive.set_limit_z(0, 0)

        outer_finger = self.robot.active_joints_map["left_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["left_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        left_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_left), pad, sapien.Pose(p_p_left)
        )
        left_drive.set_limit_x(0, 0)
        left_drive.set_limit_y(0, 0)
        left_drive.set_limit_z(0, 0)

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = -self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        body_qvel = self.robot.get_qvel()
        return torch.all(body_qvel <= threshold, dim=1)

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (panda_hand_tcp)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

    @property
    def tcp_pose(self) -> Pose: #TODO: fix this to use the tcp link
        # p = (self.finger1_link.pose.p + self.finger2_link.pose.p) / 2
        # q = (self.finger1_link.pose.q + self.finger2_link.pose.q) / 2
        # return Pose.create_from_pq(p=p, q=q)
        return self.tcp.pose


