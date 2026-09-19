import copy

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor


@register_agent()
class SO107(BaseAgent):
    """
    SO-107: an SO-101 with an extra continuous `forearm_roll` joint inserted between
    `elbow_flex` and `wrist_flex`. See mani_skill/assets/robots/so107/README.md for provenance.
    """

    uid = "so107"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/so107/so107_lab_soft_wowrobo.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2, dynamic_friction=2, restitution=0.0)
        ),
        link=dict(
            fixed_finger_link=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            moving_finger_link=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0, -1.5708, 1.5708, 0, 0.66, 0, -1.1]),
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        ),
        zero=Keyframe(
            qpos=np.array([0.0] * 7),
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        ),
    )

    arm_joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "forearm_roll",
        "wrist_flex",
        "wrist_roll",
    ]
    gripper_joint_names = [
        "gripper",
    ]
    ee_link_name = "gripper_frame_link"
    kinematics_root_link_name = "base_link"

    # Measured from the URDF collision meshes in gripper_frame_link coordinates.
    gripper_open_command = 0.8
    gripper_closed_command = -0.174533
    grasp_closing_axis_index = 0
    require_vertical_grasp_closing_axis = True
    antipodal_grasp_box_center = np.array([0.0, 0.0, -0.015])
    antipodal_grasp_box_extents = np.array([0.070, 0.030, 0.020])
    grasp_contact_min_force = 0.01

    @property
    def _controller_configs(self):
        pd_joint_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            lower=None,
            upper=None,
            stiffness=[1e3] * 7,
            damping=[1e2] * 7,
            force_limit=100,
            normalize_action=False,
        )

        # max delta permitted of 0.05 since the robot is not as accurate as more expensive arms
        # and moving too fast can cause the robot to shake too much and damage the hardware
        pd_joint_delta_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            [-0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.2],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2],
            stiffness=[1e3] * 7,
            damping=[1e2] * 7,
            force_limit=100,
            use_delta=True,
            use_target=False,
        )

        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=10,
            normalize_action=False,
        )
        gripper_pd_joint_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            None,
            None,
            stiffness=1e3,
            damping=1e2,
            force_limit=1.0,
            normalize_action=False,
        )

        controller_configs = dict(
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_pos=pd_joint_pos,
            pd_joint_target_delta_pos=pd_joint_target_delta_pos,
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel, gripper=gripper_pd_joint_pos
            ),
        )
        return deepcopy_dict(controller_configs)

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="wrist_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            )
        ]

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.finger1_link = self.robot.links_map["fixed_finger_link"]
        self.finger2_link = self.robot.links_map["moving_finger_link"]
        # generic single-link TCP expected by built-in tabletop tasks (e.g. PushCube-v1);
        # reuses gripper_frame_link, the same frame lerobot's own kinematics/IK targets.
        self.tcp = self.robot.links_map["gripper_frame_link"]

    def has_bilateral_gripper_contact(self, object: Actor, min_force=0.5, max_angle=110):
        min_force = min(float(min_force), self.grasp_contact_min_force)
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        lflag = torch.logical_and(
            lforce >= min_force,
            torch.rad2deg(common.compute_angle_between(ldirection, l_contact_forces)) <= max_angle,
        )
        rflag = torch.logical_and(
            rforce >= min_force,
            torch.rad2deg(common.compute_angle_between(rdirection, r_contact_forces)) <= max_angle,
        )
        return torch.logical_and(lflag, rflag)

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=60):
        return self.has_bilateral_gripper_contact(
            object, min_force=min_force, max_angle=max_angle
        )

    def is_static(self, threshold=0.2):
        qvel = self.robot.get_qvel()[:, :-1]  # exclude the gripper joint
        return torch.max(torch.abs(qvel), 1)[0] <= threshold
