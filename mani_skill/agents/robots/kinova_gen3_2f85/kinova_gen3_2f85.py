from copy import deepcopy

import numpy as np
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDEEPosControllerConfig,
    PDEEPoseControllerConfig,
    PDJointPosControllerConfig,
    PDJointPosMimicControllerConfig,
    PDJointPosVelControllerConfig,
    PDJointVelControllerConfig,
)
from mani_skill.agents.registration import register_agent


def _deepcopy_dict(configs):
    return {k: deepcopy(v) for k, v in configs.items()}


@register_agent()
class KinovaGen3_2F85(BaseAgent):
    uid = "kinova_gen3_2f85"
    urdf_path = (
        f"{PACKAGE_ASSET_DIR}/robots/kinova_gen3_2f85/kinova_gen3_2f85.urdf"
    )

    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=8.0, dynamic_friction=8.0, restitution=0.0)
        ),
        link={
            name: dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1)
            for name in (
                "gen3_robotiq_85_left_knuckle_link",
                "gen3_robotiq_85_right_knuckle_link",
                "gen3_robotiq_85_left_finger_link",
                "gen3_robotiq_85_right_finger_link",
                "gen3_robotiq_85_left_inner_knuckle_link",
                "gen3_robotiq_85_right_inner_knuckle_link",
                "gen3_robotiq_85_left_finger_tip_link",
                "gen3_robotiq_85_right_finger_tip_link",
            )
        },
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    0.2618,
                    3.1416,
                    -2.2689,
                    0.0,
                    0.9599,
                    1.5708,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            pose=sapien.Pose(),
        )
    )

    def __init__(self, *args, **kwargs):
        self.arm_joint_names = [
            "gen3_joint_1",
            "gen3_joint_2",
            "gen3_joint_3",
            "gen3_joint_4",
            "gen3_joint_5",
            "gen3_joint_6",
            "gen3_joint_7",
        ]
        self.ee_link_name = "gen3_robotiq_85_tcp"

        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 120

        self.gripper_stiffness = 1e5
        self.gripper_damping = 1e3
        # The default mimic-joint force/friction was too weak for sustained
        # handle pulls in the grocery fridge scenarios. Keep this below the
        # UR10e/Panda 100 N-style settings, but high enough for form closure.
        self.gripper_force_limit = 50.0
        self.gripper_friction = 4.0

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness * 10,
            damping=self.arm_damping * 10,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

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
            frame="body_translation:body_aligned_body_rotation",
            cpu_differential_ik=True,
            differential_joint_target=True,
            max_differential_target_lag=1.0,
            delta_solver_config=dict(
                type="levenberg_marquardt", alpha=1.0, damping=1e-3, max_joint_delta=0.08
            ),
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True
        arm_pd_ee_target_delta_pose.frame = "root_translation:root_aligned_body_rotation"
        arm_pd_ee_target_delta_pose.cpu_differential_ik = False
        arm_pd_ee_target_delta_pose.differential_joint_target = False

        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,
            self.arm_force_limit,
        )
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )

        finger_joint_names = [
            "gen3_robotiq_85_left_knuckle_joint",
            "gen3_robotiq_85_right_knuckle_joint",
            "gen3_robotiq_85_left_inner_knuckle_joint",
            "gen3_robotiq_85_right_inner_knuckle_joint",
            "gen3_robotiq_85_left_finger_tip_joint",
            "gen3_robotiq_85_right_finger_tip_joint",
        ]

        mimic_cfg = dict(
            gen3_robotiq_85_right_knuckle_joint=dict(
                joint="gen3_robotiq_85_left_knuckle_joint",
                multiplier=-1.0,
                offset=0.0,
            ),
            gen3_robotiq_85_left_inner_knuckle_joint=dict(
                joint="gen3_robotiq_85_left_knuckle_joint",
                multiplier=1.0,
                offset=0.0,
            ),
            gen3_robotiq_85_right_inner_knuckle_joint=dict(
                joint="gen3_robotiq_85_left_knuckle_joint",
                multiplier=-1.0,
                offset=0.0,
            ),
            gen3_robotiq_85_left_finger_tip_joint=dict(
                joint="gen3_robotiq_85_left_knuckle_joint",
                multiplier=-1.0,
                offset=0.0,
            ),
            gen3_robotiq_85_right_finger_tip_joint=dict(
                joint="gen3_robotiq_85_left_knuckle_joint",
                multiplier=1.0,
                offset=0.0,
            ),
        )

        finger_mimic_pd_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=finger_joint_names,
            lower=0.0,
            upper=0.8,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=False,
            mimic=mimic_cfg,
        )

        finger_mimic_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            joint_names=finger_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=True,
            use_delta=True,
            mimic=mimic_cfg,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                finger=finger_mimic_pd_joint_delta_pos,
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                finger=finger_mimic_pd_joint_pos,
            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                finger=finger_mimic_pd_joint_delta_pos,
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                # Match UR10e 2f85 behavior: EE delta-pose arm control with
                # absolute-position gripper control (not delta gripper control).
                finger=finger_mimic_pd_joint_pos,
            ),
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos,
                finger=finger_mimic_pd_joint_pos,
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos,
                finger=finger_mimic_pd_joint_pos,
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose,
                finger=finger_mimic_pd_joint_pos,
            ),
            pd_joint_vel=dict(
                arm=arm_pd_joint_vel,
                finger=finger_mimic_pd_joint_pos,
            ),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel,
                finger=finger_mimic_pd_joint_pos,
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel,
                finger=finger_mimic_pd_joint_pos,
            ),
        )

        return _deepcopy_dict(controller_configs)

    def _after_init(self):
        self.tcp = self.robot.links_map[self.ee_link_name]
