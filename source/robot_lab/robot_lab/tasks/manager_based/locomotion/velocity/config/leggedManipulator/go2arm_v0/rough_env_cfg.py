# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

# from isaaclab_nhb.terrain.config.rough import ROUGH_TERRAINS_SIMPLE_CFG

from robot_lab.tasks.manager_based.locomotion.velocity.cus_velocity_env_cfg import LocomotionVelocityRoughEnvCfg

# from isaaclab.sensors import RayCasterCfg, patterns

##
# Pre-defined configs
##
# # use cloud assets
# from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
# use local assets
from robot_lab.assets.unitree import UNITREE_Go2Arm_CFG  # isort: skip

# 继承velocity_env_cfg里的LocomotionVelocityRoughEnvCfg(速度控制机器人)类，
# 并指定实例名UnitreeGo2ArmRoughEnvCfg，与gym.register里的env_cfg_entry_point一致，并重写了部分配置
@configclass
class UnitreeGo2ArmRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name: list[str] = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    
    reward_group_terms: dict[str, list[str]] | None = None
    reward_group_strict: bool = True

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_Go2Arm_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # 新添加的足端高度扫描器（这里暂时不用，因为足端位置可由base位置经正运动学推动后直接从Sim里读取，增加新的扫描器会有多余的资源开销）
        self.scene.FL_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.foot_link_name[0]
        self.scene.FR_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.foot_link_name[1]
        self.scene.RL_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.foot_link_name[2]
        self.scene.RR_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.foot_link_name[3]

        # ------------------------------Observations------------------------------

        # Policy observations scale and clip configuration
        # 按照 cus_velocity_env_cfg.py 中 obs 配置的顺序
        self.observations.policy.joint_pos_history.scale = 1.0
        self.observations.policy.joint_pos_history.clip = (-3.14, 3.14)
        self.observations.policy.projected_gravity.scale = 1.0
        self.observations.policy.projected_gravity.clip = (-1.0, 1.0)
        self.observations.policy.base_lin_vel.scale = 1.0
        self.observations.policy.base_lin_vel.clip = (-5.0, 5.0)
        self.observations.policy.base_ang_vel.scale = 1.0
        self.observations.policy.base_ang_vel.clip = (-5.0, 5.0)
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_pos.clip = (-3.14, 3.14)
        self.observations.policy.joint_vel.scale = 1.0
        self.observations.policy.joint_vel.clip = (-20.0, 20.0)
        self.observations.policy.actions.scale = 1.0
        self.observations.policy.actions.clip = (-1.0, 1.0)
        self.observations.policy.base_velocity_command.scale = 1.0
        self.observations.policy.base_velocity_command.clip = (-1.5, 1.5)
        self.observations.policy.ee_twist_command.scale = 1.0
        self.observations.policy.ee_twist_command.clip = (-5.0, 5.0)
        self.observations.policy.feet_swing_height_command.scale = 1.0
        self.observations.policy.feet_swing_height_command.clip = (0.0, 0.2)

        # Critic observations scale and clip configuration
        self.observations.critic.joint_pos_history.scale = 1.0
        self.observations.critic.joint_pos_history.clip = (-3.14, 3.14)
        self.observations.critic.projected_gravity.scale = 1.0
        self.observations.critic.projected_gravity.clip = (-1.0, 1.0)
        self.observations.critic.base_lin_vel.scale = 1.0
        self.observations.critic.base_lin_vel.clip = (-5.0, 5.0)
        self.observations.critic.base_ang_vel.scale = 1.0
        self.observations.critic.base_ang_vel.clip = (-5.0, 5.0)
        self.observations.critic.joint_pos.scale = 1.0
        self.observations.critic.joint_pos.clip = (-3.14, 3.14)
        self.observations.critic.joint_vel.scale = 1.0
        self.observations.critic.joint_vel.clip = (-20.0, 20.0)
        self.observations.critic.feet_contact_state.scale = 1.0
        self.observations.critic.feet_contact_state.clip = (-1.0, 1.0)
        self.observations.critic.static_friction.scale = 1.0
        self.observations.critic.static_friction.clip = (0.0, 2.0)
        self.observations.critic.feet_air_time.scale = 1.0
        self.observations.critic.feet_air_time.clip = (0.0, 2.0)
        self.observations.critic.base_external_wrench.scale = 1.0
        self.observations.critic.base_external_wrench.clip = (-100.0, 100.0)
        self.observations.critic.base_external_push_velocity.scale = 1.0
        self.observations.critic.base_external_push_velocity.clip = (-10.0, 10.0)
        self.observations.critic.base_mass_disturbance.scale = 1.0
        self.observations.critic.base_mass_disturbance.clip = (-10.0, 10.0)
        self.observations.critic.ee_external_wrench.scale = 1.0
        self.observations.critic.ee_external_wrench.clip = (-100.0, 100.0)
        self.observations.critic.ee_mass_disturbance.scale = 1.0
        self.observations.critic.ee_mass_disturbance.clip = (-10.0, 10.0)
        self.observations.critic.actions.scale = 1.0
        self.observations.critic.actions.clip = (-1.0, 1.0)
        self.observations.critic.base_velocity_command.scale = 1.0
        self.observations.critic.base_velocity_command.clip = (-1.5, 1.5)
        self.observations.critic.ee_twist_command.scale = 1.0
        self.observations.critic.ee_twist_command.clip = (-5.0, 5.0)
        self.observations.critic.feet_swing_height_command.scale = 1.0
        self.observations.critic.feet_swing_height_command.clip = (0.0, 0.2)
        

        # ------------------------------Actions------------------------------
        # reduce action scale
        # self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.legs.scale = 0.05
        self.actions.arm.scale = 0.05
        

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # 父类LocomotionVelocityRoughEnvCfg里已经定义了一些reward term，默认权重为0，这里修改其权重和参数
        # loco
        self.rewards.base_linear_velocity.weight = 2.0
        self.rewards.base_angular_velocity.weight = 2.0
        self.rewards.torso_height.weight = 0.5
        self.rewards.base_roll_pitch_angles.weight = 0.1
        self.rewards.torso_linear_velocity.weight = 0.5
        self.rewards.torso_roll_pitch_velocities.weight = 2.5
        self.rewards.is_alive.weight = 0.05
        self.rewards.is_terminated.weight = -400.0
        self.rewards.undesired_robot_contacts.weight = -1.0
        self.rewards.robot_action_rate.weight = -0.001
        self.rewards.robot_joint_torque.weight = 1e-5
        self.rewards.robot_joint_velocity.weight = 1e-4
        # mani
        self.rewards.ee_position.weight = 5.0
        self.rewards.ee_orientation.weight = 4.0
        self.rewards.undesired_arm_contacts.weight = -1.0
        self.rewards.arm_action_rate.weight = 0.1
        self.rewards.arm_joint_torques.weight = 1e-5
        self.rewards.arm_joint_velocities.weight = 1e-4
        # contact
        self.rewards.feet_contact_rough.weight = 1.0
        self.rewards.feet_air_time_variance.weight = 1.0
        self.rewards.feet_air_time.weight = 0.25

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2ArmRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # Reward groups for multi-critic PPO
        self.reward_group_terms = {
            "loco": [
                "base_linear_velocity",
                "base_angular_velocity",
                "torso_height",
                "base_roll_pitch_angles",
                "torso_linear_velocity",
                "torso_roll_pitch_velocities",
                "is_alive",
                "is_terminated",
                "undesired_robot_contacts",
                "robot_action_rate",
                "robot_joint_torque",
                "robot_joint_velocity"
            ],
            "mani": [
                "ee_position",
                "ee_orientation",
                "undesired_arm_contacts",
                "arm_action_rate",
                "arm_joint_torques",
                "arm_joint_velocities"
            ],
            "contact": [
                "feet_contact_rough",
                "feet_air_time_variance",
                "feet_air_time"
            ],
        }

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        # self.terminations.illegal_contact = None

        # 只允许足端接地：非足端任意刚体与地面接触即终止
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        # self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_lin_vel = None   # 不使用速度curriculum
        self.curriculum.command_levels_ang_vel = None   # 不使用速度curriculum

        # ------------------------------Commands------------------------------
        # 具体参数写在 cus_velocity_env_cfg.py 里
