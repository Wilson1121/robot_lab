# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

# from isaaclab_nhb.terrain.config.rough import ROUGH_TERRAINS_SIMPLE_CFG

from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

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
    foot_link_name = ".*_foot"
    # fmt: off
    joint_names = [
        # FR
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        # FL
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        # RR
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        # RL
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        # Arm
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    ]
    FOOT_LINK_NAMES: list[str] = [
    "FL_foot", "FR_foot", "RL_foot", "RR_foot",
]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_Go2Arm_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # 新添加的足端高度扫描器
        self.scene.FL_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.FOOT_LINK_NAMES[0]
        self.scene.FR_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.FOOT_LINK_NAMES[1]
        self.scene.RL_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.FOOT_LINK_NAMES[2]
        self.scene.RR_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.FOOT_LINK_NAMES[3]

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None    # Policy不观测线速度
        self.observations.policy.height_scan = None   # Policy不观测高度扫描
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

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
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0         # 惩罚：垂直速度过大
        self.rewards.ang_vel_xy_l2.weight = -0.05       # 惩罚：避免roll/pitch角速度过大
        self.rewards.flat_orientation_l2.weight = 0     # 惩罚：基座倾斜 --- 关闭 ---
        self.rewards.base_height_l2.weight = 0          # 惩罚：基座高度偏离目标高度 --- 关闭 ---
        self.rewards.base_height_l2.params["target_height"] = 0.33
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0         # 惩罚：基座线加速度过大 --- 关闭 ---
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2.5e-5  # 惩罚：关节力矩过大
        self.rewards.joint_vel_l2.weight = 0            # 惩罚：关节速度过大 --- 关闭 ---
        self.rewards.joint_acc_l2.weight = -2.5e-7      # 惩罚：关节加速度过大（平滑）
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])  # 惩罚：髋关节位置偏离初始位置
        self.rewards.joint_pos_limits.weight = -5.0     # 惩罚：关节位置接近软限位
        self.rewards.joint_vel_limits.weight = 0        # 惩罚：关节速度接近极限 --- 关闭 ---
        self.rewards.joint_power.weight = -2e-5         # 惩罚：关节功率过大（力矩*速度）
        self.rewards.stand_still.weight = -2.0          # 惩罚：与初始站立姿势偏离过大
        self.rewards.joint_pos_penalty.weight = -1.0    # 惩罚：关节位置偏离初始位置
        self.rewards.joint_mirror.weight = -0.05        # 惩罚：左右关节动作不对称
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01      # 惩罚：动作变化过大（平滑）

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0   # 惩罚：非足端刚体与地面接触（没惩罚身体间碰撞）
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4    # 惩罚：足端接触力过大
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0  # 奖励：线速度xy跟踪
        self.rewards.track_ang_vel_z_exp.weight = 1.5   # 奖励：角速度z跟踪

        # Others
        self.rewards.feet_air_time.weight = 0.1             # 奖励：足端离地时间（鼓励跳跃）
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -1.0   # 惩罚：足端离地时间差异过大（鼓励均匀步态）
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0                # 奖励：足端接地 --- 关闭 ---
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1  # 奖励：无命令时足端接地
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0                # 惩罚：足端绊倒（踩到陡坡或踢到墙壁） --- 关闭 ---
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.1               # 惩罚：足端滑动    
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0                 # 奖励：足端高度接近目标觉得高度 --- 关闭 ---
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -5.0         # 惩罚：足端相对于基座高度过低（不抬腿）
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0.5                 # 奖励：足端步态同步
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.upward.weight = 1.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2ArmRoughEnvCfg":
            self.disable_zero_weight_rewards()

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
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
