# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from collections.abc import Callable
from dataclasses import MISSING
from typing import cast

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.mdp import observations as mdp_obs
# 添加实验室的包
# import isaaclab_nhb.tasks.mdp_nhb as mdp_nhb
##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # 添加四个足端高度扫描传感器
    FL_foot_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
            offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 2.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.2, 0.2)),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
    FR_foot_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/FR_foot",
            offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 2.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.2, 0.2)),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
    RL_foot_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/RL_foot",
            offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 2.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.2, 0.2)),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
    RR_foot_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/RR_foot",
            offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 2.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.2, 0.2)),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
    
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # 基座速度命令
    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        # resampling_time_range=(10.0, 10.0),
        resampling_time_range=(6.0, 8.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            # lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
            lin_vel_x=(-1.5, 1.5), lin_vel_y=(-1.5, 1.5), ang_vel_z=(-1.5, 1.5)     # 不使用 heading 命令
        ),
    )
    # 末端执行器（EE）轨迹 + twist 命令
    # 注意：ee_body_name / torso_body_name / shoulder_body_name 需与你的机器人 body_names 一致
    ee_twist = mdp.EndEffectorTwistTrajectoryCommandCfg(
        asset_name="robot",
        # resampling_time_range=(10.0, 10.0),   # 轨迹采用持续时间范围
        resampling_time_range=(4.0, 6.0),   # 轨迹采用持续时间范围
        ee_body_name="link6",
        torso_body_name="base",
        shoulder_body_name="link2",
        reject_cuboid=(-0.25, 0.25, -0.16, 0.16, -0.08, 0.20),  # torso和hip的最大外包络长方体（估计值）
        trajectory_duration_range=(3.0, 5.0),   # 轨迹时间范围
        local_trajectory_probability=0.5,
        # Paper curriculum (Sec. 7.2.1): start with base-frame commands, then switch to control-frame.
        command_frame="base",
    )
    # 足端摆动高度命令（论文 Eq. (5)）
    feet_swing_height = mdp.DesiredFeetSwingHeightCommandCfg(
        # resampling_time_range=(6.0, 6.0),
        resampling_time_range=(6.0, 8.0),   # 与base velocity对应
        # max_height=0.12,
        max_height=0.08,
        gait_frequency=2.0,
        phase_offsets=(0.0, 0.5, 0.5, 0.0),  # [FL, FR, RL, RR]写死顺序了
        clip_to_positive=True,
    )

    # 四足步态命令（用于步态奖励与观测）
    # 注意：QuadrupedGaitCommand 的参考足为 LF（左前），Go2Z1 对应为 "FL_foot"。
    # 这里默认固定为 trot：LF+RB 同相，RF+LB 同相（相位差 0.5）。
    # gait_command = mdp_nhb.QuadrupedGaitCommandCfg(
    #     resampling_time_range=(2.0, 4.0),
    #     ranges=mdp_nhb.QuadrupedGaitCommandCfg.Ranges(
    #         stance_rate=(0.70, 0.70),
    #         rf_offset=(0.50, 0.50),
    #         lb_offset=(0.50, 0.50),
    #         rb_offset=(0.00, 0.00),
    #         gait_frequency=(1.50, 1.50),
    #     ),
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # 配置为增量式关节位置控制
    # 论文公式(2): q_target = q_current + action * scale (增量模式)
    # RelativeJointPositionAction 实际为:
    #   q_target = q_current + (action * scale + offset)
    # use_zero_offset=True 会强制 offset=0（推荐，匹配论文增量模式并避免误配置 offset）
    legs = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint"
        ],
        # 经验：交叉腿常由髋外展/内收关节过大动作引起，单独降低 hip 的 scale 更有效
        scale={".*_hip_joint": 0.08, "^(?!.*_hip_joint).*": 0.25},  # 降低 hip 关节的 scale后，机器人不趴地
        use_zero_offset=True,
        preserve_order=True,
    )
    arm = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6"
        ],
        scale=0.10,
        use_zero_offset=True,
        preserve_order=True,
    )



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # History_term: 4帧关节位置, 72 Dim
        # 关节位置历史（过去4帧），维度 = 关节数 × 4
        joint_pos_history = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "FL_hip_joint",
                        "FL_thigh_joint",
                        "FL_calf_joint",
                        "FR_hip_joint",
                        "FR_thigh_joint",
                        "FR_calf_joint",
                        "RL_hip_joint",
                        "RL_thigh_joint",
                        "RL_calf_joint",
                        "RR_hip_joint",
                        "RR_thigh_joint",
                        "RR_calf_joint",
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                    ],
                    preserve_order=True,
                )
            },
            # noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
            history_length=4,          # 存储过去4帧
            flatten_history_dim=True,  # 将历史维度展平为2D (num_envs, joint_num * 4)
        )
        # Proprioception_term: 重力投影, 3 Dim
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )        
        # Proprioception_term: 基座线速度, 3 Dim
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Proprioception_term: 基座角速度, 3 Dim
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Proprioception_term: 关节位置, 18 Dim
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "FL_hip_joint",
                        "FL_thigh_joint",
                        "FL_calf_joint",
                        "FR_hip_joint",
                        "FR_thigh_joint",
                        "FR_calf_joint",
                        "RL_hip_joint",
                        "RL_thigh_joint",
                        "RL_calf_joint",
                        "RR_hip_joint",
                        "RR_thigh_joint",
                        "RR_calf_joint",
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                    ],
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Proprioception_term: 关节速度, 18 Dim
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "FL_hip_joint",
                        "FL_thigh_joint",
                        "FL_calf_joint",
                        "FR_hip_joint",
                        "FR_thigh_joint",
                        "FR_calf_joint",
                        "RL_hip_joint",
                        "RL_thigh_joint",
                        "RL_calf_joint",
                        "RR_hip_joint",
                        "RR_thigh_joint",
                        "RR_calf_joint",
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                    ],
                    preserve_order=True,
                )
            },
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Previous_action_term: 上一步动作, 18 Dim
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Command term: Base velocity command, 3 Dim
        # 基座速度命令（相对于机器人坐标系 body frame）
        # [0]: lin_vel_x 前向线速度 (m/s)
        # [1]: lin_vel_y 侧向线速度 (m/s)
        # [2]: ang_vel_z 偏航角速度 (rad/s)
        base_velocity_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-10.0, 10.0),
            scale=1.0,
        )
        # Command term: End-effector twist + goal pose command, 13 Dim
        ee_twist_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_twist"},
            clip=(-10.0, 10.0),
            scale=1.0,
        )
        # Command term: Feet swing height command, 4 Dim
        feet_swing_height_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "feet_swing_height"},
            clip=(0.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True   # policy 加噪声，有一些观测项不加噪声如 velocity_commands、gait_commands等
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        # History_term: 4帧关节位置, 72 Dim
        # 关节位置历史（过去4帧），维度 = 关节数 × 4
        joint_pos_history = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "FL_hip_joint",
                        "FL_thigh_joint",
                        "FL_calf_joint",
                        "FR_hip_joint",
                        "FR_thigh_joint",
                        "FR_calf_joint",
                        "RL_hip_joint",
                        "RL_thigh_joint",
                        "RL_calf_joint",
                        "RR_hip_joint",
                        "RR_thigh_joint",
                        "RR_calf_joint",
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                    ],
                    preserve_order=True,
                )
            },
            # noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
            history_length=4,          # 存储过去4帧
            flatten_history_dim=True,  # 将历史维度展平为2D (num_envs, joint_num * 4)
        )
        # Proprioception_term: 重力投影, 3 Dim
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )        
        # Proprioception_term: 基座线速度, 3 Dim
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Proprioception_term: 基座角速度, 3 Dim
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Proprioception_term: 关节位置, 18 Dim
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "FL_hip_joint",
                        "FL_thigh_joint",
                        "FL_calf_joint",
                        "FR_hip_joint",
                        "FR_thigh_joint",
                        "FR_calf_joint",
                        "RL_hip_joint",
                        "RL_thigh_joint",
                        "RL_calf_joint",
                        "RR_hip_joint",
                        "RR_thigh_joint",
                        "RR_calf_joint",
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                    ],
                    preserve_order=True,
                )
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Proprioception_term: 关节速度, 18 Dim
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "FL_hip_joint",
                        "FL_thigh_joint",
                        "FL_calf_joint",
                        "FR_hip_joint",
                        "FR_thigh_joint",
                        "FR_calf_joint",
                        "RL_hip_joint",
                        "RL_thigh_joint",
                        "RL_calf_joint",
                        "RR_hip_joint",
                        "RR_thigh_joint",
                        "RR_calf_joint",
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                    ],
                    preserve_order=True,
                )
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # Privileged info term: Feet contact state, 4 Dim
        # Feet contact state: 足端接触状态（二值），4 Dim（四足机器人）
        # 需要在子类中配置 body_names，如 ".*_foot"
        feet_contact_state = ObsTerm(
            func=mdp.feet_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
                "threshold": 1.0,
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )
        # Privileged info term: Static friction, 4 Dim ————————————————————————————————————————————————————————————————保持质疑
        # Static friction: 静摩擦系数（域随机化量），4 Dim（四足机器人）
        # 读取 randomize_rigid_body_material 随机化后的摩擦系数
        # 需要在子类中配置 body_names，如 ".*_foot"
        static_friction = ObsTerm(
            func=mdp.static_friction,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"])},
            clip=(0.0, 2.0),
            scale=1.0,
        )
        # Privileged info term: Feet air time, 4 Dim
        # Feet air time: 足端滞空时间，4 Dim（四足机器人）
        # 需要在子类中配置 body_names，如 ".*_foot"
        # 注意：需要 ContactSensorCfg.track_air_time=True
        feet_air_time = ObsTerm(
            func=mdp_obs.feet_air_time,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"])},
            clip=(0.0, 1.0),
            scale=1.0,
        )

        # Privileged info term: Base external wrench, 6 Dim
        # 基座外部力/力矩（域随机化量），由 randomize_apply_external_force_torque 设置
        # 需要在子类中配置 body_names 为基座名称
        base_external_wrench = ObsTerm(
            func=mdp.base_external_wrench,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # Privileged info term: Base external push velocity, 6 Dim
        # 基座外部推动速度（域随机化量），由 push_by_setting_velocity 设置
        # 返回当前根速度（包含推动扰动）
        # 需要在子类中配置 body_names 为基座名称（虽然函数内部读取的是 root_vel_w）
        base_external_push_velocity = ObsTerm(
            func=mdp.base_external_push_velocity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])},
            clip=(-10.0, 10.0),
            scale=1.0,
        )

        # Privileged info term: Base mass disturbance, 1 Dim
        # 基座质量扰动（域随机化量），由 randomize_rigid_body_mass 设置
        # 返回当前质量与默认质量的差值
        # 需要在子类中配置 body_names 为基座名称
        base_mass_disturbance = ObsTerm(
            func=mdp.base_mass_disturbance,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])},
            clip=(-10.0, 10.0),
            scale=1.0,
        )

        # Privileged info term: End-effector external wrench, 6 Dim
        # 末端执行器外部力/力矩（域随机化量），由 randomize_apply_external_force_torque 设置
        # 需要在子类中配置 body_names 为末端执行器名称
        ee_external_wrench = ObsTerm(
            func=mdp.ee_external_wrench,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["link6"])},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # Privileged info term: End-effector mass disturbance, 1 Dim
        # 末端执行器质量扰动（域随机化量），由 randomize_rigid_body_mass 设置
        # 返回当前质量与默认质量的差值
        # 需要在子类中配置 body_names 为末端执行器名称
        ee_mass_disturbance = ObsTerm(
            func=mdp.ee_mass_disturbance,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["link6"])},
            clip=(-10.0, 10.0),
            scale=1.0,
        )


        # Previous_action_term: 上一步动作, 18 Dim
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        #***********************************************************************************************************************
        # 处理command观测项
        # Command term: Base velocity command, 3 Dim
        # 基座速度命令（相对于机器人坐标系 body frame）
        # [0]: lin_vel_x 前向线速度 (m/s)
        # [1]: lin_vel_y 侧向线速度 (m/s)
        # [2]: ang_vel_z 偏航角速度 (rad/s)
        base_velocity_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-10.0, 10.0),
            scale=1.0,
        )
        # Command term: End-effector twist + goal pose command, 13 Dim
        ee_twist_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_twist"},
            clip=(-10.0, 10.0),
            scale=1.0,
        )
        # Command term: Feet swing height command, 4 Dim
        feet_swing_height_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "feet_swing_height"},
            clip=(0.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False  # critic 不加噪声
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    # 足端摩擦系数随机化
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            # "static_friction_range": (0.5, 1.2),
            # "dynamic_friction_range": (0.3, 1.2),
            # "restitution_range": (0.0, 0.2),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    # 基座质量随机化
    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            # "mass_distribution_params": (-2.0, 2.0),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    # EE质量随机化
    randomize_rigid_body_mass_ee = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["link6"]),
            # "mass_distribution_params": (0.0, 1.8),
            "mass_distribution_params": (0.0, 0.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    # 基座外力/力矩随机化
    randomize_apply_external_force_torque_base = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",   # 每个 Episode 开始时（环境重置时），随机采样一个力和力矩，并持续施加直到该 Episode 结束
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            # "force_range": (-50.0, 50.0),
            # "torque_range": (-20.0, 20.0),
            "force_range": (-20.0, 20.0),
            "torque_range": (-10.0, 10.0),
        },
    )
    # EE外力/力矩随机化
    randomize_apply_external_force_torque_ee = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["link6"]),
            # "force_range": (-3.0, 3.0),
            "force_range": (-0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )
    # 基座随机速度扰动
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "velocity_range": {
                "x": (-0.2, 0.2),   # m/s
                "y": (-0.1, 0.1),
                # "x": (-0.2, 0.2),   # m/s
                # "y": (-0.2, 0.2),
                # "z": (-0.2, 0.2),
                # "roll": (-0.2, 0.2),   # rad/s
                # "pitch": (-0.2, 0.2),
                # "yaw": (-0.2, 0.2),
            },
        },
    )
    # 质心位置随机化（只加了base，其他link也可加）
    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.02, 0.02)},
            # "com_range": {"x": (-0.00, 0.00), "y": (-0.00, 0.00), "z": (-0.00, 0.00)},
        },
    )
    # 关节初始状态随机化
    randomize_reset_joints = EventTerm(
        # func=mdp.reset_joints_by_scale,
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[
                    "FL_hip_joint",
                    "FL_thigh_joint",
                    "FL_calf_joint",
                    "FR_hip_joint",
                    "FR_thigh_joint",
                    "FR_calf_joint",
                    "RL_hip_joint",
                    "RL_thigh_joint",
                    "RL_calf_joint",
                    "RR_hip_joint",
                    "RR_thigh_joint",
                    "RR_calf_joint",
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "joint5",
                    "joint6",
                ],),
            "position_range": (-0.1, 0.1),
            # "position_range": (-0.0, 0.0),
            "velocity_range": ( 0.0, 0.0),
        },
    )
    # base初始状态随机化
    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.0, 0.0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


    # Skip: inertia updated via mass randomization by setting recompute_inertia=True
    # randomize_rigid_body_inertia = EventTerm(
    #     func=mdp.randomize_rigid_body_inertia,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "inertia_distribution_params": (0.5, 1.5),
    #         "operation": "scale",
    #     },
    # )

    # # robot执行器增益随机化
    # randomize_actuator_gains_robot = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", 
    #             joint_names=[
    #                 "FL_hip_joint",
    #                 "FL_thigh_joint",
    #                 "FL_calf_joint",
    #                 "FR_hip_joint",
    #                 "FR_thigh_joint",
    #                 "FR_calf_joint",
    #                 "RL_hip_joint",
    #                 "RL_thigh_joint",
    #                 "RL_calf_joint",
    #                 "RR_hip_joint",
    #                 "RR_thigh_joint",
    #                 "RR_calf_joint"
    #             ],),
    #         "stiffness_distribution_params": (0.5, 2.0),
    #         "damping_distribution_params": (0.5, 2.0),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )
    # # arm执行器增益随机化
    # randomize_actuator_gains_arm = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", 
    #             joint_names=[
    #                 "joint1",
    #                 "joint2",
    #                 "joint3",
    #                 "joint4",
    #                 "joint5",
    #                 "joint6",
    #             ],),
    #         "stiffness_distribution_params": (0.8, 1.2),
    #         "damping_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # loco
    base_linear_velocity = RewTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=0.0, 
        params={
            "command_name": "base_velocity", 
            "std": math.sqrt(0.1)
        }
    )
    base_angular_velocity = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=0.0, 
        params={
            "command_name": "base_velocity", 
            "std": math.sqrt(0.05)
        }
    )
    torso_height = RewTerm(
        func=mdp.base_height_exp, 
        weight=0.0, 
        params={
            "target_height": 0.4, 
            "std": math.sqrt(0.1), 
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "sensor_cfg": SceneEntityCfg("height_scanner_base")
        }
    )
    base_roll_pitch_angles = RewTerm(
        func=mdp.flat_orientation_exp, 
        weight=0.0, 
        params={
            "std": math.sqrt(0.1),
            "asset_cfg": SceneEntityCfg("robot", body_names="base")                
        }
    )
    torso_linear_velocity = RewTerm(
        func=mdp.lin_vel_z_exp,
        weight=0.0,
        params={
            "std": math.sqrt(0.2),
            "asset_cfg": SceneEntityCfg("robot", body_names="base")
        }
    )
    torso_roll_pitch_velocities = RewTerm(
        func=mdp.ang_vel_xy_exp,
        weight=0.0,
        params={
            "std": math.sqrt(0.2),
            "asset_cfg": SceneEntityCfg("robot", body_names="base")
        }
    )
    is_alive = RewTerm(
        func=mdp.is_alive, 
        weight=0.0
    )
    is_terminated = RewTerm(
        func=mdp.is_terminated, 
        weight=0.0
    )
    undesired_robot_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["^(?!.*_foot).*"]),
            "threshold": 1.0,
        },
    )
    robot_action_rate = RewTerm(
        func=mdp.action_rate_exp_by_joint,
        weight=0.0,
        params={
            "std": math.sqrt(0.1),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                        "FL_hip_joint",
                        "FL_thigh_joint",
                        "FL_calf_joint",
                        "FR_hip_joint",
                        "FR_thigh_joint",
                        "FR_calf_joint",
                        "RL_hip_joint",
                        "RL_thigh_joint",
                        "RL_calf_joint",
                        "RR_hip_joint",
                        "RR_thigh_joint",
                        "RR_calf_joint"]),
        },
    )
    robot_joint_torque = RewTerm(
        func=mdp.joint_torques_exp,
        weight=0.0,
        params={
            "std": math.sqrt(40.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                        "FL_hip_joint",
                        "FL_thigh_joint",
                        "FL_calf_joint",
                        "FR_hip_joint",
                        "FR_thigh_joint",
                        "FR_calf_joint",
                        "RL_hip_joint",
                        "RL_thigh_joint",
                        "RL_calf_joint",
                        "RR_hip_joint",
                        "RR_thigh_joint",
                        "RR_calf_joint"]),
        },
    )
    robot_joint_velocity = RewTerm(
        func=mdp.joint_vel_exp,
        weight=0.0,
        params={
            "std": math.sqrt(4.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                        "FL_hip_joint",
                        "FL_thigh_joint",
                        "FL_calf_joint",
                        "FR_hip_joint",
                        "FR_thigh_joint",
                        "FR_calf_joint",
                        "RL_hip_joint",
                        "RL_thigh_joint",
                        "RL_calf_joint",
                        "RR_hip_joint",
                        "RR_thigh_joint",
                        "RR_calf_joint"]),
        },
    )
    # # 不动惩罚：有速度命令但不动时惩罚
    # velocity_mismatch_penalty = RewTerm(
    #     func=mdp.velocity_mismatch_penalty,
    #     weight=0.0,  # 负权重，作为惩罚
    #     params={
    #         "command_name": "base_velocity",
    #         "command_threshold": 0.1,   # 速度命令阈值
    #         "velocity_threshold": 0.05, # 实际速度阈值
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    ##############################################################################################################################
    # mani
    ee_position = RewTerm(
        func=cast(Callable[..., torch.Tensor], mdp.EndEffectorPositionReward),
        weight=0.0,
        params={
            "std": math.sqrt(0.005),
            "command_name": "ee_twist",  # 你的 EndEffectorTwistTrajectoryCommand 名称
            "ee_body_cfg": SceneEntityCfg("robot", body_names="link6"),
            "torso_body_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )
    ee_orientation = RewTerm(
        func=cast(Callable[..., torch.Tensor], mdp.EndEffectorOrientationReward),
        weight=0.0,
        params={
            "std": math.sqrt(0.01),
            "command_name": "ee_twist",
            "ee_body_cfg": SceneEntityCfg("robot", body_names="link6"),
            "torso_body_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )
    undesired_arm_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["link1", "link2", "link3", "link4", "link5", "link6"]),
            "threshold": 1.0,
        },
    )
    arm_action_rate = RewTerm(
        func=mdp.action_rate_exp_by_joint,
        weight=0.0,
        params={
            "std": math.sqrt(0.5),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]),
        },
    )
    arm_joint_torques = RewTerm(
        func=mdp.joint_torques_exp,
        weight=0.0,
        params={
            "std": math.sqrt(40.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]),
        },
    )
    arm_joint_velocities = RewTerm(
        func=mdp.joint_vel_exp,
        weight=0.0,
        params={
            "std": math.sqrt(4.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]),
        },
    )

    ##############################################################################################################################
    # contact
    feet_contact_rough = RewTerm(
        func=mdp.feet_contact_paper,
        weight=0.0,  
        params={
            "command_name": "feet_swing_height",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            "torso_body_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_std": math.sqrt(1.0),
            "height_std": math.sqrt(0.05),
            "vel_std": math.sqrt(0.01),
            "contact_force_threshold": 1.0,
            # 如果只在平地训练，可以去掉 ground_sensor_names，直接用世界系 Z 高度
            "ground_sensor_names": ["FL_foot_scanner", "FR_foot_scanner", "RL_foot_scanner", "RR_foot_scanner"],
            # 新增：关联速度命令，速度小于阈值时期望静止站立
            "base_velocity_command_name": "base_velocity",
            "velocity_threshold": 0.02,
        },
    )
    feet_air_time_variance = RewTerm(
        func=cast(Callable[..., torch.Tensor], mdp.FeetAirTimeVarianceReward),
        weight=0.0, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot","FR_foot","RL_foot","RR_foot"]),
            "history_len": 3,
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_paper,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot","FR_foot","RL_foot","RR_foot"]),
            "command_name": "base_velocity",    # 有速度命令时只才奖励抬脚
            "command_threshold": 0.02,
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # MDP terminations
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # command_resample
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )

    # 倾倒终止
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=""), "limit_angle": 1.0},
    )

    # Contact sensor
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    # EE twist command frame curriculum (paper Sec. 7.2.1):
    # represent commands in base frame until a set number of iterations, then switch to control frame.
    ee_twist_command_frame = CurrTerm(
        func=mdp.ee_twist_command_frame_switch,
        params={
            "command_name": "ee_twist",
            # Paper uses ~3000 PPO iterations; with rsl_rl num_steps_per_env=24 => 3000*24 env steps.
            "switch_after_steps": 3000 * 24,
            "base_frame": "base",
            "control_frame": "control",
        },
    )

    # command_levels_lin_vel = CurrTerm(
    #     func=mdp.command_levels_lin_vel,
    #     params={
    #         "reward_term_name": "track_lin_vel_xy_exp",
    #         "range_multiplier": (0.1, 1.0),
    #     },
    # )

    # command_levels_ang_vel = CurrTerm(
    #     func=mdp.command_levels_ang_vel,
    #     params={
    #         "reward_term_name": "track_ang_vel_z_exp",
    #         "range_multiplier": (0.1, 1.0),
    #     },
    # )


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # Reward group mapping for multi-critic (optional)
    reward_group_terms: dict[str, list[str]] | None = None
    # If True, require all reward terms to be assigned to a group
    reward_group_strict: bool = False

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 8     # 4->8
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.0025    # 0.005->0.0025,变为400Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)
