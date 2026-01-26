# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def velocity_mismatch_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    velocity_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize not moving when velocity command is given.
    
    当有速度命令（|cmd| > command_threshold）但机器人实际速度很小（|vel| < velocity_threshold）时，
    给予惩罚。这可以防止机器人学会"站着不动"的局部最优策略。
    
    返回值：
        惩罚值（正数），当命令大但速度小时返回1.0，否则返回0.0
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 获取速度命令和实际速度
    vel_cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(vel_cmd[:, :2], dim=1)  # 线速度命令大小
    
    actual_vel = asset.data.root_lin_vel_b[:, :2]
    vel_norm = torch.norm(actual_vel, dim=1)  # 实际线速度大小
    
    # 有命令但不动时惩罚
    has_command = cmd_norm > command_threshold
    not_moving = vel_norm < velocity_threshold
    
    penalty = (has_command & not_moving).float()
    return penalty


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_limit_margin_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin_ratio: float = 0.10,
    min_margin: float = 0.05,
    max_penalty_per_joint: float = 1.0,
    oob_scale: float = 1.0,
    max_oob_penalty_per_joint: float = 10.0,
    use_soft_limits: bool = False,
) -> torch.Tensor:
    """Continuous penalty when joints approach position limits.

    This is a "soft-constraint" barrier: penalty is zero when a joint is comfortably within limits,
    and increases smoothly to :attr:`max_penalty_per_joint` as the joint approaches the limit.

    Unlike :func:`isaaclab.envs.mdp.joint_pos_limits`, which only penalizes when crossing soft limits,
    this term penalizes *proximity* to the limits, which helps prevent policies from constantly pushing
    joints into hard stops when using relative (delta) joint position actions.

    Args:
        margin_ratio: Fraction of the joint range considered as the "near-limit" zone.
        min_margin: Minimum near-limit zone size [rad] (avoid tiny ranges causing numerical issues).
        max_penalty_per_joint: Upper bound per joint (keeps penalty scale well-behaved).
        use_soft_limits: If True, compute proximity w.r.t. soft limits; else w.r.t. hard limits.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    if use_soft_limits:
        limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    else:
        limits = asset.data.joint_pos_limits[:, asset_cfg.joint_ids]

    lo = limits[..., 0]
    hi = limits[..., 1]
    rng = (hi - lo).clamp(min=1.0e-6)

    # Distance to nearest limit (>= 0 inside limits).
    dist = torch.minimum(q - lo, hi - q)

    # Near-limit zone size (per joint).
    margin = torch.clamp(float(margin_ratio) * rng, min=float(min_margin))

    # Near-limit barrier: 0 when dist >= margin, smoothly increases to 1 as dist -> 0.
    s = ((margin - dist) / margin).clamp(min=0.0, max=1.0)
    per_joint = (s * s) * float(max_penalty_per_joint)

    # Out-of-bounds (OOB) penalty: increases with the magnitude of the violation (no early saturation).
    # If dist < 0, the joint is outside limits by (-dist) radians.
    if float(oob_scale) > 0.0:
        oob = (-dist).clamp(min=0.0)
        oob_norm = oob / margin
        oob_pen = (oob_norm * oob_norm) * float(oob_scale)
        if float(max_oob_penalty_per_joint) > 0.0:
            oob_pen = oob_pen.clamp(max=float(max_oob_penalty_per_joint))
        per_joint = per_joint + oob_pen
    return torch.sum(per_joint, dim=1)


class EndEffectorPositionReward(ManagerTermBase):
    """End-effector position tracking reward using desired twist (paper Eq. for r_EE^t).

    The target position is r_EE^{t-1} + v_hat_EE * dt, where v_hat_EE comes from the command.
    Positions/velocities are computed in the same frame as the command representation:
    - command_frame="control": gravity-aligned yaw-only torso frame (paper "control frame")
    - command_frame="base": torso/body frame (full orientation)
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._dt = float(env.cfg.decimation * env.cfg.sim.dt)
        self._ee_body_cfg: SceneEntityCfg = cfg.params["ee_body_cfg"]
        self._torso_body_cfg: SceneEntityCfg = cfg.params["torso_body_cfg"]
        if self._ee_body_cfg.name != self._torso_body_cfg.name:
            raise ValueError("ee_body_cfg and torso_body_cfg must reference the same asset.")
        if not self._ee_body_cfg.body_ids or len(self._ee_body_cfg.body_ids) != 1:
            raise ValueError("ee_body_cfg must specify exactly one body.")
        if not self._torso_body_cfg.body_ids or len(self._torso_body_cfg.body_ids) != 1:
            raise ValueError("torso_body_cfg must specify exactly one body.")
        self._asset: Articulation = env.scene[self._ee_body_cfg.name]
        self._ee_body_id = self._ee_body_cfg.body_ids[0]
        self._torso_body_id = self._torso_body_cfg.body_ids[0]
        self._prev_ee_pos_t = torch.zeros(self.num_envs, 3, device=self.device)
        self._command_frame: str | None = None
        self._command_name: str = str(cfg.params.get("command_name", ""))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        command_frame = self._command_frame or "control"
        if self._command_name:
            try:
                term = self._env.command_manager.get_term(self._command_name)
                command_frame = str(getattr(term, "command_frame", command_frame))
            except Exception:
                pass
        ee_pos_t = self._get_ee_pos_t(command_frame)
        if env_ids is None:
            self._prev_ee_pos_t = ee_pos_t
            return
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        self._prev_ee_pos_t[env_ids] = ee_pos_t[env_ids]

    def _get_ee_pos_t(self, command_frame: str) -> torch.Tensor:
        torso_pos_w = self._asset.data.body_pos_w[:, self._torso_body_id]
        torso_quat_w = self._asset.data.body_quat_w[:, self._torso_body_id]
        if command_frame == "base":
            task_quat_w = torso_quat_w
        else:
            task_quat_w = yaw_quat(torso_quat_w)
        ee_pos_w = self._asset.data.body_pos_w[:, self._ee_body_id]
        ee_quat_w = self._asset.data.body_quat_w[:, self._ee_body_id]
        ee_pos_t, _ = math_utils.subtract_frame_transforms(torso_pos_w, task_quat_w, ee_pos_w, ee_quat_w)
        return ee_pos_t

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        ee_body_cfg: SceneEntityCfg,
        torso_body_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        command_term = env.command_manager.get_term(command_name)
        command_frame = str(getattr(command_term, "command_frame", "control"))
        ee_pos_t = self._get_ee_pos_t(command_frame)
        if self._command_frame != command_frame:
            self._command_frame = command_frame
            self._prev_ee_pos_t = ee_pos_t
        v_hat = env.command_manager.get_command(command_name)[:, 0:3]
        pos_pred = self._prev_ee_pos_t + v_hat * self._dt
        pos_error = torch.sum(torch.square(ee_pos_t - pos_pred), dim=1)
        reward = torch.exp(-pos_error / std**2)
        self._prev_ee_pos_t = ee_pos_t
        return reward


class EndEffectorOrientationReward(ManagerTermBase):
    """End-effector orientation tracking reward using desired twist (paper Eq. for R_EE^t)."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._dt = float(env.cfg.decimation * env.cfg.sim.dt)
        self._ee_body_cfg: SceneEntityCfg = cfg.params["ee_body_cfg"]
        self._torso_body_cfg: SceneEntityCfg = cfg.params["torso_body_cfg"]
        if self._ee_body_cfg.name != self._torso_body_cfg.name:
            raise ValueError("ee_body_cfg and torso_body_cfg must reference the same asset.")
        if not self._ee_body_cfg.body_ids or len(self._ee_body_cfg.body_ids) != 1:
            raise ValueError("ee_body_cfg must specify exactly one body.")
        if not self._torso_body_cfg.body_ids or len(self._torso_body_cfg.body_ids) != 1:
            raise ValueError("torso_body_cfg must specify exactly one body.")
        self._asset: Articulation = env.scene[self._ee_body_cfg.name]
        self._ee_body_id = self._ee_body_cfg.body_ids[0]
        self._torso_body_id = self._torso_body_cfg.body_ids[0]
        self._prev_ee_quat_t = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_ee_quat_t[:, 0] = 1.0
        self._command_frame: str | None = None
        self._command_name: str = str(cfg.params.get("command_name", ""))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        command_frame = self._command_frame or "control"
        if self._command_name:
            try:
                term = self._env.command_manager.get_term(self._command_name)
                command_frame = str(getattr(term, "command_frame", command_frame))
            except Exception:
                pass
        ee_quat_t = self._get_ee_quat_t(command_frame)
        if env_ids is None:
            self._prev_ee_quat_t = ee_quat_t
            return
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        self._prev_ee_quat_t[env_ids] = ee_quat_t[env_ids]

    def _get_ee_quat_t(self, command_frame: str) -> torch.Tensor:
        torso_pos_w = self._asset.data.body_pos_w[:, self._torso_body_id]
        torso_quat_w = self._asset.data.body_quat_w[:, self._torso_body_id]
        if command_frame == "base":
            task_quat_w = torso_quat_w
        else:
            task_quat_w = yaw_quat(torso_quat_w)
        ee_pos_w = self._asset.data.body_pos_w[:, self._ee_body_id]
        ee_quat_w = self._asset.data.body_quat_w[:, self._ee_body_id]
        _, ee_quat_t = math_utils.subtract_frame_transforms(torso_pos_w, task_quat_w, ee_pos_w, ee_quat_w)
        return ee_quat_t

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        ee_body_cfg: SceneEntityCfg,
        torso_body_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        command_term = env.command_manager.get_term(command_name)
        command_frame = str(getattr(command_term, "command_frame", "control"))
        ee_quat_t = self._get_ee_quat_t(command_frame)
        if self._command_frame != command_frame:
            self._command_frame = command_frame
            self._prev_ee_quat_t = ee_quat_t
        w_hat = env.command_manager.get_command(command_name)[:, 3:6]
        quat_pred = math_utils.quat_box_plus(self._prev_ee_quat_t, w_hat * self._dt)
        rotvec_err = math_utils.quat_box_minus(ee_quat_t, quat_pred)
        rot_error = torch.sum(torch.square(rotvec_err), dim=1)
        reward = torch.exp(-rot_error / std**2)
        self._prev_ee_quat_t = ee_quat_t
        return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_rate_exp(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Reward small action rate using a Gaussian kernel."""
    action_error = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return torch.exp(-action_error / std**2)


def action_rate_exp_by_joint(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward small action rate for a subset of joints using a Gaussian kernel."""
    if not asset_cfg.joint_ids:
        raise ValueError("action_rate_exp_by_joint requires asset_cfg.joint_ids to be resolved.")
    action = env.action_manager.action[:, asset_cfg.joint_ids]
    prev_action = env.action_manager.prev_action[:, asset_cfg.joint_ids]
    action_error = torch.sum(torch.square(action - prev_action), dim=1)
    return torch.exp(-action_error / std**2)


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_paper(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str | None = None,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Feet air time reward (paper Table 6)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum(last_air_time * first_contact, dim=1)
    if command_name is not None:
        reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > command_threshold
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


class FeetAirTimeVarianceReward(ManagerTermBase):
    """Variance of air/contact time over the recent window (paper Table 6)."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._history_len = int(cfg.params.get("history_len", 3))
        if self._history_len < 1:
            raise ValueError("history_len must be >= 1.")
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        if not sensor_cfg.body_ids:
            raise ValueError("sensor_cfg must specify body_names for feet.")
        self._sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        self._body_ids = sensor_cfg.body_ids
        num_feet = len(self._body_ids)
        self._air_hist = torch.zeros(self._history_len, self.num_envs, num_feet, device=self.device)
        self._contact_hist = torch.zeros_like(self._air_hist)
        self._air_hist_idx = torch.zeros(self.num_envs, num_feet, dtype=torch.long, device=self.device)
        self._contact_hist_idx = torch.zeros_like(self._air_hist_idx)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        air = self._sensor.data.last_air_time[:, self._body_ids]
        contact = self._sensor.data.last_contact_time[:, self._body_ids]
        if env_ids is None:
            self._air_hist[:] = air
            self._contact_hist[:] = contact
            self._air_hist_idx.zero_()
            self._contact_hist_idx.zero_()
            return
        self._air_hist[:, env_ids] = air[env_ids]
        self._contact_hist[:, env_ids] = contact[env_ids]
        self._air_hist_idx[env_ids] = 0
        self._contact_hist_idx[env_ids] = 0

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, history_len: int = 3) -> torch.Tensor:
        if int(history_len) != self._history_len:
            raise ValueError("history_len must match the value used at initialization.")

        first_contact = self._sensor.compute_first_contact(env.step_dt)[:, self._body_ids]
        last_air = self._sensor.data.last_air_time[:, self._body_ids]
        first_air = self._sensor.compute_first_air(env.step_dt)[:, self._body_ids]
        last_contact = self._sensor.data.last_contact_time[:, self._body_ids]

        num_feet = len(self._body_ids)
        for foot_id in range(num_feet):
            env_ids = torch.where(first_contact[:, foot_id])[0]
            if env_ids.numel() > 0:
                idx = self._air_hist_idx[env_ids, foot_id]
                self._air_hist[idx, env_ids, foot_id] = last_air[env_ids, foot_id]
                self._air_hist_idx[env_ids, foot_id] = (idx + 1) % self._history_len

            env_ids = torch.where(first_air[:, foot_id])[0]
            if env_ids.numel() > 0:
                idx = self._contact_hist_idx[env_ids, foot_id]
                self._contact_hist[idx, env_ids, foot_id] = last_contact[env_ids, foot_id]
                self._contact_hist_idx[env_ids, foot_id] = (idx + 1) % self._history_len

        air_var = torch.var(self._air_hist, dim=0, unbiased=False)
        contact_var = torch.var(self._contact_hist, dim=0, unbiased=False)
        reward = torch.sum(air_var + contact_var, dim=1)
        # 避免“趴地后惩罚变小”
        # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_paper(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    torso_body_cfg: SceneEntityCfg,
    force_std: float,
    height_std: float,
    vel_std: float,
    contact_force_threshold: float = 1.0,
    height_contact_epsilon: float = 1.0e-6,
    ground_sensor_names: Sequence[str] | None = None,
    base_velocity_command_name: str | None = None,
    velocity_threshold: float = 0.1,
) -> torch.Tensor:
    """Foot contact schedule reward (paper Table 6).
    
    修改说明：当速度命令小于阈值时，期望所有脚都着地（静止站立），
    避免在没有速度命令时仍然要求抬脚的问题。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    # desired swing heights (command order must match asset_cfg.body_names)
    h_hat = env.command_manager.get_command(command_name)
    if h_hat.shape[-1] != len(asset_cfg.body_ids):
        raise ValueError("feet_contact_paper expects command to match the number of feet.")

    # 检查速度命令：如果速度很小，期望全部站立（不要求抬脚）
    if base_velocity_command_name is not None:
        vel_cmd = env.command_manager.get_command(base_velocity_command_name)
        vel_cmd_norm = torch.norm(vel_cmd[:, :3], dim=1, keepdim=True)  # 只看线速度+角速度
        # 当速度命令小于阈值时，将 h_hat 置零（期望全部站立）
        h_hat = torch.where(vel_cmd_norm < velocity_threshold, torch.zeros_like(h_hat), h_hat)

    # desired contact state: 1 = stance, 0 = swing
    c_des = (h_hat <= height_contact_epsilon).float()

    # contact forces (world frame)
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_mag = torch.linalg.norm(net_forces, dim=-1)
    force_z = torch.abs(net_forces[..., 2])

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    if ground_sensor_names is not None:
        if len(ground_sensor_names) != len(asset_cfg.body_ids):
            raise ValueError("ground_sensor_names must match the number of feet.")
        ground_z = torch.zeros(env.num_envs, len(asset_cfg.body_ids), device=env.device)
        for i, sensor_name in enumerate(ground_sensor_names):
            sensor: RayCaster = env.scene.sensors[sensor_name]
            ray_hits = sensor.data.ray_hits_w[..., 2]
            if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
                ground_z[:, i] = foot_pos_w[:, i, 2]
            else:
                ground_z[:, i] = torch.mean(ray_hits, dim=1)
        h_z = foot_pos_w[..., 2] - ground_z
    else:
        # foot height in world z (flat terrain assumption)
        h_z = foot_pos_w[..., 2]

    # swing: encourage no contact and match desired height
    height_err = h_hat - h_z
    swing_term = (1.0 - c_des) * torch.exp(-(force_mag**2) / (force_std**2)) * torch.exp(
        -(height_err**2) / (height_std**2)
    )

    # stance: if in contact, penalize lateral foot velocity
    in_contact = (force_z > contact_force_threshold).float()
    torso_quat_w = asset.data.body_quat_w[:, torso_body_cfg.body_ids[0]]
    task_quat_w = yaw_quat(torso_quat_w)
    foot_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    task_quat_w = task_quat_w.unsqueeze(1).expand(-1, foot_vel_w.shape[1], -1)
    foot_vel_t = math_utils.quat_apply_inverse(task_quat_w, foot_vel_w)
    vel_xy = torch.linalg.norm(foot_vel_t[..., :2], dim=-1)
    stance_term = c_des * in_contact * torch.exp(-(vel_xy**2) / (vel_std**2))

    reward = torch.sum(swing_term + stance_term, dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

# 直接读取的世界坐标系下的足端的高度，不考虑地形因此可能不够准确
def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

# 计算的足端相对于机体坐标系下的高度，更加准确，且不需要考虑地形高度
def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)

# 它计算的是重力向量与 Z 轴的偏差
def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward tracking of base height using a Gaussian kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        adjusted_target_height = target_height
    height_error = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward = torch.exp(-height_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

# z方向速度惩罚
def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def lin_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of z-axis base linear velocity using a Gaussian kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_error = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward = torch.exp(-vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_torques_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint torques using a Gaussian kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    torque_error = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    return torch.exp(-torque_error / std**2)

def joint_vel_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities using a Gaussian kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_error = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return torch.exp(-vel_error / std**2)

def ang_vel_xy_exp(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    err = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward = torch.exp(-err / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    # 删除，避免“趴地不再受罚”
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward flat base orientation using a Gaussian kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    orientation_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward = torch.exp(-orientation_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward
