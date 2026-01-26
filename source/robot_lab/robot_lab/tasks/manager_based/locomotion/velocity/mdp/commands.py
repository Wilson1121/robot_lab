# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_box_minus,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
    yaw_quat,
)

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _safe_normalize(vec: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return vec / torch.clamp(torch.norm(vec, dim=-1, keepdim=True), min=eps)


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation for quaternions in (w, x, y, z)."""
    # Ensure shapes (..., 4) and t (..., 1) or (...,)
    if t.ndim == q0.ndim - 1:
        t = t.unsqueeze(-1)

    # Flip to take shortest path
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1_adj = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot)

    # If very close, fall back to lerp
    DOT_THRESHOLD = 0.9995
    lerp = _safe_normalize(q0 + t * (q1_adj - q0))

    theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))  # angle between
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * t
    sin_theta = torch.sin(theta)

    s0 = torch.where(sin_theta_0 > 1.0e-6, torch.cos(theta) - dot * sin_theta / sin_theta_0, 1.0 - t)
    s1 = torch.where(sin_theta_0 > 1.0e-6, sin_theta / sin_theta_0, t)
    slerp = s0 * q0 + s1 * q1_adj
    slerp = _safe_normalize(slerp)

    return torch.where(dot > DOT_THRESHOLD, lerp, slerp)


def _quat_to_rotvec(q: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to rotation vector (axis-angle)."""
    q = _safe_normalize(q, eps=eps)
    w = torch.clamp(q[..., 0], -1.0, 1.0)
    v = q[..., 1:4]
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(v_norm, w.unsqueeze(-1))
    axis = v / torch.clamp(v_norm, min=eps)
    return axis * angle


def _quat_from_orthonormal_axes(x_axis: torch.Tensor, y_axis: torch.Tensor, z_axis: torch.Tensor) -> torch.Tensor:
    """Create quaternion (w, x, y, z) from orthonormal axes as columns of rotation matrix."""
    # Rotation matrix with columns [x y z]
    r00 = x_axis[..., 0]
    r10 = x_axis[..., 1]
    r20 = x_axis[..., 2]
    r01 = y_axis[..., 0]
    r11 = y_axis[..., 1]
    r21 = y_axis[..., 2]
    r02 = z_axis[..., 0]
    r12 = z_axis[..., 1]
    r22 = z_axis[..., 2]

    trace = r00 + r11 + r22
    q = torch.zeros((*trace.shape, 4), dtype=x_axis.dtype, device=x_axis.device)

    # Branchless-ish conversion
    t_pos = trace > 0.0
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1.0e-8)) * 2.0
    q_w = 0.25 * s
    q_x = (r21 - r12) / torch.clamp(s, min=1.0e-8)
    q_y = (r02 - r20) / torch.clamp(s, min=1.0e-8)
    q_z = (r10 - r01) / torch.clamp(s, min=1.0e-8)
    q[t_pos, 0] = q_w[t_pos]
    q[t_pos, 1] = q_x[t_pos]
    q[t_pos, 2] = q_y[t_pos]
    q[t_pos, 3] = q_z[t_pos]

    # For negative trace cases, pick the dominant diagonal
    t_neg = ~t_pos
    if torch.any(t_neg):
        r00n = r00[t_neg]
        r11n = r11[t_neg]
        r22n = r22[t_neg]
        r01n = r01[t_neg]
        r02n = r02[t_neg]
        r10n = r10[t_neg]
        r12n = r12[t_neg]
        r20n = r20[t_neg]
        r21n = r21[t_neg]

        cond_x = (r00n > r11n) & (r00n > r22n)
        cond_y = (~cond_x) & (r11n > r22n)
        cond_z = (~cond_x) & (~cond_y)

        # x-dominant
        sx = torch.sqrt(torch.clamp(1.0 + r00n - r11n - r22n, min=1.0e-8)) * 2.0
        qx_w = (r21n - r12n) / torch.clamp(sx, min=1.0e-8)
        qx_x = 0.25 * sx
        qx_y = (r01n + r10n) / torch.clamp(sx, min=1.0e-8)
        qx_z = (r02n + r20n) / torch.clamp(sx, min=1.0e-8)

        # y-dominant
        sy = torch.sqrt(torch.clamp(1.0 + r11n - r00n - r22n, min=1.0e-8)) * 2.0
        qy_w = (r02n - r20n) / torch.clamp(sy, min=1.0e-8)
        qy_x = (r01n + r10n) / torch.clamp(sy, min=1.0e-8)
        qy_y = 0.25 * sy
        qy_z = (r12n + r21n) / torch.clamp(sy, min=1.0e-8)

        # z-dominant
        sz = torch.sqrt(torch.clamp(1.0 + r22n - r00n - r11n, min=1.0e-8)) * 2.0
        qz_w = (r10n - r01n) / torch.clamp(sz, min=1.0e-8)
        qz_x = (r02n + r20n) / torch.clamp(sz, min=1.0e-8)
        qz_y = (r12n + r21n) / torch.clamp(sz, min=1.0e-8)
        qz_z = 0.25 * sz

        qn = torch.zeros((r00n.shape[0], 4), dtype=q.dtype, device=q.device)
        qn[cond_x, 0] = qx_w[cond_x]
        qn[cond_x, 1] = qx_x[cond_x]
        qn[cond_x, 2] = qx_y[cond_x]
        qn[cond_x, 3] = qx_z[cond_x]

        qn[cond_y, 0] = qy_w[cond_y]
        qn[cond_y, 1] = qy_x[cond_y]
        qn[cond_y, 2] = qy_y[cond_y]
        qn[cond_y, 3] = qy_z[cond_y]

        qn[cond_z, 0] = qz_w[cond_z]
        qn[cond_z, 1] = qz_x[cond_z]
        qn[cond_z, 2] = qz_y[cond_z]
        qn[cond_z, 3] = qz_z[cond_z]

        q[t_neg] = qn

    return _safe_normalize(q)


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects "pits" terrain and applies restrictions:
    - For pit terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor):
        """Resample velocity commands with threshold."""
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        # First, call parent's update command
        super()._update_command()

        # Check which robots are currently on pit terrain (real-time check every step)
        on_pits = is_robot_on_terrain(self._env, "pits")

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._resample_command(left_pit_env_ids)

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation
            # Set heading to 0 for pit robots
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0

        # Update tracking state
        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedRLEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor):
        """Resample commands for the given environments."""
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[int(idx)] for idx in sampled_indices.tolist()], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """


class EndEffectorTwistTrajectoryCommand(CommandTerm):
    """End-effector trajectory sampling + twist command (paper Eq. (3)(4)).

    Command is the stacked vector: [vEE (3), wEE (3), goal_pos (3), goal_quat (4)], where
    vEE = (ri - rEE) / dt and wEE = (theta_i ⊟ theta_EE) / dt.

    The trajectory is represented in a robot-centric, gravity-aligned (yaw-only) task frame.
    """

    cfg: EndEffectorTwistTrajectoryCommandCfg

    def __init__(self, cfg: EndEffectorTwistTrajectoryCommandCfg, env: ManagerBasedRLEnv):
        self._goal_visualizer = None
        self._goal_marker_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/ee_goal")
        super().__init__(cfg, env)
        if not cfg.ee_body_name or not cfg.torso_body_name or not cfg.shoulder_body_name:
            raise ValueError(
                "EndEffectorTwistTrajectoryCommandCfg requires ee_body_name, torso_body_name, shoulder_body_name to be set."
            )
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._dt = float(env.cfg.decimation * env.cfg.sim.dt)

        self._ee_body_index = self.robot.body_names.index(cfg.ee_body_name)
        self._torso_body_index = self.robot.body_names.index(cfg.torso_body_name)
        self._shoulder_body_index = self.robot.body_names.index(cfg.shoulder_body_name)

        # Trajectory state in task frame
        self._start_pos_t = torch.zeros(self.num_envs, 3, device=self.device)
        self._start_quat_t = torch.zeros(self.num_envs, 4, device=self.device)
        self._start_quat_t[:, 0] = 1.0

        self._goal_pos_t = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_quat_t = torch.zeros(self.num_envs, 4, device=self.device)
        self._goal_quat_t[:, 0] = 1.0

        self._traj_duration = torch.ones(self.num_envs, device=self.device) * cfg.trajectory_duration_range[0]
        self._traj_time = torch.zeros(self.num_envs, device=self.device)
        self._use_local = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Command frame representation.
        # - "base": torso/body frame (full orientation)
        # - "control": gravity-aligned yaw-only torso frame (paper "control frame")
        self.command_frame: str = str(getattr(cfg, "command_frame", "control"))

        # Output command: [v(3), w(3), goal_pos(3), goal_quat(4)]
        self._command = torch.zeros(self.num_envs, 13, device=self.device)

        # Metrics buffers for per-trajectory goal error in world frame
        self._metric_goal_pos_err_sum = torch.zeros(self.num_envs, device=self.device)
        self._metric_goal_rot_err_sum = torch.zeros(self.num_envs, device=self.device)
        self._metric_goal_err_count = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _get_task_frame_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        torso_pos_w = self.robot.data.body_pos_w[:, self._torso_body_index]
        torso_quat_w = self.robot.data.body_quat_w[:, self._torso_body_index]
        return torso_pos_w, yaw_quat(torso_quat_w)

    def set_command_frame(self, command_frame: str, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Update the representation frame for the exposed command vector."""
        command_frame = str(command_frame)
        if command_frame == self.command_frame:
            return
        if command_frame not in ("base", "control"):
            raise ValueError(f"Unknown command_frame: {command_frame}. Expected 'base' or 'control'.")
        self.command_frame = command_frame
        self._update_command()

    def _get_body_pose_t(self, body_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        task_pos_w, task_quat_w = self._get_task_frame_w()
        body_pos_w = self.robot.data.body_pos_w[:, body_index]
        body_quat_w = self.robot.data.body_quat_w[:, body_index]
        pos_t, quat_t = subtract_frame_transforms(task_pos_w, task_quat_w, body_pos_w, body_quat_w)
        return pos_t, quat_t

    def _sample_goal_pos_t(self, env_ids: torch.Tensor) -> torch.Tensor:
        # Shoulder position in task frame
        shoulder_pos_t, _ = self._get_body_pose_t(self._shoulder_body_index)
        shoulder_pos_t = shoulder_pos_t[env_ids]

        # Rejection sampling in a unit sphere, scaled by radius
        radius = float(self.cfg.position_sphere_radius)
        front_hemisphere = bool(getattr(self.cfg, "front_hemisphere", False))
        front_min_x = float(getattr(self.cfg, "front_min_x", 0.0))
        max_tries = int(self.cfg.max_sampling_tries)
        n = env_ids.numel()

        out = torch.zeros(n, 3, device=self.device)
        valid = torch.zeros(n, dtype=torch.bool, device=self.device)

        # Cuboid rejection (in task frame, centered at torso/task origin)
        cub = self.cfg.reject_cuboid
        cub_min = torch.tensor([cub[0], cub[2], cub[4]], device=self.device)
        cub_max = torch.tensor([cub[1], cub[3], cub[5]], device=self.device)

        for _ in range(max_tries):
            remaining = torch.where(~valid)[0]
            if remaining.numel() == 0:
                break

            # Uniform in [-1,1]^3 then reject outside unit ball
            samp = sample_uniform(-1.0, 1.0, (remaining.numel(), 3), device=self.device)
            accept_mask = torch.norm(samp, dim=-1) <= 1.0
            if front_hemisphere:
                accept_mask &= samp[:, 0] >= front_min_x
            if not accept_mask.any():
                continue
            samp = samp[accept_mask]

            # Map to remaining slots
            rem_idx = remaining[accept_mask]
            pos = shoulder_pos_t[rem_idx] + radius * samp

            in_cuboid = torch.all((pos >= cub_min) & (pos <= cub_max), dim=-1)
            accept = ~in_cuboid

            out[rem_idx[accept]] = pos[accept]
            valid[rem_idx[accept]] = True

        # Fallback: if still invalid, just clamp to avoid NaNs
        if (~valid).any():
            out[~valid] = shoulder_pos_t[~valid]

        return out

    def _sample_goal_quat_t(self, goal_pos_t: torch.Tensor) -> torch.Tensor:
        # Reference orientation:
        #  - x-axis aligns with gravity
        #  - z-axis points from torso/task origin to goal position
        n = goal_pos_t.shape[0]
        x_axis = torch.zeros(n, 3, device=self.device)
        x_axis[:, 2] = -1.0  # gravity direction in task frame

        z_axis = _safe_normalize(goal_pos_t)
        # Avoid near-parallel axes
        dot = torch.sum(_safe_normalize(x_axis) * z_axis, dim=-1, keepdim=True).abs()
        alt_x = torch.zeros_like(x_axis)
        alt_x[:, 1] = 1.0
        x_axis = torch.where(dot > 0.95, alt_x, x_axis)
        x_axis = _safe_normalize(x_axis)

        y_axis = _safe_normalize(torch.cross(z_axis, x_axis, dim=-1))
        x_axis = _safe_normalize(torch.cross(y_axis, z_axis, dim=-1))

        ref_q = _quat_from_orthonormal_axes(x_axis, y_axis, z_axis)

        # Random perturbation about each axis bounded within pi/4
        bound = float(self.cfg.orientation_perturb_bound)
        euler = sample_uniform(-bound, bound, (n, 3), device=self.device)
        dq = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        return quat_mul(dq, ref_q)

    # reset函数被调用时会调用这个函数
    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(list(env_ids), dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        # Reset time
        self._traj_time[env_ids] = 0.0

        # Current EE pose (task frame) becomes start pose
        ee_pos_t, ee_quat_t = self._get_body_pose_t(self._ee_body_index)
        self._start_pos_t[env_ids] = ee_pos_t[env_ids]
        self._start_quat_t[env_ids] = ee_quat_t[env_ids]

        # Sample goal pose
        goal_pos = self._sample_goal_pos_t(env_ids)
        goal_quat = self._sample_goal_quat_t(goal_pos)
        self._goal_pos_t[env_ids] = goal_pos
        self._goal_quat_t[env_ids] = goal_quat

        # Sample duration
        d0, d1 = self.cfg.trajectory_duration_range
        self._traj_duration[env_ids] = sample_uniform(float(d0), float(d1), (env_ids.numel(),), device=self.device)

        # Local/global mix
        p_local = float(self.cfg.local_trajectory_probability)
        self._use_local[env_ids] = (torch.rand(env_ids.numel(), device=self.device) < p_local)

        # Reset per-trajectory metrics
        self._metric_goal_pos_err_sum[env_ids] = 0.0
        self._metric_goal_rot_err_sum[env_ids] = 0.0
        self._metric_goal_err_count[env_ids] = 0.0

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        self._metric_goal_pos_err_sum[env_ids] = 0.0
        self._metric_goal_rot_err_sum[env_ids] = 0.0
        self._metric_goal_err_count[env_ids] = 0.0
        return extras

    def _update_metrics(self):
        # Optional: expose norms of commands for logging
        self.metrics["ee_cmd_lin_vel_norm"] = torch.norm(self._command[:, 0:3], dim=-1)
        self.metrics["ee_cmd_ang_vel_norm"] = torch.norm(self._command[:, 3:6], dim=-1)
        self.metrics["ee_cmd_frame_is_control"] = torch.full(
            (self.num_envs,), float(self.command_frame == "control"), device=self.device
        )

        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_index]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_index]
        task_pos_w, task_quat_w = self._get_task_frame_w()
        goal_pos_w = task_pos_w + quat_apply(task_quat_w, self._goal_pos_t)
        goal_quat_w = quat_mul(task_quat_w, self._goal_quat_t)

        pos_err = torch.norm(ee_pos_w - goal_pos_w, dim=-1)
        rotvec_err = quat_box_minus(goal_quat_w, ee_quat_w)
        rot_err = torch.norm(rotvec_err, dim=-1)

        self._metric_goal_pos_err_sum += pos_err
        self._metric_goal_rot_err_sum += rot_err
        self._metric_goal_err_count += 1.0

        count = torch.clamp(self._metric_goal_err_count, min=1.0)
        self.metrics["ee_goal_pos_error_w"] = self._metric_goal_pos_err_sum / count
        self.metrics["ee_goal_rot_error_w"] = self._metric_goal_rot_err_sum / count

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if self._goal_visualizer is None:
                self._goal_visualizer = VisualizationMarkers(self._goal_marker_cfg)
            self._goal_visualizer.set_visibility(True)
        else:
            if self._goal_visualizer is not None:
                self._goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized or self._goal_visualizer is None:
            return
        task_pos_w, task_quat_w = self._get_task_frame_w()
        goal_pos_w = task_pos_w + quat_apply(task_quat_w, self._goal_pos_t)
        goal_quat_w = quat_mul(task_quat_w, self._goal_quat_t)
        self._goal_visualizer.visualize(goal_pos_w, goal_quat_w)

    # step函数被调用时会调用这个函数
    # 13 维 command 在 cfg.command_frame 指定的 frame 下定义
    def _update_command(self):
        # Advance time
        self._traj_time += self._dt

        # Compute current EE pose
        ee_pos_t, ee_quat_t = self._get_body_pose_t(self._ee_body_index)

        # Intermediate goal pose (Eq. (3))
        # Global: interpolate from fixed start_pos_t to goal_pos_t based on elapsed time
        # Local: always start from current ee_pos_t and take a single step towards goal
        use_local = self._use_local

        # 论文 3.2：local 轨迹每步重置起点以增加“接近目标”的样本密度；
        # global 轨迹在整个持续时间内保持固定起点。
        # For local trajectory: fixed small step (dt/duration), always from current pose
        # For global trajectory: accumulated time ratio from fixed start pose
        alpha_local = torch.clamp(self._dt / torch.clamp(self._traj_duration, min=1.0e-6), 0.0, 1.0)
        alpha_global = torch.clamp(self._traj_time / torch.clamp(self._traj_duration, min=1.0e-6), 0.0, 1.0)
        
        # Local always resets start to current pose; Global uses fixed sampled start
        pos_start = torch.where(use_local.unsqueeze(-1), ee_pos_t, self._start_pos_t)
        quat_start = torch.where(use_local.unsqueeze(-1), ee_quat_t, self._start_quat_t)
        alpha = torch.where(use_local, alpha_local, alpha_global)

        pos_i = pos_start + alpha.unsqueeze(-1) * (self._goal_pos_t - pos_start)
        quat_i = _quat_slerp(quat_start, self._goal_quat_t, alpha)

        # Twist command (Eq. (4))
        v_ee = (pos_i - ee_pos_t) / self._dt
        q_err = quat_mul(quat_i, quat_inv(ee_quat_t))
        rotvec = _quat_to_rotvec(q_err)
        w_ee = rotvec / self._dt

        goal_pos = self._goal_pos_t
        goal_quat = self._goal_quat_t

        if self.command_frame == "base":
            torso_quat_w = self.robot.data.body_quat_w[:, self._torso_body_index]
            control_quat_w = yaw_quat(torso_quat_w)
            q_control_to_base = quat_mul(quat_inv(torso_quat_w), control_quat_w)
            v_ee = quat_apply(q_control_to_base, v_ee)
            w_ee = quat_apply(q_control_to_base, w_ee)
            goal_pos = quat_apply(q_control_to_base, goal_pos)
            goal_quat = quat_mul(q_control_to_base, goal_quat)

        self._command[:, 0:3] = v_ee
        self._command[:, 3:6] = w_ee
        self._command[:, 6:9] = goal_pos
        self._command[:, 9:13] = goal_quat


@configclass
class EndEffectorTwistTrajectoryCommandCfg(CommandTermCfg):
    """Config matching paper Sec. 3.2 'Command Formulation'."""

    class_type: type = EndEffectorTwistTrajectoryCommand

    # 机器人与关键 body/link 名称
    asset_name: str = "robot"
    ee_body_name: str = ""
    torso_body_name: str = ""
    shoulder_body_name: str = ""

    # Command representation frame:
    # - "base": torso/body frame (full orientation)
    # - "control": gravity-aligned yaw-only torso frame (paper "control frame")
    command_frame: str = "control"

    # Trajectory sampling
    # 目标位置采样球半径（球心在肩部）
    position_sphere_radius: float = 0.5
    # Whether to restrict goal samples to the "front" hemisphere (task-frame +X).
    # This helps avoid sampling unreachable goals behind the robot for arms with one-sided joint limits.
    front_hemisphere: bool = False
    # Minimum x component (in unit-ball sample space) when front_hemisphere=True.
    # 0.0 => half-ball (hemisphere), >0.0 => narrower forward cone.
    front_min_x: float = 0.0
    # Cuboid bounds in task frame to reject goals inside torso/hip region: (xmin, xmax, ymin, ymax, zmin, zmax)
    reject_cuboid: tuple[float, float, float, float, float, float] = (-0.25, 0.35, -0.25, 0.25, -0.25, 0.35)
    # 目标位置采样的最大尝试次数
    max_sampling_tries: int = 64
    # 目标朝向扰动范围（每轴 ±bound）
    orientation_perturb_bound: float = math.pi / 6.0

    # Trajectory duration (seconds)
    # 轨迹时间范围（秒）
    trajectory_duration_range: tuple[float, float] = (1.0, 3.0)

    # Local trajectory subset probability
    # 选择 local 轨迹的概率（其余为 global）
    local_trajectory_probability: float = 0.5


class DesiredFeetSwingHeightCommand(CommandTerm):
    """Desired feet swing height command (paper Eq. (5)).

    Command is a 4D vector of desired foot heights for [FL, FR, RL, RR] in task frame.
    代码中写死顺序为 [FL, FR, RL, RR]
    """

    cfg: DesiredFeetSwingHeightCommandCfg

    def __init__(self, cfg: DesiredFeetSwingHeightCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if len(cfg.phase_offsets) != 4:
            raise ValueError("phase_offsets must have 4 elements (FL, FR, RL, RR).")
        self._dt = float(env.cfg.decimation * env.cfg.sim.dt)
        self._phase = torch.zeros(self.num_envs, device=self.device)
        self._max_height = torch.zeros(self.num_envs, device=self.device)
        self._command = torch.zeros(self.num_envs, 4, device=self.device)
        self._phase_offsets = torch.tensor(cfg.phase_offsets, device=self.device).view(1, 4)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if env_ids is None:
            self._phase = torch.rand(self.num_envs, device=self.device)
            return extras
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        self._phase[env_ids] = torch.rand(len(env_ids), device=self.device)
        return extras

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor):
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        if self.cfg.height_range is not None:
            r = torch.empty(len(env_ids), device=self.device)
            self._max_height[env_ids] = r.uniform_(*self.cfg.height_range)
        else:
            self._max_height[env_ids] = float(self.cfg.max_height)

    def _update_command(self):
        # Update gait phase in cycles [0, 1)
        self._phase = torch.remainder(self._phase + self._dt * float(self.cfg.gait_frequency), 1.0)
        phase = torch.remainder(self._phase.unsqueeze(-1) + self._phase_offsets, 1.0)
        heights = self._max_height.unsqueeze(-1) * torch.sin(2.0 * math.pi * phase)
        if self.cfg.clip_to_positive:
            heights = torch.clamp(heights, min=0.0)
        self._command = heights


@configclass
class DesiredFeetSwingHeightCommandCfg(CommandTermCfg):
    """Config for desired feet swing height command (paper Eq. (5))."""

    class_type: type = DesiredFeetSwingHeightCommand

    # 最大摆动高度（m），若 height_range 为 None 则固定使用该值
    max_height: float = 0.12
    # 采样高度范围（m），如需随机化可设置为 (min, max)
    height_range: tuple[float, float] | None = None
    # 步态频率（Hz），相位每秒前进 gait_frequency 个周期
    gait_frequency: float = 1.5
    # 相位偏移，顺序为 [FL, FR, RL, RR]（与足端顺序保持一致）
    phase_offsets: tuple[float, float, float, float] = (0.0, 0.5, 0.5, 0.0)
    # 是否将高度截断为非负（摆动期为正，支撑期为 0）
    clip_to_positive: bool = True
