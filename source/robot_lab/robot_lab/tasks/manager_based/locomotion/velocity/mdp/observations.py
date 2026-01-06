# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# =============================================================================
# Feet Contact / Air Time Observations
# Reference: robot_lab mdp/rewards.py - feet_contact, feet_air_time
# =============================================================================

def feet_contact_state(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Returns binary contact state for feet (1 if in contact, 0 otherwise).
    
    Reference: rewards.py - feet_contact (uses compute_first_contact)
               rewards.py - feet_stumble (uses net_forces_w)
    
    Args:
        env: The environment instance.
        sensor_cfg: The configuration for the contact sensor. Should specify body_names for the feet.
        threshold: Force threshold to determine contact. Defaults to 1.0 N.
    
    Returns:
        Tensor of shape (num_envs, num_feet) with binary contact states.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Get net contact forces for specified bodies
    # Shape: (num_envs, num_bodies, 3)
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    # Compute force magnitude and compare with threshold
    in_contact = (torch.norm(net_forces, dim=-1) > threshold).float()
    return in_contact


def feet_air_time(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Returns the current air time for each foot.
    
    Reference: rewards.py - feet_air_time (uses last_air_time)
               rewards.py - feet_air_time_positive_biped (uses current_air_time)
    
    Note:
        Requires ContactSensorCfg.track_air_time=True to work properly.
    
    Args:
        env: The environment instance.
        sensor_cfg: The configuration for the contact sensor. Should specify body_names for the feet.
    
    Returns:
        Tensor of shape (num_envs, num_feet) with current air time in seconds.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Use current_air_time for real-time observation
    # Alternative: last_air_time for previous air phase duration (used in reward)
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return air_time


def static_friction(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Returns the static friction coefficient for each foot from PhysX material properties.
    
    This reads the randomized friction values applied by domain randomization
    (randomize_rigid_body_material event). Returns friction for each foot body.
    
    Note:
        The friction is read from PhysX material properties which are set during
        domain randomization. Requires asset_cfg.body_names to specify foot bodies.
        
        PhysX materials are indexed by shapes, not bodies. Each body may have
        multiple collision shapes. This function finds the first shape of each
        specified body and returns its static friction.
    
    Args:
        env: The environment instance.
        asset_cfg: The configuration for the robot asset. Should specify body_names for feet.
    
    Returns:
        Tensor of shape (num_envs, num_feet) with static friction coefficients.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get material properties from PhysX: shape is (num_envs, num_shapes, 3)
    # where the 3 values are: static_friction, dynamic_friction, restitution
    materials = asset.root_physx_view.get_material_properties()
    
    # Get body_ids for feet from asset_cfg
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        num_bodies = asset.num_bodies
        body_ids = list(range(*body_ids.indices(num_bodies)))
    
    # Compute num_shapes_per_body to map body_id to shape indices
    # Reference: isaaclab/envs/mdp/events.py - randomize_rigid_body_material
    num_shapes_per_body = []
    for link_path in asset.root_physx_view.link_paths[0]:
        link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)
        num_shapes_per_body.append(link_physx_view.max_shapes)
    
    # Extract static friction for each specified body (using first shape of each body)
    static_frictions_list = []
    for body_id in body_ids:
        # Calculate shape index for this body
        shape_idx = sum(num_shapes_per_body[:body_id])
        # Get static friction (index 0) for this shape
        static_frictions_list.append(materials[:, shape_idx, 0])
    
    # Stack to create (num_envs, num_feet) tensor
    static_frictions = torch.stack(static_frictions_list, dim=1)
    
    return static_frictions


# =============================================================================
# Domain Randomization Observations
# These observations read values set by domain randomization events
# Reference: isaaclab_nhb/envs/deepmimic_env/mdp/observation/observations.py
# =============================================================================

def base_external_wrench(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Returns the external force and torque applied to the base body.
    
    This reads the external wrench set by randomize_apply_external_force_torque event.
    The wrench is in body frame coordinates.
    
    Reference: isaaclab_nhb - push_force, push_torque
    
    Args:
        env: The environment instance.
        asset_cfg: The configuration for the robot asset. Should specify body_names for base.
    
    Returns:
        Tensor of shape (num_envs, 6) with [force_x, force_y, force_z, torque_x, torque_y, torque_z].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get body_ids from asset_cfg
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = [0]  # Default to first body (base)
    
    # Read external force and torque from asset buffers
    # Note: Using protected attributes _external_force_b/_external_torque_b as IsaacLab
    #       does not provide public getters. This is consistent with isaaclab_nhb implementation.
    # Shape: (num_envs, num_bodies, 3)
    external_force = asset._external_force_b[:, body_ids, :].flatten(1)  # (num_envs, 3*num_bodies)
    external_torque = asset._external_torque_b[:, body_ids, :].flatten(1)  # (num_envs, 3*num_bodies)
    
    # For single body, concatenate to form wrench: (num_envs, 6)
    wrench = torch.cat([external_force, external_torque], dim=-1)
    
    return wrench


def base_external_push_velocity(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Returns the current root velocity which includes push disturbances.
    
    This returns the root linear and angular velocity in world frame.
    When push_by_setting_velocity is applied, these values reflect the disturbance.
    
    Note:
        This returns the actual velocity, not the push delta. To get the pure push
        value, you would need to track the velocity before and after the push event.
    
    Args:
        env: The environment instance.
        asset_cfg: The configuration for the robot asset.
    
    Returns:
        Tensor of shape (num_envs, 6) with [lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get root velocity in world frame: (num_envs, 6)
    # Contains [lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
    root_vel = asset.data.root_vel_w
    
    return root_vel


def base_mass_disturbance(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Returns the mass disturbance (delta from default) for the base body.
    
    This reads the difference between current mass and cached default mass,
    reflecting the randomization applied by randomize_rigid_body_mass.
    
    Reference: isaaclab_nhb - mass_delta_norm
    
    Note:
        Uses cached default mass to ensure correct delta calculation after randomization.
    
    Args:
        env: The environment instance.
        asset_cfg: The configuration for the robot asset. Should specify body_names for base.
    
    Returns:
        Tensor of shape (num_envs, 1) with mass disturbance value.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get body_ids from asset_cfg
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = [0]  # Default to first body (base)
    body_id = body_ids[0] if isinstance(body_ids, list) else body_ids
    
    # Get current mass
    current_mass = asset.root_physx_view.get_masses()[:, body_id].to(asset.device)
    
    # Cache default mass on first call (before randomization takes effect in observation)
    cache_name = f"_default_mass_{asset_cfg.name}_{body_id}"
    if not hasattr(env, cache_name):
        # Use the default_mass from asset data if available
        setattr(env, cache_name, asset.data.default_mass[:, body_id].clone())
    default_mass = getattr(env, cache_name)
    
    # Compute mass disturbance
    mass_disturbance = (current_mass - default_mass).unsqueeze(-1)
    
    return mass_disturbance


def ee_external_wrench(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Returns the external force and torque applied to the end-effector body.
    
    This reads the external wrench set by randomize_apply_external_force_torque event.
    The wrench is in body frame coordinates.
    
    Reference: isaaclab_nhb - push_force, push_torque
    
    Args:
        env: The environment instance.
        asset_cfg: The configuration for the robot asset. Should specify body_names for end-effector.
    
    Returns:
        Tensor of shape (num_envs, 6) with [force_x, force_y, force_z, torque_x, torque_y, torque_z].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get body_ids from asset_cfg
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        raise ValueError("ee_external_wrench requires explicit body_names for end-effector")
    
    # Read external force and torque from asset buffers
    # Note: Using protected attributes _external_force_b/_external_torque_b as IsaacLab
    #       does not provide public getters. This is consistent with isaaclab_nhb implementation.
    external_force = asset._external_force_b[:, body_ids, :].flatten(1)  # (num_envs, 3*num_bodies)
    external_torque = asset._external_torque_b[:, body_ids, :].flatten(1)  # (num_envs, 3*num_bodies)

    # Concatenate to form wrench: (num_envs, 6)
    wrench = torch.cat([external_force, external_torque], dim=-1)

    return wrench


def ee_mass_disturbance(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Returns the mass disturbance (delta from default) for the end-effector body.
    
    This reads the difference between current mass and cached default mass,
    reflecting the randomization applied by randomize_rigid_body_mass.
    
    Reference: isaaclab_nhb - mass_delta_norm
    
    Note:
        Uses cached default mass to ensure correct delta calculation after randomization.
    
    Args:
        env: The environment instance.
        asset_cfg: The configuration for the robot asset. Should specify body_names for end-effector.
    
    Returns:
        Tensor of shape (num_envs, 1) with mass disturbance value.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get body_ids from asset_cfg
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        raise ValueError("ee_mass_disturbance requires explicit body_names for end-effector")
    body_id = body_ids[0] if isinstance(body_ids, list) else body_ids
    
    # Get current mass
    current_mass = asset.root_physx_view.get_masses()[:, body_id].to(asset.device)
    
    # Cache default mass on first call
    cache_name = f"_default_mass_ee_{asset_cfg.name}_{body_id}"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, asset.data.default_mass[:, body_id].clone())
    default_mass = getattr(env, cache_name)
    
    # Compute mass disturbance
    mass_disturbance = (current_mass - default_mass).unsqueeze(-1)
    
    return mass_disturbance


# =============================================================================

def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor
