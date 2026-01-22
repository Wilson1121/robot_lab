# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a checkpoint for Go2Arm and disable custom randomizations."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--fixed-commands",
    action="store_true",
    default=False,
    help="Use zero commands for standing diagnostics.",
)
parser.add_argument(
    "--zero-actions",
    action="store_true",
    default=False,
    help="Force zero actions to test passive standing stability.",
)
parser.add_argument(
    "--hold-default",
    action="store_true",
    default=False,
    help="Drive joints toward default pose to test standing stability.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow

# PLACEHOLDER: Extension template (do not remove this comment)


def _disable_randomizations(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    """Disable randomization/curriculum terms for play."""
    # 观测噪声
    # if hasattr(env_cfg, "observations"):
    #     if hasattr(env_cfg.observations, "policy"):
    #         env_cfg.observations.policy.enable_corruption = False
    #     if hasattr(env_cfg.observations, "critic"):
    #         env_cfg.observations.critic.enable_corruption = False

    if hasattr(env_cfg, "events") and env_cfg.events is not None:
        for name in (
            "randomize_rigid_body_material",
            # "randomize_rigid_body_mass_base",
            "randomize_rigid_body_mass_ee",
            # "randomize_apply_external_force_torque_base",
            "randomize_apply_external_force_torque_ee",
            # "randomize_push_robot",
            # "randomize_com_positions",
            # "randomize_reset_joints",
            # "randomize_actuator_gains_robot",
            # "randomize_actuator_gains_arm",
            # "randomize_reset_base",
            # legacy names used in some configs
            # "randomize_apply_external_force_torque",
            # "push_robot",
        ):
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)

    if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
        for name in ("command_levels_lin_vel", "command_levels_ang_vel", "terrain_levels"):
            if hasattr(env_cfg.curriculum, name):
                setattr(env_cfg.curriculum, name, None)


def _apply_fixed_commands(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    """Force command observations to zeros to test standing stability."""
    if hasattr(env_cfg, "commands") and env_cfg.commands is not None:
        if hasattr(env_cfg.commands, "base_velocity"):
            env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
            env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        if hasattr(env_cfg.commands, "feet_swing_height"):
            env_cfg.commands.feet_swing_height.max_height = 0.0

    if hasattr(env_cfg, "observations"):
        if hasattr(env_cfg.observations, "policy"):
            env_cfg.observations.policy.base_velocity_command = ObsTerm(
                func=lambda env: torch.zeros((env.num_envs, 3), device=env.device),
            )
            env_cfg.observations.policy.ee_twist_command = ObsTerm(
                func=lambda env: torch.zeros((env.num_envs, 13), device=env.device),
            )
            env_cfg.observations.policy.feet_swing_height_command = ObsTerm(
                func=lambda env: torch.zeros((env.num_envs, 4), device=env.device),
            )
        if hasattr(env_cfg.observations, "critic"):
            env_cfg.observations.critic.base_velocity_command = ObsTerm(
                func=lambda env: torch.zeros((env.num_envs, 3), device=env.device),
            )
            env_cfg.observations.critic.ee_twist_command = ObsTerm(
                func=lambda env: torch.zeros((env.num_envs, 13), device=env.device),
            )
            env_cfg.observations.critic.feet_swing_height_command = ObsTerm(
                func=lambda env: torch.zeros((env.num_envs, 4), device=env.device),
            )


def _compute_hold_default_actions(env: RslRlVecEnvWrapper) -> torch.Tensor:
    """Compute actions that drive joints to default positions for relative position control."""
    base_env = env.unwrapped
    if not hasattr(base_env, "action_manager"):
        return torch.zeros(env.num_envs, env.num_actions, device=base_env.device)
    asset = base_env.scene["robot"]
    action_chunks = []
    for term in base_env.action_manager._terms.values():
        joint_ids = term._joint_ids
        q_current = asset.data.joint_pos[:, joint_ids]
        q_default = asset.data.default_joint_pos[:, joint_ids]
        processed = q_default - q_current
        scale = term._scale
        offset = term._offset
        raw = (processed - offset) / scale
        action_chunks.append(raw)
    return torch.cat(action_chunks, dim=1)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    _disable_randomizations(env_cfg)
    if args_cli.fixed_commands:
        _apply_fixed_commands(env_cfg)

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # debug: print action offsets to verify whether relative actions include a non-zero offset
    try:
        base_env = env.unwrapped
        legs_term = base_env.action_manager._terms.get("legs")
        arm_term = base_env.action_manager._terms.get("arm")
        if legs_term is not None:
            legs_offset = legs_term._offset
            if hasattr(legs_offset, "shape"):
                legs_offset = legs_offset[0]
            print(f"[DEBUG] legs offset: {legs_offset}")
        if arm_term is not None:
            arm_offset = arm_term._offset
            if hasattr(arm_offset, "shape"):
                arm_offset = arm_offset[0]
            print(f"[DEBUG] arm offset: {arm_offset}")
    except Exception as exc:
        print(f"[DEBUG] offset print skipped: {exc}")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if args_cli.zero_actions:
                actions = torch.zeros(env.num_envs, env.num_actions, device=env.unwrapped.device)
            elif args_cli.hold_default:
                actions = _compute_hold_default_actions(env)
            else:
                actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
