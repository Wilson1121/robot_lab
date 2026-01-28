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
    "--zero-legs",
    action="store_true",
    default=False,
    help="Force leg actions to zero (keep arm actions) to test arm tracking when the base is unsupported.",
)
parser.add_argument(
    "--zero-arm",
    action="store_true",
    default=False,
    help="Force arm actions to zero (keep leg actions) to isolate locomotion behavior.",
)
parser.add_argument(
    "--hold-default",
    action="store_true",
    default=False,
    help="Drive joints toward default pose to test standing stability.",
)
parser.add_argument(
    "--goal-vis",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Visualize end-effector sampled goal pose (and current EE) in the viewport.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Print debug info (commands/actions) periodically.",
)
parser.add_argument(
    "--debug-interval",
    type=int,
    default=50,
    help="Debug print interval in steps (when --debug is enabled).",
)
parser.add_argument(
    "--debug-env",
    type=int,
    default=0,
    help="Environment index to print (when --debug is enabled).",
)
parser.add_argument(
    "--ee-fast",
    action="store_true",
    default=False,
    help="Override EE command sampling to shorter durations for easier visualization during play.",
)
parser.add_argument(
    "--stand-still",
    action="store_true",
    default=False,
    help="Force locomotion commands to zero (keep EE commands) for easier arm motion inspection.",
)
parser.add_argument(
    "--no-ee-task",
    action="store_true",
    default=False,
    help="Disable EE task conditioning during play by zeroing the ee_twist command observations (policy/critic). Useful to isolate base command tracking.",
)
parser.add_argument(
    "--keep-disturbances",
    action="store_true",
    default=False,
    help="Do not disable disturbance/randomization event terms for play.",
)
parser.add_argument(
    "--joint2-bias",
    type=float,
    default=0.0,
    help="Play-only debug: add a positive bias to arm joint2 action when joint2 is near/below the soft lower limit. (<=0 disables)",
)
parser.add_argument(
    "--joint2-bias-margin",
    type=float,
    default=0.0,
    help="Margin (rad) above joint2 soft lower limit to start applying --joint2-bias.",
)
parser.add_argument(
    "--no-terminate",
    action="store_true",
    default=False,
    help="Disable common termination terms (bad_orientation/illegal_contact/time_out) to keep the episode running for debugging.",
)
parser.add_argument(
    "--freeze-ee-goal",
    action="store_true",
    default=False,
    help="Freeze EE goal sampling by setting ee_twist.resampling_time_range to a very large value (prevents goal from changing).",
)
parser.add_argument(
    "--ee-command-frame",
    type=str,
    default="base",
    choices=("base", "control"),
    help="Override EE twist command frame ('base' or 'control') for play. Default: base.",
)
parser.add_argument(
    "--manual-action-clip",
    action="store_true",
    default=False,
    help="Manually clamp actions to [-clip_actions, clip_actions] before stepping the env (useful to verify clipping behavior).",
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
            "randomize_rigid_body_mass_base",
            "randomize_rigid_body_mass_ee",
            "randomize_apply_external_force_torque_base",
            "randomize_apply_external_force_torque_ee",
            "randomize_push_robot",
            "randomize_com_positions",
            "randomize_reset_joints",
            "randomize_actuator_gains_robot",
            "randomize_actuator_gains_arm",
            "randomize_reset_base",
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


def _disable_ee_task_observations(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    """Disable EE task conditioning for the policy/critic by zeroing ee_twist command observations."""
    if not hasattr(env_cfg, "observations"):
        return
    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.ee_twist_command = ObsTerm(
            func=lambda env: torch.zeros((env.num_envs, 13), device=env.device),
        )
    if hasattr(env_cfg.observations, "critic"):
        env_cfg.observations.critic.ee_twist_command = ObsTerm(
            func=lambda env: torch.zeros((env.num_envs, 13), device=env.device),
        )


def _disable_terminations(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    """Disable common termination terms so the episode doesn't reset while debugging (e.g. after a fall)."""
    if not hasattr(env_cfg, "terminations") or env_cfg.terminations is None:
        return
    for name in ("bad_orientation", "illegal_contact", "time_out"):
        if hasattr(env_cfg.terminations, name):
            setattr(env_cfg.terminations, name, None)


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


def _try_get_action_slices(base_env) -> dict[str, slice]:
    """Best-effort mapping from action term name -> slice in flattened action vector."""
    if not hasattr(base_env, "action_manager") or not hasattr(base_env.action_manager, "_terms"):
        return {}
    slices: dict[str, slice] = {}
    start = 0
    for name, term in base_env.action_manager._terms.items():
        dim = None
        for attr in ("action_dim", "_action_dim"):
            if hasattr(term, attr):
                try:
                    dim = int(getattr(term, attr))
                    break
                except Exception:
                    pass
        if dim is None and hasattr(term, "_joint_ids"):
            try:
                dim = int(len(term._joint_ids))
            except Exception:
                dim = None
        if dim is None or dim <= 0:
            continue
        slices[str(name)] = slice(start, start + dim)
        start += dim
    return slices


def _format_vec(vec: torch.Tensor, max_elems: int = 6) -> str:
    v = vec.detach().flatten().cpu()
    n = min(int(v.numel()), int(max_elems))
    items = ", ".join([f"{float(v[i]): .3f}" for i in range(n)])
    if v.numel() > n:
        items += ", ..."
    return f"[{items}]"


def _try_get_joint_indices(robot, joint_names: list[str]) -> list[int]:
    out: list[int] = []
    if not hasattr(robot, "joint_names"):
        return out
    for name in joint_names:
        try:
            out.append(int(robot.joint_names.index(name)))
        except Exception:
            pass
    return out


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
    if not args_cli.keep_disturbances:
        _disable_randomizations(env_cfg)
    if args_cli.fixed_commands:
        _apply_fixed_commands(env_cfg)

    if args_cli.no_terminate:
        _disable_terminations(env_cfg)

    if args_cli.stand_still and hasattr(env_cfg, "commands") and env_cfg.commands is not None:
        # Keep EE commands but stop the base from being commanded to move.
        if hasattr(env_cfg.commands, "base_velocity"):
            env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
            env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        if hasattr(env_cfg.commands, "feet_swing_height"):
            env_cfg.commands.feet_swing_height.max_height = 0.0

    if args_cli.no_ee_task:
        _disable_ee_task_observations(env_cfg)

    if args_cli.ee_fast and hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "ee_twist"):
        # Make the commanded EE motion more visible in play.
        try:
            env_cfg.commands.ee_twist.resampling_time_range = (3.0, 5.0)
            env_cfg.commands.ee_twist.trajectory_duration_range = (2.0, 4.0)
            env_cfg.commands.ee_twist.local_trajectory_probability = 0.0
        except Exception:
            pass

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

    # Enable goal visualization for EE twist command (implemented in the command term).
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "ee_twist"):
        try:
            env_cfg.commands.ee_twist.debug_vis = bool(args_cli.goal_vis)
        except Exception:
            pass
        # Override EE command frame for play.
        try:
            env_cfg.commands.ee_twist.command_frame = str(args_cli.ee_command_frame)
        except Exception:
            pass
        if args_cli.freeze_ee_goal:
            try:
                env_cfg.commands.ee_twist.resampling_time_range = (1.0e9, 1.0e9)
            except Exception:
                pass
        if args_cli.debug:
            try:
                ee_cfg = env_cfg.commands.ee_twist
                print("[DEBUG] EE twist cfg:")
                for k in (
                    "command_frame",
                    "resampling_time_range",
                    "trajectory_duration_range",
                    "local_trajectory_probability",
                    "ee_pos_kp",
                    "ee_rot_kp",
                    "ee_ang_vel_max",
                    "ee_lin_vel_max",
                ):
                    if hasattr(ee_cfg, k):
                        print(f"        {k}={getattr(ee_cfg, k)}")
            except Exception:
                pass

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

    base_env = env.unwrapped
    action_slices = _try_get_action_slices(base_env)
    robot = None
    base_body_id = None
    arm_joint_ids: list[int] = []
    leg_joint_ids: list[int] = []
    try:
        if hasattr(base_env, "scene"):
            try:
                robot = base_env.scene["robot"]
            except Exception:
                robot = None
        if robot is not None:
            if hasattr(robot, "body_names") and "base" in robot.body_names:
                base_body_id = int(robot.body_names.index("base"))
            arm_joint_ids = _try_get_joint_indices(robot, ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"])
            leg_joint_ids = _try_get_joint_indices(
                robot,
                [
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
                ],
            )
    except Exception:
        pass

    if args_cli.debug:
        print("[DEBUG] Play flags:")
        print(
            f"        fixed_commands={args_cli.fixed_commands} zero_actions={args_cli.zero_actions} hold_default={args_cli.hold_default}"
        )
        print(f"        zero_legs={args_cli.zero_legs} zero_arm={args_cli.zero_arm} manual_action_clip={args_cli.manual_action_clip}")
        print(f"        ee_command_frame={args_cli.ee_command_frame}")
        print(f"        no_ee_task={args_cli.no_ee_task}")
        print(
            f"        goal_vis={args_cli.goal_vis} ee_fast={args_cli.ee_fast} keep_disturbances={args_cli.keep_disturbances} keyboard={args_cli.keyboard}"
        )
        if action_slices:
            print(f"[DEBUG] Action term slices: {action_slices}")
        try:
            print(f"[DEBUG] agent_cfg.clip_actions={getattr(agent_cfg, 'clip_actions', None)}")
        except Exception:
            pass
        if hasattr(base_env, "command_manager") and hasattr(base_env.command_manager, "_terms"):
            try:
                print(f"[DEBUG] Command terms: {list(base_env.command_manager._terms.keys())}")
            except Exception:
                pass
        if hasattr(base_env, "events") and hasattr(base_env.events, "_terms"):
            try:
                enabled_events = [k for k, v in base_env.events._terms.items() if v is not None]
                print(f"[DEBUG] Enabled event terms: {enabled_events}")
            except Exception:
                pass
        if robot is not None:
            print(f"[DEBUG] base_body_id={base_body_id} arm_joint_ids={arm_joint_ids} leg_joint_ids={leg_joint_ids}")

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
    last_actions_raw = None
    last_actions_used = None
    last_dones = None
    joint2_bias_last = None
    joint2_bias_q2_last = None
    joint2_bias_lo_last = None
    joint2_bias_a2_before_last = None
    joint2_bias_a2_after_last = None
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if args_cli.zero_actions:
                actions_raw = torch.zeros(env.num_envs, env.num_actions, device=env.unwrapped.device)
            elif args_cli.hold_default:
                actions_raw = _compute_hold_default_actions(env)
            else:
                actions_raw = policy(obs)

            actions = actions_raw
            # Optional action masking for diagnostics.
            if args_cli.zero_legs and "legs" in action_slices:
                actions[:, action_slices["legs"]] = 0.0
            if args_cli.zero_arm and "arm" in action_slices:
                actions[:, action_slices["arm"]] = 0.0

            # Play-only debug: rescue joint2 if policy drives it into one-sided lower limit.
            joint2_bias_last = None
            joint2_bias_q2_last = None
            joint2_bias_lo_last = None
            joint2_bias_a2_before_last = None
            joint2_bias_a2_after_last = None
            if (
                args_cli.joint2_bias > 0.0
                and not args_cli.zero_actions
                and not args_cli.hold_default
                and robot is not None
                and "arm" in action_slices
                and len(arm_joint_ids) >= 2
                and actions.shape[1] > int(action_slices["arm"].start) + 1
                and hasattr(robot.data, "soft_joint_pos_limits")
            ):
                joint2_action_idx = int(action_slices["arm"].start) + 1
                joint2_joint_id = int(arm_joint_ids[1])
                q2 = robot.data.joint_pos[:, joint2_joint_id]
                lo = robot.data.soft_joint_pos_limits[:, joint2_joint_id, 0]
                thresh = lo + float(args_cli.joint2_bias_margin)
                mask = q2 < thresh
                if bool(mask.any().item()):
                    a2_before = actions[:, joint2_action_idx].clone()
                    actions[mask, joint2_action_idx] = torch.clamp(
                        actions[mask, joint2_action_idx] + float(args_cli.joint2_bias), min=-1.0, max=1.0
                    )
                    a2_after = actions[:, joint2_action_idx].clone()
                    joint2_bias_last = mask
                    joint2_bias_q2_last = q2.clone()
                    joint2_bias_lo_last = lo.clone()
                    joint2_bias_a2_before_last = a2_before
                    joint2_bias_a2_after_last = a2_after
            # Optional manual clipping (RslRlVecEnvWrapper may already clip).
            if args_cli.manual_action_clip and getattr(agent_cfg, "clip_actions", None) is not None:
                try:
                    clip = float(agent_cfg.clip_actions)
                    if clip > 0.0:
                        actions = torch.clamp(actions, min=-clip, max=clip)
                except Exception:
                    pass
            # env stepping
            obs, _, dones, _ = env.step(actions)
            last_actions_raw = actions_raw
            last_actions_used = actions
            last_dones = dones
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

        timestep += 1

        if args_cli.debug and (timestep % max(int(args_cli.debug_interval), 1) == 0):
            env_id = int(args_cli.debug_env)
            if env_id < 0 or env_id >= int(env.num_envs):
                env_id = 0
            try:
                cmd_base = base_env.command_manager.get_command("base_velocity")
            except Exception:
                cmd_base = None
            try:
                cmd_ee = base_env.command_manager.get_command("ee_twist")
            except Exception:
                cmd_ee = None

            print(f"[DEBUG] step={timestep} env={env_id}")
            try:
                if last_dones is not None:
                    done = bool(last_dones[env_id].item())
                    print(f"        done={int(done)}")
                    if done:
                        print("        note: env reset -> commands/goals may resample and goal marker can jump")
            except Exception:
                pass
            if robot is not None:
                try:
                    if base_body_id is not None:
                        base_z = float(robot.data.body_pos_w[env_id, base_body_id, 2].item())
                    else:
                        base_z = float(robot.data.root_pos_w[env_id, 2].item())
                    pg = None
                    if hasattr(robot.data, "projected_gravity_b"):
                        pg = robot.data.projected_gravity_b[env_id]
                    if pg is not None:
                        print(f"        base_z={base_z:.3f} proj_g={_format_vec(pg,3)}")
                    else:
                        print(f"        base_z={base_z:.3f}")
                    try:
                        if hasattr(robot.data, "root_lin_vel_b") and hasattr(robot.data, "root_ang_vel_b"):
                            vxy_b = robot.data.root_lin_vel_b[env_id, 0:2]
                            wz_b = robot.data.root_ang_vel_b[env_id, 2]
                            print(f"        base_vel vxy={_format_vec(vxy_b,2)} wz={float(wz_b.item()): .3f}")
                    except Exception:
                        pass
                    if arm_joint_ids:
                        q = robot.data.joint_pos[env_id, arm_joint_ids]
                        qd = robot.data.joint_vel[env_id, arm_joint_ids]
                        print(
                            f"        arm_q={_format_vec(q,6)} arm_qd={_format_vec(qd,6)}"
                        )
                        if hasattr(robot.data, "soft_joint_pos_limits"):
                            try:
                                lim = robot.data.soft_joint_pos_limits[env_id, arm_joint_ids]
                                lo = lim[:, 0]
                                hi = lim[:, 1]
                                margin = torch.minimum(q - lo, hi - q)
                                min_margin, min_idx = torch.min(margin, dim=0)
                                min_idx = int(min_idx.item())
                                joint_name = None
                                if hasattr(robot, "joint_names"):
                                    try:
                                        joint_name = str(robot.joint_names[arm_joint_ids[min_idx]])
                                    except Exception:
                                        joint_name = None
                                q_min = float(q[min_idx].item())
                                lo_min = float(lo[min_idx].item())
                                hi_min = float(hi[min_idx].item())
                                print(
                                    f"        arm_limit_margin_min={float(min_margin.item()):.3f} near_limit={bool(torch.any(margin < 0.05).item())} min_joint={joint_name} q={q_min:.3f} lim=[{lo_min:.3f},{hi_min:.3f}]"
                                )
                            except Exception:
                                pass
                except Exception as exc:
                    print(f"        robot state debug skipped: {exc}")
            # EE goal tracking metrics exposed by command term (if available)
            try:
                ee_term = base_env.command_manager.get_term("ee_twist")
                metrics = getattr(ee_term, "metrics", None)
                if isinstance(metrics, dict):
                    if "ee_goal_pos_err" in metrics:
                        v = metrics["ee_goal_pos_err"][env_id]
                        print(f"        metric ee_goal_pos_err={float(v.item()):.3f}")
                    if "ee_goal_rot_err" in metrics:
                        v = metrics["ee_goal_rot_err"][env_id]
                        print(f"        metric ee_goal_rot_err={float(v.item()):.3f}")
                    if "ee_goal_reached" in metrics:
                        v = metrics["ee_goal_reached"][env_id]
                        print(f"        metric ee_goal_reached={float(v.item()):.0f}")
            except Exception:
                pass
            if cmd_base is not None:
                vxy = cmd_base[env_id, 0:2]
                wz = cmd_base[env_id, 2:3]
                print(f"        base_cmd vxy={_format_vec(vxy, 2)} wz={float(wz.item()): .3f}")
            if cmd_ee is not None and cmd_ee.shape[-1] >= 13:
                v = cmd_ee[env_id, 0:3]
                w = cmd_ee[env_id, 3:6]
                goal_p = cmd_ee[env_id, 6:9]
                print(
                    f"        ee_cmd |v|={float(torch.norm(v).item()):.3f} |w|={float(torch.norm(w).item()):.3f} v={_format_vec(v,3)} w={_format_vec(w,3)}"
                )
                print(f"        ee_goal_p={_format_vec(goal_p,3)}")
            try:
                a = last_actions_used[env_id] if last_actions_used is not None else actions[env_id]
                a_raw = last_actions_raw[env_id] if last_actions_raw is not None else None
                msg = (
                    f"        action_used |a|={float(torch.norm(a).item()):.3f} mean|a|={float(torch.mean(torch.abs(a)).item()):.3f}"
                )
                if a_raw is not None:
                    msg += (
                        f" |a_raw|={float(torch.norm(a_raw).item()):.3f} mean|a_raw|={float(torch.mean(torch.abs(a_raw)).item()):.3f}"
                    )
                print(msg)
                if "arm" in action_slices:
                    arm_a = a[action_slices["arm"]]
                    print(
                        f"        arm_action |a|={float(torch.norm(arm_a).item()):.3f} mean|a|={float(torch.mean(torch.abs(arm_a)).item()):.3f} a={_format_vec(arm_a,6)}"
                    )
                if "legs" in action_slices:
                    legs_a = a[action_slices["legs"]]
                    print(
                        f"        legs_action |a|={float(torch.norm(legs_a).item()):.3f} mean|a|={float(torch.mean(torch.abs(legs_a)).item()):.3f}"
                    )
            except Exception as exc:
                print(f"        action debug skipped: {exc}")
            # Joint2 rescue debug (shows action before/after bias).
            try:
                if joint2_bias_last is not None and bool(joint2_bias_last[env_id].item()):
                    a2_b = float(joint2_bias_a2_before_last[env_id].item())
                    a2_a = float(joint2_bias_a2_after_last[env_id].item())
                    q2 = float(joint2_bias_q2_last[env_id].item())
                    lo = float(joint2_bias_lo_last[env_id].item())
                    print(
                        f"        joint2_bias applied: q2={q2:.3f} soft_lo={lo:.3f} a2 {a2_b:.3f}->{a2_a:.3f}"
                    )
            except Exception:
                pass
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep >= args_cli.video_length:
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
