#!/usr/bin/env python3
"""Test script to verify Multi-critic PPO configuration is correctly set up.

Run this script before training to check if:
1. The agent config has critic_names set
2. The env config has reward_group_terms set
3. The env outputs reward_groups in extras
"""

import sys
import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test Multi-critic PPO configuration.")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-Rough-Go2Arm-v0", 
                    help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments.")
parser.add_argument(
    "--ee_frame_switch_steps",
    type=int,
    default=None,
    help="Override ee_twist command-frame curriculum switch step for a quick sanity check.",
)
parser.add_argument(
    "--ee_frame_check_steps",
    type=int,
    default=10,
    help="Number of env steps to run when checking ee_twist command-frame curriculum.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Always run headless for this test
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401


def test_multi_critic():
    """Test multi-critic configuration."""
    print("\n" + "="*60)
    print("Multi-Critic PPO Configuration Test")
    print("="*60 + "\n")
    
    # 1. Load agent config
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)

    if (
        args_cli.ee_frame_switch_steps is not None
        and getattr(getattr(env_cfg, "curriculum", None), "ee_twist_command_frame", None) is not None
    ):
        env_cfg.curriculum.ee_twist_command_frame.params["switch_after_steps"] = int(args_cli.ee_frame_switch_steps)
    
    # Load agent config
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    
    print("[1] Checking agent_cfg.policy.critic_names...")
    critic_names = getattr(agent_cfg.policy, "critic_names", None)
    if critic_names:
        print(f"    ✅ critic_names = {critic_names}")
    else:
        print(f"    ❌ critic_names not set! Multi-critic will NOT be enabled.")
        return False
    
    # 2. Check env config
    print("\n[2] Checking env_cfg.reward_group_terms...")
    reward_group_terms = getattr(env_cfg, "reward_group_terms", None)
    if reward_group_terms:
        print(f"    ✅ reward_group_terms defined:")
        for group, terms in reward_group_terms.items():
            print(f"       - {group}: {len(terms)} terms")
    else:
        print(f"    ⚠️  reward_group_terms not explicitly set, will use auto-assignment")
    
    # 3. Create environment and check extras
    print("\n[3] Creating environment and checking step() output...")
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Get the unwrapped env for accessing attributes
    unwrapped_env = env.unwrapped
    num_envs = unwrapped_env.num_envs
    num_actions = unwrapped_env.action_manager.total_action_dim
    device = unwrapped_env.device
    
    # Reset and step
    obs, info = env.reset()
    actions = torch.zeros(num_envs, num_actions, device=device)
    obs, reward, terminated, truncated, extras = env.step(actions)
    
    if "reward_groups" in extras:
        print(f"    ✅ extras['reward_groups'] present!")
        print(f"       Keys: {list(extras['reward_groups'].keys())}")
        for name, val in extras['reward_groups'].items():
            print(f"       - {name}: shape={val.shape}, sum={val.sum().item():.4f}")
    else:
        print(f"    ❌ extras['reward_groups'] NOT found!")
        print(f"       Available extras keys: {list(extras.keys())}")
        return False
    
    # 4. Verify critic_names match reward_group keys
    print("\n[4] Checking critic_names match reward_groups...", flush=True)
    reward_group_keys = set(extras["reward_groups"].keys())
    critic_names_set = set(critic_names)
    
    if critic_names_set == reward_group_keys:
        print(f"    ✅ Perfect match!", flush=True)
    elif critic_names_set.issubset(reward_group_keys):
        print(f"    ✅ All critic_names found in reward_groups", flush=True)
        extra_groups = reward_group_keys - critic_names_set
        if extra_groups:
            print(f"       (Extra groups not used: {extra_groups})", flush=True)
    else:
        missing = critic_names_set - reward_group_keys
        print(f"    ❌ Missing reward groups for critics: {missing}", flush=True)
        return False
    
    # 5. Check to_dict() includes critic_names
    print("\n[5] Checking agent_cfg.to_dict() includes critic_names...", flush=True)
    cfg_dict = agent_cfg.to_dict()
    policy_cfg = cfg_dict.get("policy", {})
    if "critic_names" in policy_cfg:
        print(f"    ✅ critic_names in to_dict(): {policy_cfg['critic_names']}", flush=True)
    else:
        print(f"    ❌ critic_names NOT in to_dict()!", flush=True)
        print(f"       Policy keys: {list(policy_cfg.keys())}", flush=True)
        return False

    # 6. Optional: check EE twist command-frame curriculum wiring.
    if args_cli.ee_frame_switch_steps is not None:
        print("\n[6] Checking ee_twist command-frame curriculum...", flush=True)
        if getattr(unwrapped_env.command_manager, "get_term", None) is None:
            print("    ⚠️  command_manager.get_term not available, skipping.", flush=True)
        else:
            try:
                ee_term = unwrapped_env.command_manager.get_term("ee_twist")
            except Exception as e:
                print(f"    ⚠️  ee_twist term not available: {e}", flush=True)
            else:
                before = getattr(ee_term, "command_frame", None)
                for _ in range(int(args_cli.ee_frame_check_steps)):
                    obs, reward, terminated, truncated, extras = env.step(actions)
                after = getattr(ee_term, "command_frame", None)
                print(
                    f"    ee_twist.command_frame: {before} -> {after} "
                    f"(switch_after_steps={args_cli.ee_frame_switch_steps})",
                    flush=True,
                )
    
    env.close()
    
    print("\n" + "="*60, flush=True)
    print("✅ All Multi-Critic configuration checks passed!", flush=True)
    print("="*60 + "\n", flush=True)
    return True


if __name__ == "__main__":
    success = test_multi_critic()
    simulation_app.close()
    sys.exit(0 if success else 1)
