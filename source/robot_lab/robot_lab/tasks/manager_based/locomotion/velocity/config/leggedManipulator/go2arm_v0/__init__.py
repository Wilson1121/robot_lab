# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# 注册文件

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Go2Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # UnitreeGo2ArmFlatEnvCfg这是一个环境配置类
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2ArmFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ArmFlatPPORunnerCfg",
        # "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2ArmFlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Go2Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2ArmRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ArmRoughPPORunnerCfg",
        # "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2ArmRoughTrainerCfg",
    },
)

