# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2ArmRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 每个环境每次更新前收集的步数。值越小更新频率越高，但样本利用率低；值越大样本利用率高但更新滞后
    num_steps_per_env = 24  
    # 最大迭代次数。值越大训练时间越长，通常能获得更好的性能但收益递减
    max_iterations = 20000
    # 保存模型的间隔。值越小磁盘占用越多，值越大可能错过最优模型
    save_interval = 100
    # 实验名称。用于日志和模型保存的标识
    experiment_name = "unitree_go2arm_rough"
    
    policy = RslRlPpoActorCriticCfg(
        # 初始动作噪声标准差。值越大探索性越强但学习不稳定，值越小容易陷入局部最优
        init_noise_std=1.0,
        # Actor网络是否进行观测归一化。True可加快收敛但增加计算量
        actor_obs_normalization=True,   # mark：论文要求的
        # Critic网络是否进行观测归一化。True提高价值估计稳定性
        critic_obs_normalization=True,  # mark：论文要求的
        # Actor隐层维度。值越大模型容量越强但易过拟合，值越小训练快但表达能力弱
        actor_hidden_dims=[512, 256, 128],
        # Critic隐层维度。影响价值函数拟合质量
        critic_hidden_dims=[512, 256, 128],
        # 激活函数。elu比relu更平滑，梯度流更好
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        # 价值损失系数。值越大越重视价值学习，防止策略发散但可能影响探索
        value_loss_coef=1.0,    # mark：论文要求的
        # 是否使用裁剪价值损失。True防止价值函数过度更新导致训练不稳定
        use_clipped_value_loss=True,    # mark：论文要求的
        # PPO裁剪参数。值越小策略更新越保守，训练更稳定但收敛慢；值越大更新激进但易不稳定
        clip_param=0.2,          # mark：论文要求的
        # 熵系数。值越大探索越强，值越小利用越强；平衡探索-利用的关键参数
        entropy_coef=0.002,      # mark：论文要求的
        # 每批数据的学习轮数。值越大单批数据利用越充分但易过拟合，值越小训练快但样本利用率低
        num_learning_epochs=8,  # mark：论文要求的
        # 小批次数量。值越大梯度估计越稳定，值越小内存占用越少
        num_mini_batches=4,     # mark：论文要求的
        # 学习率。值越大收敛快但易震荡，值越小收敛慢但稳定
        learning_rate=3.0e-4,   # mark：论文要求的
        # 学习率调度策略。adaptive根据KL散度自动调整，fixed保持不变
        schedule="adaptive",
        # 折扣因子。值越大看得越远但可能目标模糊，值越小只关注近期奖励
        gamma=0.99,        # mark：论文要求的
        # GAE折扣因子。值越大优势估计方差大，值越小偏差大
        lam=0.95,          # mark：论文要求的
        # 目标KL散度。adaptive模式下，KL超过此值则降低学习率以保持稳定
        desired_kl=0.01,
        # 梯度裁剪。防止梯度爆炸，值越小约束越强
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        super().__post_init__()
        # Enable multi-critic heads to match reward_group_terms.
        self.policy.critic_names = ["loco", "mani", "contact"]

@configclass
class UnitreeGo2ArmFlatPPORunnerCfg(UnitreeGo2ArmRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "unitree_go2arm_flat"
