# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class RewardGroupManagerBasedRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv that adds per-group rewards into extras."""

    def step(self, actions):
        obs, rewards, terminated, time_outs, extras = super().step(actions)
        extras["reward_groups"] = self._compute_reward_groups()
        return obs, rewards, terminated, time_outs, extras

    def _compute_reward_groups(self) -> dict[str, torch.Tensor]:
        term_names = self.reward_manager.active_terms
        step_rewards = self.reward_manager._step_reward
        dt = self.step_dt

        group_terms = getattr(self.cfg, "reward_group_terms", None)
        if group_terms is None:
            group_terms = self._default_reward_group_terms(term_names)
        else:
            group_terms = dict(group_terms)

        for name in ("loco", "mani", "contact"):
            group_terms.setdefault(name, [])

        term_indices = {name: idx for idx, name in enumerate(term_names)}
        reward_groups = {}
        for group, terms in group_terms.items():
            if not terms:
                reward_groups[group] = torch.zeros(self.num_envs, device=self.device)
                continue
            missing = [t for t in terms if t not in term_indices]
            if missing:
                raise ValueError(f"Unknown reward terms for group '{group}': {missing}")
            idx = torch.as_tensor([term_indices[t] for t in terms], device=self.device)
            reward_groups[group] = step_rewards.index_select(1, idx).sum(dim=1) * dt

        if getattr(self.cfg, "reward_group_strict", False):
            covered = {t for terms in group_terms.values() for t in terms}
            missing_terms = [t for t in term_names if t not in covered]
            if missing_terms:
                raise ValueError(f"Unassigned reward terms: {missing_terms}")

        return reward_groups

    @staticmethod
    def _default_reward_group_terms(term_names: list[str]) -> dict[str, list[str]]:
        group_terms = {"loco": [], "mani": [], "contact": []}
        for name in term_names:
            lname = name.lower()
            if any(key in lname for key in ("contact", "stumble", "slide", "impact", "collision")):
                group_terms["contact"].append(name)
            elif any(key in lname for key in ("ee_", "end_effector", "manip", "arm", "gripper", "wrist", "hand")):
                group_terms["mani"].append(name)
            else:
                group_terms["loco"].append(name)
        return group_terms
