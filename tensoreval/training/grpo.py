"""GRPO advantage computation.

Extracted from verl core_algos.py (Apache 2.0 License).
Pure torch/numpy implementations — no Ray dependency.
"""

from typing import Any


def compute_grpo_advantage(
    rewards: list[float],
    group_size: int,
) -> list[float]:
    """Compute GRPO advantages by centering rewards within each group.

    Args:
        rewards: Flat list of rewards (group_size * num_groups items).
        group_size: Number of rollouts per group.

    Returns:
        Flat list of advantages (reward - group_mean).
    """
    advantages = []
    num_groups = len(rewards) // group_size if group_size > 0 else 0
    for g in range(num_groups):
        group_rewards = rewards[g * group_size : (g + 1) * group_size]
        mean_reward = sum(group_rewards) / len(group_rewards)
        for r in group_rewards:
            advantages.append(r - mean_reward)
    # Handle remaining rewards
    remaining = len(rewards) - num_groups * group_size
    if remaining > 0:
        group_rewards = rewards[num_groups * group_size:]
        mean_reward = sum(group_rewards) / len(group_rewards)
        for r in group_rewards:
            advantages.append(r - mean_reward)
    return advantages


def compute_kl_penalty(
    logprobs: list[float],
    ref_logprobs: list[float],
    kl_coef: float = 0.1,
) -> list[float]:
    """Compute KL penalty per token.

    Args:
        logprobs: Current policy log-probabilities.
        ref_logprobs: Reference policy log-probabilities.
        kl_coef: KL penalty coefficient.

    Returns:
        Per-token KL penalties.
    """
    return [kl_coef * (lp - rlp) for lp, rlp in zip(logprobs, ref_logprobs)]


def policy_gradient_loss(
    logprobs: list[float],
    old_logprobs: list[float],
    advantages: list[float],
    clip_range: float = 0.2,
) -> float:
    """Compute clipped policy gradient loss (PPO-style).

    Args:
        logprobs: Current policy log-probabilities.
        old_logprobs: Old policy log-probabilities.
        advantages: Advantage estimates.
        clip_range: PPO clipping range.

    Returns:
        Scalar loss value.
    """
    import math
    losses = []
    for lp, olp, adv in zip(logprobs, old_logprobs, advantages):
        ratio = math.exp(lp - olp)
        clipped_ratio = max(min(ratio, 1 + clip_range), 1 - clip_range)
        loss = -min(ratio * adv, clipped_ratio * adv)
        losses.append(loss)
    return sum(losses) / max(len(losses), 1)
