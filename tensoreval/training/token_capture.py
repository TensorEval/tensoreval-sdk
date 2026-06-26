"""Token-level logprobs capture for RL training data.

Based on ART's messages_and_choices pattern and HUD's Sample pattern.
Captures per-token logprobs during evaluation for use in GRPO training.
"""

from typing import Any
from pydantic import BaseModel, Field


class TokenSample(BaseModel):
    """Token-level training data from a single model turn.

    Captures the prompt tokens, output tokens, and per-token logprobs
    needed for RL training (GRPO, PPO, etc.).
    """

    prompt_token_ids: list[int] = Field(default_factory=list)
    """Token IDs for the prompt (input to the model)."""

    output_token_ids: list[int] = Field(default_factory=list)
    """Token IDs for the model's output (completion)."""

    output_logprobs: list[float] = Field(default_factory=list)
    """Per-token log-probabilities for the output tokens."""

    prompt_chunks: list[dict[str, Any]] | None = None
    """Optional multimodal prompt chunks."""

    def __len__(self) -> int:
        return len(self.output_token_ids)


class TrajectoryData(BaseModel):
    """Full trajectory data for RL training.

    Contains all token-level samples from a single evaluation run,
    along with the reward and metadata needed for training.
    """

    trajectory_id: str
    """Unique identifier for this trajectory."""

    task_id: str = ""
    """Task/Query ID this trajectory corresponds to."""

    samples: list[TokenSample] = Field(default_factory=list)
    """Token-level samples from each model turn."""

    reward: float = 0.0
    """Final reward for this trajectory."""

    metrics: dict[str, float] = Field(default_factory=dict)
    """Scoring metrics."""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    """Full conversation messages for context."""

    def total_tokens(self) -> int:
        """Total output tokens across all samples."""
        return sum(len(s) for s in self.samples)

    def to_sft_format(self) -> dict[str, Any]:
        """Convert to SFT training format."""
        return {
            "messages": self.messages,
            "reward": self.reward,
        }

    def to_dpo_format(self, rejected_messages: list[dict] | None = None) -> dict[str, Any]:
        """Convert to DPO training format."""
        return {
            "prompt": self.messages[0]["content"] if self.messages else "",
            "chosen": self.messages[-1]["content"] if self.messages else "",
            "rejected": rejected_messages[-1]["content"] if rejected_messages else "",
        }


def capture_logprobs_from_response(response: Any) -> TokenSample | None:
    """Extract token-level logprobs from an API response.

    Supports both OpenAI and Anthropic response formats.
    """
    # OpenAI format
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "logprobs") and choice.logprobs:
            token_ids = []
            logprobs = []
            for token_logprob in choice.logprobs.content or []:
                if hasattr(token_logprob, "token"):
                    # Try to get token ID
                    if hasattr(token_logprob, "bytes"):
                        token_ids.append(int.from_bytes(token_logprob.bytes, "big") if token_logprob.bytes else 0)
                    logprobs.append(token_logprob.logprob if hasattr(token_logprob, "logprob") else 0.0)
            return TokenSample(
                output_token_ids=token_ids,
                output_logprobs=logprobs,
            )

    # Anthropic format
    if hasattr(response, "usage") and hasattr(response.usage, "output_tokens"):
        # Anthropic doesn't expose per-token logprobs in the standard API
        # Return token count only
        return TokenSample(
            output_token_ids=list(range(response.usage.output_tokens)),
            output_logprobs=[0.0] * response.usage.output_tokens,
        )

    return None


def build_training_data(
    trajectories: list[TrajectoryData],
    group_size: int = 1,
) -> list[dict[str, Any]]:
    """Build training data from trajectories.

    Groups trajectories and computes GRPO advantages.

    Args:
        trajectories: List of trajectory data from evaluation.
        group_size: Number of rollouts per group for GRPO.

    Returns:
        List of training data dicts with advantages computed.
    """
    from tensoreval.training.grpo import compute_grpo_advantage

    if group_size <= 1:
        # No grouping — use raw rewards
        return [
            {
                "messages": t.messages,
                "reward": t.reward,
                "advantage": t.reward,
                "logprobs": [s.output_logprobs for s in t.samples],
                "token_ids": [s.output_token_ids for s in t.samples],
            }
            for t in trajectories
        ]

    # Group trajectories and compute advantages
    training_data = []
    for i in range(0, len(trajectories), group_size):
        group = trajectories[i : i + group_size]
        rewards = [t.reward for t in group]
        advantages = compute_grpo_advantage(rewards, group_size=len(group))

        for traj, advantage in zip(group, advantages):
            training_data.append({
                "messages": traj.messages,
                "reward": traj.reward,
                "advantage": advantage,
                "logprobs": [s.output_logprobs for s in traj.samples],
                "token_ids": [s.output_token_ids for s in traj.samples],
            })

    return training_data
