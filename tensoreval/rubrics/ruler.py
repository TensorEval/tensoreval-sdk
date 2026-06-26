"""RULER — Relative Universal LLM-Elicited Rewards.

Ported from OpenPipe ART (Apache 2.0 License).
Zero-config reward via LLM-as-judge for GRPO training.
"""

import json
from textwrap import dedent

from pydantic import BaseModel, Field


class TrajectoryScore(BaseModel):
    """Individual score for a single trajectory."""

    trajectory_id: str = Field(description="The id of the trajectory being scored.")
    explanation: str = Field(description="A short description of the trajectory's performance.")
    score: float = Field(description="A score between 0 and 1.")


class RulerResponse(BaseModel):
    """Response format expected from the LLM judge."""

    scores: list[TrajectoryScore] = Field(description="The scores for each trajectory.")


DEFAULT_RUBRIC = dedent("""\
    - A trajectory that achieves its goal should always get a significantly higher score than a trajectory that does not achieve its goal.
    - A trajectory that achieves its goal more efficiently (eg. by avoiding unproductive detours) should get a higher score than a trajectory that achieves its goal less efficiently.
    - If one trajectory is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
    - You may give some partial credit for a trajectory that makes progress towards its goal but does not complete it.
""")


async def ruler(
    message_lists: list[list[dict]],
    judge_model: str = "mimo-v2.5-pro",
    extra_litellm_params: dict[str, object] | None = None,
    rubric: str = DEFAULT_RUBRIC,
    *,
    debug: bool = False,
) -> list[TrajectoryScore]:
    """Score trajectories using relative ranking by an LLM judge.

    RULER works by:
    1. Extracting common prefixes from trajectories to save tokens
    2. Passing all trajectories to an LLM judge for relative scoring
    3. Returning scores that can be used directly as rewards in GRPO

    Args:
        message_lists: Each item is a list of message dicts for one trajectory.
        judge_model: The model to use for judging.
        extra_litellm_params: Additional parameters for the judge call.
        rubric: The grading rubric.
        debug: If True, print the judge's reasoning.

    Returns:
        A list of TrajectoryScore objects with scores and explanations.
    """
    if not message_lists:
        return []

    # Find common prefix length
    common_prefix_len = 0
    for idx, msg in enumerate(message_lists[0]):
        if all(len(msg_list) > idx and msg_list[idx] == msg for msg_list in message_lists):
            common_prefix_len += 1
        else:
            break

    # Detect identical trajectories
    all_identical = all(len(msg_list) == common_prefix_len for msg_list in message_lists)

    # Build context
    user_text = ""
    if common_prefix_len > 0 and not all_identical:
        common_prefix_messages = message_lists[0][:common_prefix_len]
        user_text += "<context>\n" + json.dumps(common_prefix_messages) + "\n</context>\n\n"

    # Serialize trajectories
    serialized_trajectories: list[str] = []
    if all_identical:
        full_trajectory = message_lists[0]
        serialized_trajectories.append(
            f'<trajectory id="1">\n' + json.dumps(full_trajectory) + "\n</trajectory>"
        )
    else:
        for idx, full_messages in enumerate(message_lists, start=1):
            trimmed_messages = full_messages[common_prefix_len:]
            serialized_trajectories.append(
                f'<trajectory id="{idx}">\n' + json.dumps(trimmed_messages) + "\n</trajectory>"
            )

    user_text += "Trajectories:\n\n" + "\n\n".join(serialized_trajectories)

    judge_prompt = dedent(f"""
        All of the trajectories below have been given the same goal. Your job is to consider each of them and give them a score between 0 and 1. Take into consideration your best judgement of the agent's goal.

        Grading standards:
        {rubric}
    """)

    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": user_text},
    ]

    # Lazy import litellm
    try:
        from litellm import acompletion
    except ImportError:
        raise RuntimeError("litellm package required for RULER. Install with: pip install litellm")

    response = await acompletion(
        model=judge_model,
        messages=messages,
        response_format=RulerResponse,
        caching=False,
        **(extra_litellm_params or {}),
    )

    if len(response.choices) == 0:
        raise ValueError(f"No choices in response: {response}")

    content = response.choices[0].message.content or "{}"
    parsed = RulerResponse.model_validate_json(content)

    if all_identical:
        if len(parsed.scores) != 1:
            raise ValueError(f"Expected 1 score for identical trajectories, but got {len(parsed.scores)}")
        single_score = parsed.scores[0]
        return [single_score.model_copy(update={"trajectory_id": str(i)}) for i in range(1, len(message_lists) + 1)]
    else:
        if len(parsed.scores) != len(message_lists):
            raise ValueError(f"Expected {len(message_lists)} scores, but got {len(parsed.scores)}")
        return parsed.scores
