"""Grader for TensorEval evaluation.

Combines Rubric-based scoring with optional RULER zero-config fallback.
"""

from typing import Any, Callable

from tensoreval.core.types import RewardFunc, State
from tensoreval.rubrics.rubric import Rubric, _maybe_await
from tensoreval.rubrics.judge_rubric import JudgeRubric
from tensoreval.rubrics.ruler import ruler, DEFAULT_RUBRIC


class Grader(Rubric):
    """High-level grading interface for TensorEval.

    Usage:
        # With custom reward functions
        grader = Grader(funcs=[my_reward_func], weights=[1.0])

        # With LLM judge
        grader = Grader(model="mimo-v2.5-pro", judge=True)

        # With RULER (zero-config)
        grader = Grader.ruler(model="mimo-v2.5-pro")
    """

    def __init__(
        self,
        funcs: list[RewardFunc] | None = None,
        weights: list[float] | None = None,
        model: str = "mimo-v2.5-pro",
        verified: bool = False,
        judge: bool = False,
        judge_prompt: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        """
        Args:
            funcs: Reward functions to use.
            weights: Weights for each reward function.
            model: Model to use for LLM judging.
            verified: If True, use tool-verified evaluation (slower but more accurate).
            judge: If True, use LLM-as-judge for scoring.
            judge_prompt: Custom prompt for the judge.
            api_key: API key for the model.
            base_url: Base URL for the model API.
        """
        super().__init__(funcs=funcs, weights=weights, **kwargs)
        self.model = model
        self.verified = verified
        self.judge_enabled = judge
        self.api_key = api_key
        self.base_url = base_url

        if judge:
            judge_client = None
            if api_key or base_url:
                try:
                    from openai import AsyncOpenAI
                    judge_client = AsyncOpenAI(
                        api_key=api_key or "dummy",
                        base_url=base_url or "https://api.openai.com/v1",
                    )
                except ImportError:
                    pass
            self._judge_rubric = JudgeRubric(
                judge_client=judge_client,
                judge_model=model,
                judge_prompt=judge_prompt or JudgeRubric.__init__.__defaults__[3] if len(JudgeRubric.__init__.__defaults__) > 3 else "",
            )
            self.add_reward_func(self._judge_reward, weight=0.5)

    async def _judge_reward(self, state: State, **kwargs) -> float:
        """Reward function using LLM-as-judge."""
        prompt = state.get("prompt", [])
        completion = state.get("completion", [])
        answer = state.get("answer", "")
        try:
            result = await self._judge_rubric.judge(prompt, completion, answer, state)
            return 1.0 if "yes" in result.lower() else 0.0
        except Exception:
            return 0.0

    @classmethod
    def ruler(
        cls,
        model: str = "mimo-v2.5-pro",
        rubric: str = DEFAULT_RUBRIC,
        **kwargs,
    ) -> "Grader":
        """Create a Grader using RULER zero-config reward.

        RULER uses an LLM-as-judge to rank trajectories relatively.
        No custom reward functions needed.
        """
        grader = cls(model=model, **kwargs)
        grader._ruler_rubric = rubric
        return grader

    @classmethod
    def from_rubrics(cls, *rubrics: Rubric) -> "Grader":
        """Create a Grader from multiple Rubric objects."""
        from tensoreval.rubrics.rubric_group import RubricGroup
        grader = cls()
        grader = RubricGroup(rubrics=list(rubrics))
        return grader
