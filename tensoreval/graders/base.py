"""Base Grader class for TensorEval."""

from typing import Any, Callable
from tensoreval.enums import GraderType


class Grader:
    """Base class for all graders.

    A grader takes a model response and returns a score between 0.0 and 1.0.
    """

    def __init__(self, grader_type: GraderType = GraderType.CUSTOM):
        self.grader_type = grader_type

    async def score(self, state: dict, **kwargs) -> float:
        """Score a single response. Override in subclasses."""
        raise NotImplementedError

    async def score_group(self, states: list[dict], **kwargs) -> list[float]:
        """Score a group of responses (for GRPO). Override in subclasses."""
        return [await self.score(s, **kwargs) for s in states]
