"""Base Grader class for TensorEval.

All graders implement the same interface:
- `score(state) -> float` — score a single response
- `score_group(states) -> list[float]` — score a group when a custom grader needs batching
"""

from __future__ import annotations

from typing import Any

from tensoreval.enums import GraderType


class Grader:
    """Base class for all graders.

    A grader takes a model response and returns a score between 0.0 and 1.0.

    Subclasses must override `score()`. Optionally override `score_group()`
    when batch scoring is more efficient.
    """

    def __init__(self, grader_type: GraderType = GraderType.CUSTOM):
        self.grader_type = grader_type

    async def score(self, state: dict[str, Any], **kwargs) -> float:
        """Score a single response. Override in subclasses.

        Args:
            state: Dict with keys:
                - query: str — the original question
                - answer: str — reference answer
                - completion: list[dict] — agent messages
                - info: dict — metadata including rubrics

        Returns:
            Score between 0.0 and 1.0.
        """
        raise NotImplementedError("Subclasses must implement score()")

    async def score_group(self, states: list[dict[str, Any]], **kwargs) -> list[float]:
        """Score a group of responses.

        Default: call score() on each state individually.
        Override for relative ranking.
        """
        return [await self.score(s, **kwargs) for s in states]
