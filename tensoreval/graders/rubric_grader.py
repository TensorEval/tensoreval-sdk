"""RubricGrader — Simple answer matching only.

For rubric-based LLM scoring, use AgentGrader instead.
"""

from tensoreval.graders.base import Grader
from tensoreval.enums import GraderType


class RubricGrader(Grader):
    """Simple answer match grader.

    Checks if reference answer appears in response. That's it.
    For real rubric scoring, use AgentGrader.
    """

    def __init__(self, rubrics: list[dict] | None = None):
        super().__init__(GraderType.RUBRIC)
        self.rubrics = rubrics or []

    async def score(self, state: dict, **kwargs) -> float:
        """Score: 1.0 if answer in response, else 0.0."""
        completion = state.get("completion", [])
        answer = state.get("answer", "")

        if not completion or not answer:
            return 0.0

        last = completion[-1]
        response = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))

        return 1.0 if answer.lower().strip() in response.lower().strip() else 0.0
