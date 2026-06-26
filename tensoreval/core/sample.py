"""Sample data model for TensorEval evaluation tasks.

Adapted from Inspect AI (MIT License, UK AISI).
"""

from typing import Any

from pydantic import BaseModel, Field


class Sample(BaseModel):
    """A single evaluation sample — one task for the agent to complete."""

    input: str
    """The input prompt to be submitted to the model."""

    target: str | list[str] = ""
    """Ideal target output. May be a literal value or narrative text for LLM grading."""

    id: int | str | None = None
    """Unique identifier for this sample."""

    metadata: dict[str, Any] | None = None
    """Arbitrary metadata associated with the sample."""

    rubrics: list[dict[str, Any]] = Field(default_factory=list)
    """Rubric definitions for scoring. Each has name, rubric (criteria), weight."""

    reference_answer: str | None = None
    """Verified correct answer, if available."""

    category: str = ""
    """Category label for this sample (e.g. 'statistical', 'code_generation')."""

    difficulty: str = "medium"
    """Difficulty level: 'easy', 'medium', or 'hard'."""

    context: dict[str, Any] | None = None
    """Additional context (file attachments, expected behavior, etc.)."""
