"""Enums for TensorEval."""

from enum import Enum


class GraderType(str, Enum):
    """Grader/scorer types."""

    AGENT = "agent"
    """LLM-as-judge scoring. Uses another model to judge responses."""

    CUSTOM = "custom"
    """User-defined scoring function."""
