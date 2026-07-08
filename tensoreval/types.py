"""Shared data types for the TensorEval SDK."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Rubric:
    """A single evaluation criterion.

    Attributes:
        id: Short identifier (e.g. ``"correctness"``).
        description: Human-readable description of what a correct response looks like.
        weight: Relative importance — normalized across all rubrics in a sample.
    """

    id: str
    description: str
    weight: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rubric":
        """Build a Rubric from a dict, accepting common key aliases."""
        return cls(
            id=str(data.get("id") or data.get("name") or "rubric"),
            description=str(data.get("description") or data.get("criteria") or data.get("rubric") or ""),
            weight=float(data.get("weight", 1.0)),
        )


@dataclass
class Sample:
    """A single evaluation case — one query plus its rubrics and metadata.

    Attributes:
        query: The input string sent to the agent.
        id: Stable identifier. Auto-generated from ``md5(query)`` if not provided.
        reference_answer: Ground-truth answer (empty if none).
        rubrics: Criteria the grader scores against.
        setup_script: Optional shell script to run before evaluation.
        verify_script: Optional shell script to run after evaluation.
        metadata: Arbitrary per-sample metadata (attachments, category, etc.).
    """

    query: str
    id: str = ""
    reference_answer: str = ""
    rubrics: list[Rubric] = field(default_factory=list)
    setup_script: str = ""
    verify_script: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"case_{hashlib.md5(self.query.encode()).hexdigest()[:8]}"
