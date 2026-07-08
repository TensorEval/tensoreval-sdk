"""Shared data types for the TensorEval SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Rubric:
    id: str
    description: str
    weight: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rubric":
        return cls(
            id=str(data.get("id") or data.get("name") or "rubric"),
            description=str(data.get("description") or data.get("criteria") or data.get("rubric") or ""),
            weight=float(data.get("weight", 1.0)),
        )


@dataclass
class Sample:
    query: str
    id: str = ""
    reference_answer: str = ""
    rubrics: list[Rubric] = field(default_factory=list)
    setup_script: str = ""
    verify_script: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"case_{abs(hash(self.query)) % 100000}"
