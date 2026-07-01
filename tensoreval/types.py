"""Core data structures for TensorEval SDK.

All shared data structures live here. Enums are in enums.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tensoreval.enums import GraderType, Difficulty


@dataclass(frozen=True)
class Rubric:
    """A single evaluation criterion."""

    name: str
    criteria: str
    weight: float = 1.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Rubric":
        return cls(
            name=d.get("name", "unnamed"),
            criteria=d.get("criteria", d.get("rubric", "")),
            weight=float(d.get("weight", 1.0)),
        )


@dataclass
class Score:
    """Result of grading a single rubric."""

    value: float
    """Score between 0.0 and 1.0."""

    explanation: str = ""
    """Why this score was given."""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sample:
    """A single evaluation task."""

    input: str
    target: str = ""
    id: str = ""
    rubrics: list[Rubric] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = f"q_{hash(self.input) % 10000:04d}"


@dataclass
class Run:
    """Result of evaluating a single sample."""

    sample_id: str
    query: str
    answer: str
    response: str
    reward: float
    scores: dict[str, Score] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""

    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    workers: int = 4
    agent_port: int | None = None
    mcp_port: int | None = None
    system_prompt: str | None = None
    timeout: float = 60.0
    pass_threshold: float = 0.8


@dataclass
class Summary:
    """Aggregate results from an evaluation run."""

    model: str
    num_runs: int
    avg_reward: float
    pass_rate: float
    pass_count: int
    fail_count: int
    total_latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "num_runs": self.num_runs,
            "avg_reward": round(self.avg_reward, 4),
            "pass_rate": round(self.pass_rate, 4),
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
        }
