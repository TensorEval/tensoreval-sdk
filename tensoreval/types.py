"""Core data structures for TensorEval SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Rubric:
    """A single evaluation criterion.

    Attributes:
        name: Short identifier for the rubric.
        criteria: Description of what a correct response looks like.
        weight: Relative importance (normalized across all rubrics).
    """

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
class Sample:
    """A single evaluation task.

    Attributes:
        input: The user query or task description.
        target: Reference answer (optional — graders can work without it).
        id: Unique identifier (auto-generated if empty).
        rubrics: Evaluation criteria. Defaults to a single "correctness" rubric.
        metadata: Arbitrary metadata (attachments, category, etc.).
    """

    input: str
    target: str = ""
    id: str = ""
    rubrics: list[Rubric] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"q_{hash(self.input) % 10000:04d}"


@dataclass
class AgentResult:
    """Structured return value from an agent callable.

    The agent receives a query string and returns an ``AgentResult``
    containing the final response and an optional tool trace — the
    list of tool calls the agent made while processing the query.

    If the agent returns a plain ``str``, it is treated as
    ``AgentResult(response=str)``.

    Attributes:
        response: The agent's final text response.
        tool_trace: Optional list of tool call records. Each entry is a
            dict with keys ``tool``, ``args``, and ``result``.
    """

    response: str = ""
    tool_trace: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def coerce(cls, value: Any) -> "AgentResult":
        """Normalize a return value into an AgentResult.

        Accepts:
            - AgentResult → returned as-is
            - str → AgentResult(response=str)
            - dict with 'response' key → AgentResult(**dict)
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(response=value)
        if isinstance(value, dict) and "response" in value:
            return cls(
                response=str(value.get("response", "")),
                tool_trace=list(value.get("tool_trace", [])),
            )
        return cls(response=str(value) if value is not None else "")


@dataclass
class Run:
    """Result of evaluating a single sample.

    Attributes:
        sample_id: ID of the evaluated sample.
        query: The original query.
        answer: Reference answer (if any).
        response: Agent's response.
        reward: Score from the grader (0.0–1.0).
        latency_ms: Time taken in milliseconds.
        error: Error message if the run failed.
        metadata: Additional data (grader_result, tool_trace, etc.).
    """

    sample_id: str
    query: str
    answer: str
    response: str
    reward: float
    latency_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    """Configuration for an evaluation run.

    Attributes:
        model: Model ID for the grader.
        api_key: API key for the model.
        base_url: OpenAI-compatible base URL.
        workers: Number of concurrent evaluation workers.
        mcp_servers: MCP server configs for grader verification.
        system_prompt: System prompt for the agent.
        timeout: Request timeout in seconds.
        pass_threshold: Score threshold for pass/fail.
    """

    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    workers: int = 4
    mcp_servers: list[Any] = field(default_factory=list)
    system_prompt: str | None = None
    timeout: float = 60.0
    pass_threshold: float = 0.8


@dataclass
class Summary:
    """Aggregate results from an evaluation run.

    Attributes:
        model: Model used for grading.
        num_runs: Number of samples evaluated.
        avg_reward: Average score across all runs.
        pass_rate: Fraction of runs that passed.
        pass_count: Number of passing runs.
        fail_count: Number of failing runs.
        total_latency_ms: Total latency in milliseconds.
    """

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
