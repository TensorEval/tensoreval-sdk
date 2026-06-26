"""Core types for TensorEval SDK.

Adapted from PrimeIntellect Verifiers (MIT License).
Provider-agnostic message types, State, RolloutInput/Output.
"""

import json
import time
import uuid
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Literal,
    TypeAlias,
    TypeVar,
    overload,
    cast,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
)


# ---------------------------------------------------------------------------
# Base model with dict-like access
# ---------------------------------------------------------------------------

class CustomBaseModel(BaseModel):
    """Allow extras and dict-like attribute access."""

    model_config = ConfigDict(extra="allow")

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return hasattr(self, key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self.model_dump() == dict(other)
        return super().__eq__(other)


# ---------------------------------------------------------------------------
# Message types (provider-agnostic)
# ---------------------------------------------------------------------------

class TextContentPart(CustomBaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageUrlSource(CustomBaseModel):
    url: str


class ImageUrlContentPart(CustomBaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrlSource


class InputAudioSource(CustomBaseModel):
    data: str
    format: str


class InputAudioContentPart(CustomBaseModel):
    type: Literal["input_audio"] = "input_audio"
    input_audio: InputAudioSource


class GenericContentPart(CustomBaseModel):
    type: str


ContentPart: TypeAlias = (
    TextContentPart
    | ImageUrlContentPart
    | InputAudioContentPart
    | GenericContentPart
    | dict[str, Any]
)
MessageContent: TypeAlias = str | list[ContentPart]


class TextMessage(CustomBaseModel):
    role: Literal["text"] = "text"
    content: str


class SystemMessage(CustomBaseModel):
    role: Literal["system"] = "system"
    content: MessageContent

    @classmethod
    def from_path(cls, path: str | Path) -> "SystemMessage":
        return cls(content=Path(path).read_text(encoding="utf-8"))


class UserMessage(CustomBaseModel):
    role: Literal["user"] = "user"
    content: MessageContent


class ToolCall(CustomBaseModel):
    id: str
    name: str
    arguments: str


class AssistantMessage(CustomBaseModel):
    role: Literal["assistant"] = "assistant"
    content: MessageContent | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolMessage(CustomBaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: MessageContent


Message: TypeAlias = (
    SystemMessage | UserMessage | AssistantMessage | ToolMessage | TextMessage
)
Messages = list[Message]


# ---------------------------------------------------------------------------
# Tool, Usage, Response
# ---------------------------------------------------------------------------

class Tool(CustomBaseModel):
    name: str
    description: str
    parameters: dict[str, object]
    strict: bool | None = None


ToolLike: TypeAlias = str | Tool | Callable[..., object]


class Usage(CustomBaseModel):
    prompt_tokens: int = 0
    reasoning_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ResponseTokens(CustomBaseModel):
    prompt_ids: list[int] = Field(default_factory=list)
    prompt_mask: list[int] = Field(default_factory=list)
    completion_ids: list[int] = Field(default_factory=list)
    completion_mask: list[int] = Field(default_factory=list)
    completion_logprobs: list[float] = Field(default_factory=list)


FinishReason = Literal["stop", "length", "tool_calls"] | None


class ResponseMessage(AssistantMessage):
    finish_reason: FinishReason = None
    is_truncated: bool | None = None
    tokens: ResponseTokens | None = None


class Response(CustomBaseModel):
    id: str = ""
    created: int = 0
    model: str = ""
    usage: Usage | None = None
    message: ResponseMessage


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

Info = dict[str, Any]
SamplingArgs = dict[str, Any]
IndividualRewardFunc = Callable[..., float | Awaitable[float]]
GroupRewardFunc = Callable[..., list[float] | Awaitable[list[float]]]
RewardFunc = IndividualRewardFunc | GroupRewardFunc


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class TimeSpan(CustomBaseModel):
    """A timed span."""
    start: float = 0.0
    end: float = 0.0

    @computed_field
    @property
    def duration(self) -> float:
        if self.end <= 0.0:
            return 0.0
        return self.end - self.start


class TimeSpans(CustomBaseModel):
    """A list of TimeSpans."""
    spans: list[TimeSpan] = Field(default_factory=list)

    @computed_field
    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.spans)


class RolloutTiming(CustomBaseModel):
    """Rollout-level timing. All values in seconds (Unix timestamps)."""
    start_time: float = Field(default_factory=time.time)
    setup: TimeSpan = Field(default_factory=TimeSpan)
    generation: TimeSpan = Field(default_factory=TimeSpan)
    scoring: TimeSpan = Field(default_factory=TimeSpan)
    model: TimeSpans = Field(default_factory=TimeSpans)
    env: TimeSpans = Field(default_factory=TimeSpans)


# ---------------------------------------------------------------------------
# Trajectory step
# ---------------------------------------------------------------------------

class TrajectoryStep(CustomBaseModel):
    prompt: Messages = Field(default_factory=list)
    completion: Messages = Field(default_factory=list)
    response: Response | None = None
    tokens: ResponseTokens | None = None
    reward: float | None = None
    advantage: float | None = None
    is_truncated: bool = False
    trajectory_id: str = ""
    extras: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# TokenUsage (defined before RolloutOutput to avoid forward reference)
# ---------------------------------------------------------------------------

class TokenUsage(dict):
    """Token usage statistics."""
    input_tokens: float = 0.0
    output_tokens: float = 0.0


# ---------------------------------------------------------------------------
# RolloutInput / RolloutOutput
# ---------------------------------------------------------------------------

class RolloutInput(CustomBaseModel):
    """Input for a single rollout."""
    prompt: Messages
    example_id: int = 0
    answer: str = ""
    info: Info = Field(default_factory=dict)


class RolloutOutput(dict):
    """Serialized output from a rollout."""

    # Required fields
    example_id: int
    prompt: Messages | None
    completion: Messages | None
    reward: float
    timing: RolloutTiming
    is_completed: bool
    is_truncated: bool
    metrics: dict[str, float]
    # Optional fields
    answer: str
    info: Info
    trajectory: list[TrajectoryStep]
    token_usage: TokenUsage


# ---------------------------------------------------------------------------
# GenerateOutputs
# ---------------------------------------------------------------------------

class GenerateMetadata(dict):
    env_id: str
    model: str
    num_examples: int
    rollouts_per_example: int
    avg_reward: float
    avg_metrics: dict[str, float]


class GenerateOutputs(dict):
    outputs: list[RolloutOutput]
    metadata: GenerateMetadata


# ---------------------------------------------------------------------------
# RolloutScore
# ---------------------------------------------------------------------------

class RolloutScore(dict):
    metrics: dict[str, float]
    reward: float


# ---------------------------------------------------------------------------
# State (dict subclass with input forwarding)
# ---------------------------------------------------------------------------

_MISSING = object()
_DefaultValue = TypeVar("_DefaultValue")


class State(dict):
    """Mutable rollout state with input-field forwarding."""

    INPUT_FIELDS = ["prompt", "answer", "info", "example_id"]
    INTERNAL_KEYS = {"is_completed", "stop_condition", "is_truncated", "error"}

    def __getitem__(self, key: str) -> Any:
        if key in self.INPUT_FIELDS and "input" in self:
            input_obj = super().__getitem__("input")
            if isinstance(input_obj, dict) and key in input_obj:
                return input_obj[key]
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self.INPUT_FIELDS and "input" in self:
            input_obj = super().__getitem__("input")
            if isinstance(input_obj, dict) and key in input_obj:
                input_obj[key] = value
                return
        super().__setitem__(key, value)

    def _set_internal(self, key: str, value: Any) -> None:
        if key not in self.INTERNAL_KEYS:
            raise KeyError(f"{key!r} is not a framework-managed state key.")
        super().__setitem__(key, value)

    def stop(self, condition: str = "state_done") -> None:
        super().__setitem__("done", True)
        self._set_internal("is_completed", True)
        self._set_internal("stop_condition", condition)

    def add_step_reward(self, reward: float | int | None) -> None:
        if reward is None:
            return
        trajectory = self.get("trajectory", [])
        if trajectory:
            step = trajectory[-1]
            current = step.get("reward", 0.0) or 0.0
            step["reward"] = float(current) + float(reward)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

StartCallback = Callable[[list[RolloutInput], Any], None]
ProgressCallback = Callable[[list[RolloutOutput], list[RolloutOutput], GenerateMetadata], None]
LogCallback = Callable[[str], None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flatten_task_input(input_data: dict) -> dict:
    """Flatten nested input for state access."""
    result = dict(input_data)
    if "info" in result and isinstance(result["info"], str):
        try:
            result["info"] = json.loads(result["info"])
        except (json.JSONDecodeError, TypeError):
            pass
    return result


def state_to_output(state: State, state_columns: list[str] | None = None) -> RolloutOutput:
    """Convert State to RolloutOutput dict."""
    output = RolloutOutput(
        example_id=state.get("example_id", 0),
        prompt=state.get("prompt"),
        completion=state.get("completion"),
        reward=state.get("reward", 0.0) or 0.0,
        timing=state.get("timing", RolloutTiming()),
        is_completed=state.get("is_completed", False),
        is_truncated=state.get("is_truncated", False),
        metrics=state.get("metrics", {}),
        answer=state.get("answer", ""),
        info=state.get("info", {}),
        trajectory=state.get("trajectory", []),
        token_usage=state.get("usage", {}),
    )
    for col in (state_columns or []):
        if col in state:
            output[col] = state[col]
    return output
