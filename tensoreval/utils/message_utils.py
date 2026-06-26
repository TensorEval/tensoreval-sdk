"""Message utilities for normalization and serialization.

Adapted from PrimeIntellect Verifiers (MIT License).
"""

from typing import Any, Sequence

from tensoreval.core.types import (
    AssistantMessage,
    Message,
    Messages,
    SystemMessage,
    TextMessage,
    ToolMessage,
    UserMessage,
)


def from_raw_message(message: dict) -> Message:
    """Convert a raw dict to the appropriate Pydantic message type."""
    role = message.get("role", "user")
    if role == "text":
        return TextMessage.model_validate(message)
    elif role == "system":
        return SystemMessage.model_validate(message)
    elif role == "user":
        return UserMessage.model_validate(message)
    elif role == "assistant":
        return AssistantMessage.model_validate(message)
    elif role == "tool":
        return ToolMessage.model_validate(message)
    else:
        raise ValueError(f"Unknown role: {role}")


def normalize_messages(value: str | Sequence, *, field_name: str = "messages") -> Messages:
    """Normalize raw/string message inputs into typed Message objects."""
    if isinstance(value, str):
        return [TextMessage(content=value)]
    normalized: Messages = []
    for message in value:
        if isinstance(message, dict):
            normalized.append(from_raw_message(dict(message)))
        elif hasattr(message, "role") and hasattr(message, "content"):
            normalized.append(message)
        else:
            raise TypeError(f"Invalid {field_name} item: {type(message).__name__}")
    return normalized


def concat_messages(messages_list: list[Messages]) -> Messages:
    """Concatenate multiple Messages lists into one."""
    result = []
    for messages in messages_list:
        result.extend(messages)
    return result


def maybe_normalize_messages(value: Messages | str, *, field_name: str = "messages") -> Messages:
    """Normalize messages only if needed."""
    if isinstance(value, str):
        return normalize_messages(value, field_name=field_name)
    if isinstance(value, list) and all(isinstance(m, Message) for m in value):
        return value
    return normalize_messages(value, field_name=field_name)
