"""Single-turn environment — model responds once.

Supports both OpenAI-compatible and Anthropic-compatible APIs.
Auto-detects API type from base_url.
"""

import time
from typing import Any

from tensoreval.core.types import (
    AssistantMessage,
    Messages,
    RolloutInput,
    SamplingArgs,
    State,
    SystemMessage,
    TrajectoryStep,
    UserMessage,
)
from tensoreval.envs.environment import Environment


def _detect_api_type(base_url: str | None) -> str:
    """Detect whether to use OpenAI or Anthropic client."""
    if base_url and "anthropic" in base_url.lower():
        return "anthropic"
    return "openai"


async def _call_openai(
    messages: list[dict],
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Call an OpenAI-compatible API."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


async def _call_anthropic(
    messages: list[dict],
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Call an Anthropic-compatible API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required. Install with: pip install anthropic")

    client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)

    # Extract system message
    system_msg = ""
    user_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        if role == "system":
            system_msg = content
        else:
            user_messages.append({"role": role, "content": content})

    if not user_messages:
        user_messages = [{"role": "user", "content": "Hello"}]

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": user_messages,
    }
    if system_msg:
        kwargs["system"] = system_msg

    response = await client.messages.create(**kwargs)
    return response.content[0].text if response.content else ""


def _messages_to_dicts(prompt: Messages) -> list[dict]:
    """Convert Messages to list of dicts for API calls."""
    messages = []
    for m in prompt:
        if hasattr(m, 'model_dump'):
            messages.append(m.model_dump())
        elif isinstance(m, dict):
            messages.append(m)
        else:
            messages.append({
                "role": getattr(m, 'role', 'user'),
                "content": getattr(m, 'content', str(m))
            })
    return messages


class SingleTurnEnv(Environment):
    """Environment for single-turn tasks.

    Model responds once, then evaluation ends.
    Supports both OpenAI-compatible and Anthropic-compatible APIs.

    Usage:
        env = SingleTurnEnv(
            rubric=my_rubric,
            system_prompt="Solve the math problem.",
        )
        result = await env.rollout(input, model="mimo-v2.5-pro", ...)
    """

    async def rollout(
        self,
        input: RolloutInput,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """Run a single-turn rollout."""
        state = self._create_state(input)
        state["timing"].generation.start = time.time()

        # Build prompt
        prompt = state.get("prompt", [])
        if isinstance(prompt, str):
            prompt = [UserMessage(content=prompt)]
        if self.system_prompt:
            prompt = [SystemMessage(content=self.system_prompt)] + prompt

        # Convert to dict messages
        messages = _messages_to_dicts(prompt)

        # Merge sampling args
        merged_args = {**self.sampling_args, **(sampling_args or {})}
        max_tokens = merged_args.get("max_tokens", 1024)
        temperature = merged_args.get("temperature", 0.7)

        # Detect API type and call
        api_type = _detect_api_type(base_url)
        resolved_key = api_key or "dummy"
        resolved_url = base_url or "https://api.openai.com/v1"

        try:
            if api_type == "anthropic":
                content = await _call_anthropic(messages, model, resolved_key, resolved_url, max_tokens, temperature)
            else:
                content = await _call_openai(messages, model, resolved_key, resolved_url, max_tokens, temperature)

            completion = [AssistantMessage(content=content)]
            state["completion"] = completion
            state["trajectory"] = [TrajectoryStep(
                prompt=prompt,
                completion=completion,
                trajectory_id=state["trajectory_id"],
            )]
        except Exception as e:
            state["error"] = e
            state["is_completed"] = True

        state["timing"].generation.end = time.time()
        state["is_completed"] = True
        return state
