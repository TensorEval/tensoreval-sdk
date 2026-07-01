"""Anthropic API provider.

Handles calls to Anthropic-compatible endpoints (Claude, Mimo, etc.).
"""

from __future__ import annotations

import asyncio
from typing import Any


async def call_anthropic(
    query: str,
    model: str,
    api_key: str,
    base_url: str,
    system_prompt: str | None = None,
    max_tokens: int = 2000,
    timeout: float = 60.0,
) -> str:
    """Call an Anthropic-compatible API.

    Args:
        query: User message content.
        model: Model name.
        api_key: API key.
        base_url: API base URL.
        system_prompt: Optional system message (passed as top-level param, not in messages).
        max_tokens: Max response tokens.
        timeout: Request timeout in seconds.

    Returns:
        Assistant response text.
    """
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": query}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    response = await asyncio.wait_for(
        client.messages.create(**kwargs),
        timeout=timeout,
    )

    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return ""


async def call_anthropic_judge(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    system_prompt: str = "You are an evaluation judge. Output only valid JSON.",
    max_tokens: int = 500,
    timeout: float = 60.0,
) -> str:
    """Call Anthropic-compatible API for judging.

    Same as call_anthropic but with a default system prompt for judges.
    """
    return await call_anthropic(
        query=prompt,
        model=model,
        api_key=api_key,
        base_url=base_url,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        timeout=timeout,
    )
