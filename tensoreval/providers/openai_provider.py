"""OpenAI-compatible API provider.

Handles calls to any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import asyncio
from typing import Any


async def call_openai(
    query: str,
    model: str,
    api_key: str,
    base_url: str,
    system_prompt: str | None = None,
    max_tokens: int = 2000,
    timeout: float = 60.0,
) -> str:
    """Call an OpenAI-compatible API.

    Args:
        query: User message content.
        model: Model name.
        api_key: API key.
        base_url: API base URL.
        system_prompt: Optional system message.
        max_tokens: Max response tokens.
        timeout: Request timeout in seconds.

    Returns:
        Assistant response text.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        ),
        timeout=timeout,
    )
    return response.choices[0].message.content or ""


async def call_openai_judge(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    system_prompt: str = "You are an evaluation judge. Output only valid JSON.",
    max_tokens: int = 500,
    timeout: float = 60.0,
) -> str:
    """Call OpenAI-compatible API for judging.

    Same as call_openai but with a default system prompt for judges.
    """
    return await call_openai(
        query=prompt,
        model=model,
        api_key=api_key,
        base_url=base_url,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        timeout=timeout,
    )
