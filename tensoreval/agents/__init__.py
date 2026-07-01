"""Agent integration for TensorEval.

TensorEval supports multiple ways to bring your own agent:

1. **Function** — simplest, just pass `async def my_agent(query) -> str`
2. **Agent class** — extend `Agent` for more control (tools, state, etc.)
3. **OpenAI endpoint** — point at any OpenAI-compatible API
4. **Anthropic endpoint** — point at Anthropic/Mimo API

Usage:
    # Option 1: Function (simplest)
    async def my_agent(query: str) -> str:
        return await my_llm.call(query)

    results = Evaluation.run(dataset, grader, agent=my_agent)

    # Option 2: Agent class (more control)
    class MyAgent(Agent):
        async def run(self, query: str, context: Context) -> str:
            tools = context.tools
            # ... custom logic ...
            return response

    results = Evaluation.run(dataset, grader, agent=MyAgent())

    # Option 3: OpenAI endpoint
    results = Evaluation.run(dataset, grader, agent="http://localhost:8000")

    # Option 4: Anthropic endpoint
    results = Evaluation.run(dataset, grader, agent="anthropic:mimo-v2.5-pro")
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class Context:
    """Context provided to agents during evaluation.

    Contains the query, system prompt, and any tools/MCP servers available.
    """

    query: str
    """The user's question/task."""

    system_prompt: str | None = None
    """System prompt from the environment."""

    tools: list[dict[str, Any]] = field(default_factory=list)
    """Available tools in OpenAI format."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional context (sample ID, rubrics, etc.)."""


class Agent(ABC):
    """Base class for agents.

    Extend this class and implement `run()` for full control over
    how your agent processes queries.

    Usage:
        class MyAgent(Agent):
            async def run(self, query: str, context: Context) -> str:
                # Call your LLM, use tools, run custom logic, etc.
                return response

        results = Evaluation.run(dataset, grader, agent=MyAgent())
    """

    @abstractmethod
    async def run(self, query: str, context: Context) -> str:
        """Process a query and return a response.

        Args:
            query: The user's question.
            context: Environment context (system prompt, tools, metadata).

        Returns:
            The agent's response as a string.
        """
        ...


class FunctionAgent(Agent):
    """Wraps a simple async function as an Agent.

    Usage:
        async def my_agent(query: str) -> str:
            return "answer"

        agent = FunctionAgent(my_agent)
    """

    def __init__(self, fn: Callable[[str], Awaitable[str]]):
        self.fn = fn

    async def run(self, query: str, context: Context) -> str:
        return await self.fn(query)


class OpenAIAgent(Agent):
    """Agent that calls an OpenAI-compatible API.

    Usage:
        agent = OpenAIAgent(
            model="gpt-4o",
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
        )
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    async def run(self, query: str, context: Context) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        messages: list[dict[str, Any]] = []
        if context.system_prompt:
            messages.append({"role": "system", "content": context.system_prompt})
        messages.append({"role": "user", "content": query})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 2000,
            "messages": messages,
        }
        if context.tools:
            kwargs["tools"] = context.tools

        response = await asyncio.wait_for(
            client.chat.completions.create(**kwargs),
            timeout=self.timeout,
        )
        return response.choices[0].message.content or ""


class AnthropicAgent(Agent):
    """Agent that calls an Anthropic-compatible API.

    Usage:
        agent = AnthropicAgent(
            model="mimo-v2.5-pro",
            api_key="tp-...",
            base_url="https://token-plan-sgp.xiaomimimo.com/anthropic",
        )
    """

    def __init__(
        self,
        model: str = "mimo-v2.5-pro",
        api_key: str = "",
        base_url: str = "",
        timeout: float = 60.0,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    async def run(self, query: str, context: Context) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.api_key, base_url=self.base_url)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": query}],
        }
        if context.system_prompt:
            kwargs["system"] = context.system_prompt

        response = await asyncio.wait_for(
            client.messages.create(**kwargs),
            timeout=self.timeout,
        )
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""


class EndpointAgent(Agent):
    """Agent that calls an HTTP endpoint (OpenAI-compatible).

    Usage:
        agent = EndpointAgent(url="http://localhost:8000/v1/chat/completions")
    """

    def __init__(self, url: str, timeout: float = 120.0):
        self.url = url
        self.timeout = timeout

    async def run(self, query: str, context: Context) -> str:
        import httpx

        messages: list[dict[str, Any]] = []
        if context.system_prompt:
            messages.append({"role": "system", "content": context.system_prompt})
        messages.append({"role": "user", "content": query})

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.url,
                json={"messages": messages},
                timeout=self.timeout,
            )
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def resolve_agent(
    agent: Agent | Callable | str | None = None,
    model: str = "gpt-4o",
    api_key: str = "",
    base_url: str = "",
    agent_port: int | None = None,
) -> Agent:
    """Resolve various agent formats into an Agent instance.

    Accepts:
    - Agent instance → returned as-is
   - Callable (async function) → wrapped in FunctionAgent
    - String starting with "http" → EndpointAgent
    - String starting with "anthropic:" → AnthropicAgent
    - String with api_key/base_url → OpenAIAgent
    - None with agent_port → EndpointAgent on localhost
    """
    if isinstance(agent, Agent):
        return agent

    if callable(agent):
        return FunctionAgent(agent)

    if isinstance(agent, str):
        if agent.startswith("http"):
            return EndpointAgent(url=agent)
        if agent.startswith("anthropic:"):
            model = agent.split(":", 1)[1]
            return AnthropicAgent(model=model, api_key=api_key, base_url=base_url)
        # Assume it's a model name
        return OpenAIAgent(model=agent, api_key=api_key, base_url=base_url)

    # Fallback: OpenAI agent with provided config
    if agent_port:
        return EndpointAgent(url=f"http://localhost:{agent_port}/v1/chat/completions")

    return OpenAIAgent(model=model, api_key=api_key, base_url=base_url)
