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
import json
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

    mcp_registry: Any = None
    """MCPToolRegistry for executing tool calls (if tools came from MCP)."""

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

    Supports a multi-turn tool-calling loop when MCP tools are provided in
    the context. The agent calls the LLM, executes any tool calls via the
    MCP registry, and repeats until the LLM stops requesting tools.

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
        max_tool_rounds: int = 10,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_tool_rounds = max_tool_rounds

    async def run(self, query: str, context: Context) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        messages: list[dict[str, Any]] = []
        if context.system_prompt:
            messages.append({"role": "system", "content": context.system_prompt})
        messages.append({"role": "user", "content": query})

        # No tools → simple single-call
        if not context.tools:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    max_tokens=2000,
                    messages=messages,
                ),
                timeout=self.timeout,
            )
            return response.choices[0].message.content or ""

        # Tool-calling loop: call LLM → execute tool calls → repeat
        openai_tools = context.tools
        last_content = ""
        for _ in range(self.max_tool_rounds):
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    max_tokens=2000,
                    messages=messages,
                    tools=openai_tools,
                ),
                timeout=self.timeout,
            )
            msg = response.choices[0].message
            last_content = msg.content or ""

            # No tool calls → agent is done
            if not msg.tool_calls:
                return last_content

            # Append the assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            # Execute each tool call via MCP registry
            for tc in msg.tool_calls:
                tool_result = await self._execute_tool_call(tc, context)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result, default=str),
                })

        return last_content

    async def _execute_tool_call(self, tool_call: Any, context: Context) -> Any:
        """Execute a tool call via the MCP registry."""
        import json as _json
        name = tool_call.function.name
        try:
            arguments = _json.loads(tool_call.function.arguments)
        except Exception:
            arguments = {}

        if context.mcp_registry:
            return await context.mcp_registry.call_tool_by_name(name, arguments)

        return {"error": f"No registry available to execute tool: {name}"}


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
