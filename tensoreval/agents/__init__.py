"""Agent integration for TensorEval.

TensorEval supports multiple ways to bring your own agent:

1. **Function** — simplest, just pass `async def my_agent(query) -> TEvalResult | str`
2. **Agent class** — extend `Agent` for more control (tools, state, etc.)
3. **OpenAI endpoint** — point at any OpenAI-compatible API

The recommended approach is **Option 1** with a plain callable that
returns a :class:`tensoreval.types.TEvalResult`:

    async def my_agent(query: str) -> te.TEvalResult:
        # Run your agent however you want — LangChain, CrewAI, raw HTTP
        # Capture the tool trace yourself and return it
        return te.TEvalResult(
            response="Refund approved",
            tool_trace=[
                {"tool": "lookup_order", "args": {"id": "O123"}, "result": {"status": "ok"}},
            ],
        )

    results = Evaluation.run(dataset, grader, agent=my_agent)

If you don't need tool trace evaluation, just return a string:

    async def my_agent(query: str) -> str:
        return "answer"

    results = Evaluation.run(dataset, grader, agent=my_agent)

"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from tensoreval.types import TEvalResult


async def _call_with_rate_limit_retry(coro_factory, *, max_retries=3, base_delay=2.0):
    """Call an async function with retry on rate-limit (429) errors.

    coor_factory: a zero-arg callable that returns a fresh coroutine each call.
    """
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            is_rate_limit = (
                getattr(e, "status_code", None) == 429
                or "429" in str(e)
                or "rate" in str(e).lower()
            )
            if not is_rate_limit or attempt == max_retries:
                raise
            delay = base_delay * (attempt + 1)
            await asyncio.sleep(delay)


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

    The function can return a ``TEvalResult``, a ``str``, or a ``dict``
    with a ``response`` key. All are normalized via ``TEvalResult.coerce``.

    Usage:
        async def my_agent(query: str) -> te.TEvalResult:
            return te.TEvalResult(response="answer", tool_trace=[...])

        agent = FunctionAgent(my_agent)
    """

    def __init__(self, fn: Callable[[str], Awaitable[Any]]):
        self.fn = fn

    async def run(self, query: str, context: Context) -> str:
        result = await self.fn(query)
        coerced = TEvalResult.coerce(result)
        # Stash tool trace in context metadata for the grader
        if coerced.tool_trace:
            context.metadata.setdefault("tool_trace", [])
            context.metadata["tool_trace"].extend(coerced.tool_trace)
        return coerced.response


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
            response = await _call_with_rate_limit_retry(
                lambda: asyncio.wait_for(
                    client.chat.completions.create(
                        model=self.model,
                        max_tokens=2000,
                        messages=messages,
                    ),
                    timeout=self.timeout,
                )
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
        name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments)
        except Exception:
            arguments = {}

        if context.mcp_registry:
            return await context.mcp_registry.call_tool_by_name(name, arguments)

        return {"error": f"No registry available to execute tool: {name}"}


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
    - String with api_key/base_url → OpenAIAgent
    """
    if isinstance(agent, Agent):
        return agent

    if callable(agent):
        return FunctionAgent(agent)

    if isinstance(agent, str):
        if agent.startswith("http") or agent.startswith("anthropic:"):
            raise ValueError("Pass existing HTTP/Anthropic agents through a wrapper function or integration adapter.")
        return OpenAIAgent(model=agent, api_key=api_key, base_url=base_url)

    if agent_port:
        raise ValueError("agent_port is no longer a public shortcut. Pass an existing agent function or integration adapter.")

    return OpenAIAgent(model=model, api_key=api_key, base_url=base_url)
