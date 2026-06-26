"""Multi-turn tool environment with proper tool dispatch.

Based on Verifiers ToolEnv pattern (MIT License).
Supports tool calling with automatic dispatch and error handling.
"""

import asyncio
import inspect
import json
import time
from typing import Any, Callable

from tensoreval.core.types import (
    AssistantMessage,
    Messages,
    RolloutInput,
    SamplingArgs,
    State,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    TrajectoryStep,
    UserMessage,
)
from tensoreval.envs.environment import Environment


def _func_to_tool_def(func: Callable) -> Tool:
    """Convert a Python function to a Tool definition.

    Inspects the function signature and docstring to generate
    a provider-agnostic tool schema.
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or f"Call {func.__name__}"

    params = {}
    required = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        param_type = "string"
        if param.annotation != inspect.Parameter.empty:
            type_map = {int: "integer", float: "number", bool: "boolean", str: "string"}
            param_type = type_map.get(param.annotation, "string")

        params[name] = {"type": param_type, "description": f"Parameter {name}"}
        if param.default == inspect.Parameter.empty:
            required.append(name)

    return Tool(
        name=func.__name__,
        description=doc,
        parameters={
            "type": "object",
            "properties": params,
            "required": required,
        },
    )


def _call_openai_with_tools(
    messages: list[dict],
    tools: list[dict],
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> dict:
    """Call OpenAI-compatible API with tool definitions."""
    from openai import AsyncOpenAI

    async def _call():
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        return {
            "content": choice.message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in (choice.message.tool_calls or [])
            ],
            "finish_reason": choice.finish_reason,
        }

    return _call()


def _call_anthropic_with_tools(
    messages: list[dict],
    tools: list[dict],
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> dict:
    """Call Anthropic-compatible API with tool definitions."""
    import anthropic

    async def _call():
        client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)

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
        if tools:
            kwargs["tools"] = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                }
                for t in tools
            ]

        response = await client.messages.create(**kwargs)

        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": json.dumps(block.input),
                })

        return {
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": "stop" if not tool_calls else "tool_calls",
        }

    return _call()


class MultiTurnToolEnv(Environment):
    """Multi-turn environment with tool calling support.

    The model can call Python functions as tools during evaluation.
    Each tool call is dispatched, executed, and the result is fed back
    to the model for the next turn.

    Usage:
        def search_database(query: str) -> str:
            '''Search the database for matching records.'''
            return db.search(query)

        def get_user_info(user_id: str) -> dict:
            '''Get user information by ID.'''
            return db.get_user(user_id)

        env = MultiTurnToolEnv(
            tools=[search_database, get_user_info],
            rubric=my_rubric,
            system_prompt="You are a helpful assistant with access to tools.",
            max_turns=10,
        )
    """

    def __init__(
        self,
        tools: list[Callable] | None = None,
        max_turns: int = 10,
        timeout_seconds: float | None = None,
        error_formatter: Callable[[Exception], str] = lambda e: f"Error: {e}",
        **kwargs,
    ):
        super().__init__(max_turns=max_turns, timeout_seconds=timeout_seconds, **kwargs)
        self.tools = tools or []
        self.tool_map = {tool.__name__: tool for tool in self.tools}
        self.tool_defs = [_func_to_tool_def(tool) for tool in self.tools]
        self.error_formatter = error_formatter

    async def _execute_tool(self, name: str, arguments: str) -> str:
        """Execute a tool and return the result as a string."""
        if name not in self.tool_map:
            return f"Error: Unknown tool '{name}'. Available: {list(self.tool_map.keys())}"

        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON arguments: {e}"

        try:
            result = self.tool_map[name](**args)
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        except Exception as e:
            return self.error_formatter(e)

    async def rollout(
        self,
        input: RolloutInput,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """Run a multi-turn rollout with tool calling."""
        state = self._create_state(input)
        state["timing"].generation.start = time.time()

        # Build initial prompt
        prompt = state.get("prompt", [])
        if isinstance(prompt, str):
            prompt = [UserMessage(content=prompt)]
        if self.system_prompt:
            prompt = [SystemMessage(content=self.system_prompt)] + prompt

        # Convert to dict messages
        messages = []
        for m in prompt:
            if hasattr(m, "model_dump"):
                messages.append(m.model_dump())
            elif isinstance(m, dict):
                messages.append(m)
            else:
                messages.append({"role": getattr(m, "role", "user"), "content": getattr(m, "content", str(m))})

        # Convert tool defs to dict format
        tools = []
        for td in self.tool_defs:
            tools.append({
                "name": td.name,
                "description": td.description,
                "parameters": td.parameters,
            })

        # Merge sampling args
        merged_args = {**self.sampling_args, **(sampling_args or {})}
        max_tokens = merged_args.get("max_tokens", 1024)
        temperature = merged_args.get("temperature", 0.7)

        # Detect API type
        is_anthropic = base_url and "anthropic" in base_url.lower()
        resolved_key = api_key or "dummy"
        resolved_url = base_url or "https://api.openai.com/v1"

        trajectory = []
        turn = 0

        try:
            while turn < self.max_turns:
                turn += 1

                # Call model
                if is_anthropic:
                    response = await _call_anthropic_with_tools(
                        messages, tools, model, resolved_key, resolved_url, max_tokens, temperature
                    )
                else:
                    response = await _call_openai_with_tools(
                        messages, tools, model, resolved_key, resolved_url, max_tokens, temperature
                    )

                content = response["content"]
                tool_calls = response["tool_calls"]
                finish_reason = response["finish_reason"]

                # Record assistant message
                assistant_msg = AssistantMessage(
                    content=content,
                    tool_calls=[ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"]) for tc in tool_calls] if tool_calls else None,
                )
                messages.append(assistant_msg.model_dump())

                # If no tool calls, we're done
                if not tool_calls:
                    trajectory.append(TrajectoryStep(
                        prompt=prompt,
                        completion=[assistant_msg],
                        trajectory_id=state["trajectory_id"],
                    ))
                    break

                # Execute tool calls
                for tc in tool_calls:
                    result = await self._execute_tool(tc["name"], tc["arguments"])
                    tool_msg = ToolMessage(tool_call_id=tc["id"], content=result)
                    messages.append(tool_msg.model_dump())

                    trajectory.append(TrajectoryStep(
                        prompt=prompt,
                        completion=[assistant_msg, tool_msg],
                        trajectory_id=state["trajectory_id"],
                    ))

            state["completion"] = [AssistantMessage(content=content)]
            state["trajectory"] = trajectory

        except Exception as e:
            state["error"] = e
            state["is_completed"] = True

        state["timing"].generation.end = time.time()
        state["is_completed"] = True
        return state
