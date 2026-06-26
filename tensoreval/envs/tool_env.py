"""Tool environment — model can call Python functions as tools.

Ported from PrimeIntellect Verifiers (MIT License).
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
    Tool,
    ToolCall,
    ToolMessage,
    TrajectoryStep,
    UserMessage,
)
from tensoreval.envs.multiturn_env import MultiTurnEnv


class ToolEnv(MultiTurnEnv):
    """Environment with tool calling support.

    Pass a list of Python functions as tools. The model can call them during evaluation.
    """

    def __init__(
        self,
        tools: list[Callable] | None = None,
        max_turns: int = 10,
        error_formatter: Callable[[Exception], str] = lambda e: f"{e}",
        **kwargs,
    ):
        super().__init__(max_turns=max_turns, **kwargs)
        self.tools = tools or []
        self.tool_map = {tool.__name__: tool for tool in self.tools}
        self.tool_defs = [self._func_to_tool_def(tool) for tool in self.tools]
        self.error_formatter = error_formatter

    @staticmethod
    def _func_to_tool_def(func: Callable) -> Tool:
        """Convert a Python function to a Tool definition."""
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or f"Call {func.__name__}"
        params = {}
        for name, param in sig.parameters.items():
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
            params[name] = {"type": param_type, "description": f"Parameter {name}"}
        return Tool(
            name=func.__name__,
            description=doc,
            parameters={"type": "object", "properties": params, "required": list(params.keys())},
        )

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        """Execute tool calls from the model's response."""
        if not messages:
            return []

        last_msg = messages[-1]
        tool_calls = getattr(last_msg, 'tool_calls', None) or (
            last_msg.get("tool_calls") if isinstance(last_msg, dict) else None
        )
        if not tool_calls:
            return []

        tool_messages = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                name = tc.get("name", "")
                args_str = tc.get("arguments", "{}")
                tc_id = tc.get("id", name)
            else:
                name = tc.name
                args_str = tc.arguments
                tc_id = tc.id

            if name in self.tool_map:
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    result = self.tool_map[name](**args)
                    if inspect.isawaitable(result):
                        result = await result
                    result_str = str(result)
                except Exception as e:
                    result_str = self.error_formatter(e)
            else:
                result_str = f"Unknown tool: {name}"

            tool_messages.append(ToolMessage(
                tool_call_id=tc_id,
                content=result_str,
            ))

        return tool_messages
