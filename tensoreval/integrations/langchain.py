"""LangChain integration helpers.

These adapters let users evaluate existing LangChain/LangGraph-style agents
without rewriting them around TensorEval.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tensoreval.agents import Agent, Context


class CallbackHandler:
    """Minimal LangChain callback handler that records tool calls.

    Pass this handler through LangChain's ``config={"callbacks": [...]}``.
    The collected ``tool_trace`` can be copied into TensorEval run metadata by
    ``wrap_agent``.
    """

    def __init__(self) -> None:
        self.tool_trace: list[dict[str, Any]] = []
        self._pending: dict[str, dict[str, Any]] = {}

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: Any = None, **_: Any) -> None:
        name = serialized.get("name") or serialized.get("id") or "tool"
        key = str(run_id or len(self.tool_trace))
        self._pending[key] = {"tool": name, "args": input_str}

    def on_tool_end(self, output: Any, *, run_id: Any = None, **_: Any) -> None:
        key = str(run_id or len(self.tool_trace))
        row = self._pending.pop(key, {"tool": "tool", "args": ""})
        row["result"] = output
        self.tool_trace.append(row)

    def on_tool_error(self, error: BaseException, *, run_id: Any = None, **_: Any) -> None:
        key = str(run_id or len(self.tool_trace))
        row = self._pending.pop(key, {"tool": "tool", "args": ""})
        row["error"] = str(error)
        self.tool_trace.append(row)


class LangChainAgent(Agent):
    """TensorEval adapter for an existing LangChain runnable/agent executor."""

    def __init__(
        self,
        runnable: Any,
        input_key: str = "input",
        output_key: str | None = "output",
        callbacks: list[Any] | None = None,
    ) -> None:
        self.runnable = runnable
        self.input_key = input_key
        self.output_key = output_key
        self.callbacks = callbacks or []

    async def run(self, query: str, context: Context) -> str:
        handler = CallbackHandler()
        callbacks = [*self.callbacks, handler]
        payload = {self.input_key: query}
        config = {"callbacks": callbacks}

        if hasattr(self.runnable, "ainvoke"):
            result = await self.runnable.ainvoke(payload, config=config)
        elif hasattr(self.runnable, "invoke"):
            result = await asyncio.to_thread(self.runnable.invoke, payload, config=config)
        else:
            raise TypeError("LangChain runnable must expose invoke() or ainvoke()")

        context.metadata["tool_trace"] = handler.tool_trace
        return _extract_output(result, self.output_key)


def wrap_agent(
    runnable: Any,
    input_key: str = "input",
    output_key: str | None = "output",
    callbacks: list[Any] | None = None,
) -> LangChainAgent:
    """Wrap an existing LangChain runnable/agent for TensorEval.

    Example:
        agent = te.integrations.langchain.wrap_agent(agent_executor)
        results = te.Evaluation.run(dataset, te.AgentGrader(), agent=agent)
    """

    return LangChainAgent(runnable, input_key=input_key, output_key=output_key, callbacks=callbacks)


def _extract_output(result: Any, output_key: str | None) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if output_key and output_key in result:
            return str(result[output_key])
        for key in ("output", "answer", "response", "content"):
            if key in result:
                return str(result[key])
    content = getattr(result, "content", None)
    if content is not None:
        return str(content)
    return str(result)
