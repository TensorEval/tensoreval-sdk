"""MCP (Model Context Protocol) tool support for TensorEval.

Single, unified MCP module using only stdlib ``urllib`` — no external
HTTP dependencies. Handles:

- Tool discovery from MCP servers (``tools/list``)
- Tool invocation (``tools/call``)
- Local Python callables exposed as OpenAI function tools
- Multi-turn LLM verification loop with MCP tool access

Usage:
    from tensoreval.mcp import MCPServer, MCPRegistry, tool

    # Direct server access
    server = MCPServer(url="http://localhost:9000/mcp")
    tools = await server.list_tools()
    result = await server.call_tool("lookup_order", {"order_id": "O7841"})

    # Registry (multiple servers + local tools)
    registry = MCPRegistry()
    registry.add_server("cx", MCPServer(url="http://localhost:9000/mcp"))
    registry.add_local_tool(my_python_function)
    tools = await registry.list_all_tools()
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Callable, get_args, get_origin


# ---------------------------------------------------------------------------
# Local tools (Python callables exposed as OpenAI function tools)
# ---------------------------------------------------------------------------

def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None, description: str | None = None):
    """Mark a Python callable as a TensorEval local tool.

    The decorated callable can be passed to ``Environment(tools=[...])``
    or ``EvalConfig(tools=[...])`` and will be exposed to agents and
    graders as an OpenAI-style function tool.
    """
    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn.__tensoreval_tool__ = {
            "name": name or getattr(fn, "__name__", "local_tool"),
            "description": description,
        }
        return fn

    if fn is not None:
        return decorate(fn)
    return decorate


class LocalTool:
    """A local Python callable exposed as an OpenAI function tool."""

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn
        meta = getattr(fn, "__tensoreval_tool__", {}) or {}
        self.name = meta.get("name") or getattr(fn, "__name__", "local_tool")
        doc = inspect.getdoc(fn) or ""
        self.description = meta.get("description") or (doc.splitlines()[0] if doc else f"Call {self.name}.")
        self.parameters = schema_from_callable(fn)

    async def call(self, arguments: dict[str, Any]) -> Any:
        result = self.fn(**arguments)
        if inspect.isawaitable(result):
            return await result
        return result

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# MCP server client (urllib only — no httpx)
# ---------------------------------------------------------------------------

@dataclass
class MCPToolInfo:
    """Discovered MCP tool metadata."""
    server_url: str
    name: str
    description: str
    parameters: dict[str, Any]


class MCPServer:
    """Client for a single MCP server.

    Args:
        url: MCP server endpoint URL.
        name: Server name (for registry identification).
        auth_token: Optional bearer token for authentication.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        url: str = "",
        name: str = "default",
        auth_token: str | None = None,
        timeout: float = 30.0,
    ):
        self.url = url
        self.name = name
        self.auth_token = auth_token
        self.timeout = timeout
        self._tools: list[MCPToolInfo] | None = None

    async def list_tools(self) -> list[MCPToolInfo]:
        """List available tools from the MCP server."""
        if self._tools is not None:
            return self._tools
        return await asyncio.to_thread(self.discover_tools)

    def discover_tools(self) -> list[MCPToolInfo]:
        """Synchronously discover tools (urllib is sync anyway)."""
        try:
            data = post_json(self.url, {
                "jsonrpc": "2.0", "id": 1,
                "method": "tools/list", "params": {},
            }, headers=self.headers(), timeout=self.timeout)
            tools_data = data.get("result", {}).get("tools", [])
            self._tools = [
                MCPToolInfo(
                    server_url=self.url,
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    parameters=t.get("inputSchema", {"type": "object", "properties": {}}),
                )
                for t in tools_data
            ]
            return self._tools
        except Exception:
            self._tools = []
            return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        return await asyncio.to_thread(self.call_tool_sync, name, arguments)

    def call_tool_sync(self, name: str, arguments: dict[str, Any]) -> Any:
        """Synchronously call a tool on the MCP server."""
        try:
            data = post_json(self.url, {
                "jsonrpc": "2.0", "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }, headers=self.headers(), timeout=self.timeout)
            result = data.get("result", {})
            return result.get("content", result)
        except Exception as exc:
            return {"error": str(exc)}

    def headers(self) -> dict[str, str]:
        result: dict[str, str] = {}
        if self.auth_token:
            result["Authorization"] = f"Bearer {self.auth_token}"
        return result


# ---------------------------------------------------------------------------
# Registry (multiple servers + local tools)
# ---------------------------------------------------------------------------

class MCPRegistry:
    """Registry of MCP servers and local Python tools.

    Args:
        urls: Optional list of MCP server URLs to register on init.
        timeout: Default request timeout for all servers.
    """

    def __init__(self, urls: list[str] | None = None, timeout: float = 30.0):
        self.servers: dict[str, MCPServer] = {}
        self.local_tools: dict[str, LocalTool] = {}
        self.timeout = timeout
        for i, url in enumerate(urls or []):
            self.add_server(f"server_{i + 1}", MCPServer(url=url, timeout=timeout))

    def add_server(self, name: str, server: MCPServer) -> None:
        self.servers[name] = server

    def add_local_tool(self, fn: Callable[..., Any]) -> None:
        tool_obj = LocalTool(fn)
        self.local_tools[tool_obj.name] = tool_obj

    async def list_all_tools(self) -> list[Any]:
        """List tools from all registered servers and local tools."""
        all_tools: list[Any] = list(self.local_tools.values())
        for server in self.servers.values():
            tools = await server.list_tools()
            all_tools.extend(tools)
        return all_tools

    def discover_all_tools_sync(self) -> None:
        """Synchronously discover tools from all servers (populates cache)."""
        for server in self.servers.values():
            if server._tools is None:
                server.discover_tools()

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Convert all tools to OpenAI function-tool format."""
        tools = [t.to_openai_tool() for t in self.local_tools.values()]
        for server in self.servers.values():
            if server._tools:
                for t in server._tools:
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        },
                    })
        return tools

    async def call_tool_by_name(self, name: str, arguments: dict[str, Any]) -> Any:
        """Find and call a tool by name across all servers and local tools."""
        if name in self.local_tools:
            return await self.local_tools[name].call(arguments)
        for server in self.servers.values():
            if server._tools:
                for tool_info in server._tools:
                    if tool_info.name == name:
                        return await server.call_tool(name, arguments)
        return {"error": f"Tool not found: {name}"}

    def call_tool_by_name_sync(self, name: str, arguments: dict[str, Any]) -> Any:
        """Synchronously find and call a tool by name."""
        if name in self.local_tools:
            tool_obj = self.local_tools[name]
            result = tool_obj.fn(**arguments)
            if inspect.isawaitable(result):
                # Best effort — try to run the coroutine
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        return {"error": "Cannot await async local tool from sync context"}
                    return asyncio.run(await_result(result))
                except RuntimeError:
                    return asyncio.run(await_result(result))
            return result
        for server in self.servers.values():
            if server._tools:
                for tool_info in server._tools:
                    if tool_info.name == name:
                        return server.call_tool_sync(name, arguments)
        return {"error": f"Tool not found: {name}"}


async def await_result(coro: Any) -> Any:
    """Await a coroutine (helper for sync-to-async bridge)."""
    return await coro


# ---------------------------------------------------------------------------
# Verification loop — multi-turn LLM + MCP tools for grader verification
# ---------------------------------------------------------------------------

def run_verification_loop(
    *,
    model: str,
    api_key: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    mcp_urls: list[str],
    max_turns: int = 10,
    timeout: float = 120.0,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run a multi-turn LLM loop with MCP tool access for verification.

    The LLM receives the system + user prompt, can call MCP tools to
    verify claims, and returns structured JSON when done.

    Returns:
        Dict with keys:
            - ``content``: final LLM text response
            - ``grader_trace``: list of tool calls and results
            - ``parsed``: parsed JSON from LLM output (or None)
    """
    registry = MCPRegistry(mcp_urls, timeout=timeout)
    if mcp_urls:
        registry.discover_all_tools_sync()

    openai_tools = registry.to_openai_tools()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    trace: list[dict[str, Any]] = []

    for turn in range(max_turns):
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if openai_tools:
            body["tools"] = openai_tools
            body["tool_choice"] = "auto"

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        data = post_json(
            f"{base_url.rstrip('/')}/chat/completions",
            body, headers=headers, timeout=timeout,
        )

        message = data.get("choices", [{}])[0].get("message", {})
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            trace.append({"type": "final", "turn": turn, "content": content[:500]})
            return {"content": content, "grader_trace": trace, "parsed": try_parse_json(content)}

        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", "{}"),
                    },
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}

            started = time.monotonic()
            result = registry.call_tool_by_name_sync(name, args)
            duration_ms = (time.monotonic() - started) * 1000

            trace.append({
                "type": "tool_call",
                "turn": turn,
                "name": name,
                "arguments": args,
                "result": result,
                "duration_ms": round(duration_ms, 2),
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": json.dumps(result, default=str),
            })

    return {
        "content": "",
        "grader_trace": trace,
        "parsed": None,
        "error": "Verification loop exceeded max_turns",
    }


# ---------------------------------------------------------------------------
# HTTP + JSON helpers (stdlib only)
# ---------------------------------------------------------------------------

def post_json(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """POST JSON and return parsed JSON response."""
    data = json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def try_parse_json(text: str) -> dict[str, Any] | None:
    """Try to extract and parse a JSON object from text."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    fence_match = re.search(r"```json\s*\n([\s\S]*?)```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    raw_match = re.search(r'\{[\s\S]*"rubric_scores"[\s\S]*\}', text)
    if raw_match:
        try:
            return json.loads(raw_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Schema inference for local tools
# ---------------------------------------------------------------------------

def schema_from_callable(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build a minimal JSON Schema from a Python callable signature."""
    signature = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        properties[name] = {"type": json_type(param.annotation)}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def json_type(annotation: Any) -> str:
    """Map a Python type annotation to a JSON Schema type string."""
    if annotation is inspect.Parameter.empty:
        return "string"
    origin = get_origin(annotation)
    if origin is not None:
        if origin in (list, tuple, set):
            return "array"
        if origin is dict:
            return "object"
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if args:
            return json_type(args[0])
    if annotation is str:
        return "string"
    if annotation is bool:
        return "boolean"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation in (dict, list):
        return "object" if annotation is dict else "array"
    return "string"
