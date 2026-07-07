"""MCP (Model Context Protocol) tool support for TensorEval.

Allows environments to expose tools via MCP servers.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Callable, get_args, get_origin


class MCPTool:
    """Represents a tool exposed by an MCP server."""

    def __init__(self, name: str, description: str, parameters: dict[str, Any], server: MCPServer):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.server = server

    async def call(self, arguments: dict[str, Any]) -> Any:
        """Call this tool with the given arguments."""
        return await self.server.call_tool(self.name, arguments)

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class LocalTool:
    """Represents a local Python callable exposed as an OpenAI function tool."""

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn
        meta = getattr(fn, "__tensoreval_tool__", {}) or {}
        self.name = meta.get("name") or getattr(fn, "__name__", "local_tool")
        doc = inspect.getdoc(fn) or ""
        self.description = meta.get("description") or (doc.splitlines()[0] if doc else f"Call local Python function {self.name}.")
        self.parameters = _schema_from_callable(fn)

    async def call(self, arguments: dict[str, Any]) -> Any:
        """Call the underlying Python function."""
        result = self.fn(**arguments)
        if inspect.isawaitable(result):
            return await result
        return result

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI Chat Completions tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class MCPServer:
    """Client for an MCP server.

    Usage:
        server = MCPServer(url="http://localhost:9000/mcp")
        tools = await server.list_tools()
        result = await server.call_tool("lookup_order", {"order_id": "O7841"})
    """

    def __init__(
        self,
        url: str = "",
        name: str = "default",
        transport: str = "streamable-http",
        auth_token: str | None = None,
    ):
        self.url = url
        self.name = name
        self.transport = transport
        self.auth_token = auth_token
        self._tools: list[MCPTool] | None = None

    async def list_tools(self) -> list[MCPTool]:
        """List available tools from the MCP server."""
        if self._tools is not None:
            return self._tools

        try:
            import httpx
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.url,
                    json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
                    headers=headers,
                    timeout=10.0,
                )
                data = response.json()
                tools_data = data.get("result", {}).get("tools", [])
                self._tools = [
                    MCPTool(
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        parameters=t.get("inputSchema", {"type": "object", "properties": {}}),
                        server=self,
                    )
                    for t in tools_data
                ]
                return self._tools
        except Exception:
            return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        try:
            import httpx
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.url,
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {"name": name, "arguments": arguments},
                        "id": 1,
                    },
                    headers=headers,
                    timeout=30.0,
                )
                data = response.json()
                result = data.get("result", {})
                return result.get("content", result)
        except Exception as e:
            return {"error": str(e)}


class MCPToolRegistry:
    """Registry of MCP servers and their tools.

    Usage:
        registry = MCPToolRegistry()
        registry.add_server("cx_app", MCPServer(url="http://localhost:9000/mcp"))
        tools = await registry.list_all_tools()
    """

    def __init__(self):
        self.servers: dict[str, MCPServer] = {}
        self.local_tools: dict[str, LocalTool] = {}

    def add_server(self, name: str, server: MCPServer) -> None:
        """Add an MCP server to the registry."""
        self.servers[name] = server

    def add_local_tool(self, fn: Callable[..., Any]) -> None:
        """Add a local Python callable to the registry."""
        tool = LocalTool(fn)
        self.local_tools[tool.name] = tool

    async def list_all_tools(self) -> list[MCPTool]:
        """List tools from all registered servers."""
        all_tools = list(self.local_tools.values())
        for server in self.servers.values():
            tools = await server.list_tools()
            all_tools.extend(tools)
        return all_tools

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on a specific server."""
        if server_name not in self.servers:
            raise ValueError(f"Unknown MCP server: {server_name}")
        return await self.servers[server_name].call_tool(tool_name, arguments)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Convert all tools to OpenAI tool format."""
        tools = [t.to_openai_tool() for t in self.local_tools.values()]
        for server in self.servers.values():
            if server._tools:
                tools.extend(t.to_openai_tool() for t in server._tools)
        return tools

    async def call_tool_by_name(self, name: str, arguments: dict[str, Any]) -> Any:
        """Find and call a tool by name across all registered servers.

        Searches each server's cached tool list. Returns {"error": ...} if
        the tool is not found on any server.
        """
        if name in self.local_tools:
            return await self.local_tools[name].call(arguments)
        for server in self.servers.values():
            if server._tools:
                for tool in server._tools:
                    if tool.name == name:
                        return await tool.call(arguments)
        return {"error": f"Tool not found: {name}"}


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None, description: str | None = None):
    """Mark a Python callable as a TensorEval local tool.

    The decorated callable can be passed to ``te.Environment(tools=[...])`` or
    ``te.EvalConfig(tools=[...])`` and will be exposed to compatible agents and
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


def _schema_from_callable(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build a minimal JSON Schema object from a Python callable signature."""
    signature = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        properties[name] = {"type": _json_type(param.annotation)}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _json_type(annotation: Any) -> str:
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
            return _json_type(args[0])
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
