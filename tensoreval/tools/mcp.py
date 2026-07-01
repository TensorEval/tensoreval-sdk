"""MCP (Model Context Protocol) tool support for TensorEval.

Allows environments to expose tools via MCP servers.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any


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

    def add_server(self, name: str, server: MCPServer) -> None:
        """Add an MCP server to the registry."""
        self.servers[name] = server

    async def list_all_tools(self) -> list[MCPTool]:
        """List tools from all registered servers."""
        all_tools = []
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
        tools = []
        for server in self.servers.values():
            if server._tools:
                tools.extend(t.to_openai_tool() for t in server._tools)
        return tools

    async def call_tool_by_name(self, name: str, arguments: dict[str, Any]) -> Any:
        """Find and call a tool by name across all registered servers.

        Searches each server's cached tool list. Returns {"error": ...} if
        the tool is not found on any server.
        """
        for server in self.servers.values():
            if server._tools:
                for tool in server._tools:
                    if tool.name == name:
                        return await tool.call(arguments)
        return {"error": f"Tool not found: {name}"}
