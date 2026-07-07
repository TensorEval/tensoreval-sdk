"""Zero-dependency MCP tool loop for active verification.

When the grader is given MCP server URLs, it can discover tools from
those servers and call them to independently verify the agent's work.

All HTTP is done via ``urllib.request`` from the standard library —
no external dependencies required.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCPToolInfo:
    """Discovered MCP tool metadata."""

    server_url: str
    name: str
    description: str
    parameters: dict[str, Any]


class MCPRegistry:
    """Discovers and calls tools on one or more MCP servers."""

    def __init__(self, urls: list[str], timeout: float = 30.0):
        self.urls = urls
        self.timeout = timeout
        self.tools: list[MCPToolInfo] = []

    def discover(self) -> list[MCPToolInfo]:
        """Query each MCP server for its tool list."""
        tools: list[MCPToolInfo] = []
        for url in self.urls:
            try:
                data = _post_json(url, {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {},
                }, timeout=self.timeout)
                for tool in data.get("result", {}).get("tools", []):
                    tools.append(MCPToolInfo(
                        server_url=url,
                        name=tool.get("name", ""),
                        description=tool.get("description", ""),
                        parameters=tool.get("inputSchema", {"type": "object", "properties": {}}),
                    ))
            except Exception:
                continue
        self.tools = tools
        return tools

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Convert discovered tools to OpenAI function-tool format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool by name on the appropriate MCP server."""
        tool = next((t for t in self.tools if t.name == name), None)
        if tool is None:
            return {"error": f"Tool not found: {name}"}
        try:
            data = _post_json(tool.server_url, {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }, timeout=self.timeout)
            return data.get("result", data)
        except Exception as exc:
            return {"error": str(exc)}


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

    Returns a dict with keys:
        - ``content``: the final LLM text response
        - ``grader_trace``: list of tool calls and results
        - ``parsed``: parsed JSON from the LLM output (or None)
    """
    registry = MCPRegistry(mcp_urls, timeout=timeout)
    tools = registry.discover() if mcp_urls else []
    openai_tools = registry.to_openai_tools() if tools else []

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

        data = _post_json(
            f"{base_url.rstrip('/')}/chat/completions",
            body,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=timeout,
        )

        message = data.get("choices", [{}])[0].get("message", {})
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            # LLM is done — return the text
            trace.append({"type": "final", "turn": turn, "content": content[:500]})
            parsed = _try_parse_json(content)
            return {"content": content, "grader_trace": trace, "parsed": parsed}

        # Append assistant message with tool calls
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

        # Execute each tool call via MCP
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}

            started = time.monotonic()
            result = registry.call(name, args)
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

    # Exceeded max_turns
    return {
        "content": "",
        "grader_trace": trace,
        "parsed": None,
        "error": "Verification loop exceeded max_turns",
    }


# ---------------------------------------------------------------------------
# HTTP + JSON helpers (stdlib only)
# ---------------------------------------------------------------------------

def _post_json(
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


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Try to extract and parse a JSON object from text."""
    import re

    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try ```json fence
    fence_match = re.search(r"```json\s*\n([\s\S]*?)```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try raw JSON object
    raw_match = re.search(r"\{[\s\S]*\"rubric_scores\"[\s\S]*\}", text)
    if raw_match:
        try:
            return json.loads(raw_match.group(0))
        except json.JSONDecodeError:
            pass

    return None
