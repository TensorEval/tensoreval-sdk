"""Provider-backed tool loop used by AgenticGrader."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""

    @classmethod
    def from_name(
        cls,
        provider: str = "openai",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> "ProviderConfig":
        provider = provider.lower()
        defaults = {
            "openai": ("gpt-4.1", "https://api.openai.com/v1", "OPENAI_API_KEY"),
            "openrouter": ("openai/gpt-4.1", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
            "ollama": ("llama3.1", "http://localhost:11434/v1", ""),
            "anthropic": ("claude-sonnet-4-5", "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
        }
        if provider not in defaults:
            raise ValueError(f"Unsupported grader provider: {provider}")
        default_model, default_base_url, env_key = defaults[provider]
        return cls(
            provider=provider,
            model=model or os.environ.get("TENSOREVAL_GRADER_MODEL") or default_model,
            api_key=api_key or os.environ.get("TENSOREVAL_GRADER_API_KEY") or (os.environ.get(env_key) if env_key else "") or "",
            base_url=(base_url or os.environ.get("TENSOREVAL_GRADER_BASE_URL") or default_base_url).rstrip("/"),
        )


@dataclass
class MCPTool:
    server_name: str
    server_url: str
    name: str
    description: str
    parameters: dict[str, Any]


class MCPRegistry:
    def __init__(self, urls: list[str], timeout: float):
        self.urls = urls
        self.timeout = timeout
        self.tools: list[MCPTool] = []

    def discover(self) -> list[MCPTool]:
        tools: list[MCPTool] = []
        for index, url in enumerate(self.urls):
            server_name = f"mcp_{index}"
            data = _post_json(url, {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}, timeout=self.timeout)
            for tool in data.get("result", {}).get("tools", []):
                tools.append(MCPTool(
                    server_name=server_name,
                    server_url=url,
                    name=tool.get("name", ""),
                    description=tool.get("description", ""),
                    parameters=tool.get("inputSchema", {"type": "object", "properties": {}}),
                ))
        self.tools = tools
        return tools

    def to_openai_tools(self) -> list[dict[str, Any]]:
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

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in self.tools
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = next((candidate for candidate in self.tools if candidate.name == name), None)
        if tool is None:
            return {"error": f"Tool not found: {name}"}
        return _post_json(
            tool.server_url,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            timeout=self.timeout,
        ).get("result", {})


class ToolLoopAgent:
    def __init__(
        self,
        provider: ProviderConfig,
        instructions: str,
        max_steps: int = 10,
        timeout: float = 120.0,
        temperature: float = 0.0,
    ):
        self.provider = provider
        self.instructions = instructions
        self.max_steps = max_steps
        self.timeout = timeout
        self.temperature = temperature

    def run(self, prompt: str, mcp_urls: list[str]) -> dict[str, Any]:
        registry = MCPRegistry(mcp_urls, timeout=self.timeout)
        tools = registry.discover() if mcp_urls else []
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        trace_steps: list[dict[str, Any]] = []

        for step_index in range(self.max_steps):
            response = self._complete(messages, registry, has_tools=bool(tools))
            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])

            if not tool_calls:
                trace_steps.append({
                    "type": "message",
                    "step": step_index,
                    "provider": self.provider.provider,
                    "model": self.provider.model,
                    "content": content,
                })
                parsed = _parse_json_object(content)
                if parsed is None:
                    parsed = {"reward": 0.0, "passed": False, "rubric_scores": {}, "reasoning_summary": content}
                parsed.setdefault("grader_trace", {"steps": trace_steps})
                parsed["grader_trace"] = {"steps": trace_steps + parsed.get("grader_trace", {}).get("steps", [])}
                return parsed

            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            for call in tool_calls:
                name = call.get("name", "")
                arguments = call.get("arguments", {})
                started = time.monotonic()
                result = registry.call(name, arguments)
                duration_ms = (time.monotonic() - started) * 1000
                trace_steps.append({
                    "type": "tool_call",
                    "id": call.get("id", ""),
                    "name": name,
                    "arguments": arguments,
                })
                trace_steps.append({
                    "type": "tool_result",
                    "tool_call_id": call.get("id", ""),
                    "name": name,
                    "result": result,
                    "duration_ms": round(duration_ms, 2),
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": json.dumps(result, default=str),
                })

        return {
            "reward": 0.0,
            "passed": False,
            "rubric_scores": {},
            "reasoning_summary": "Grader exceeded max_steps",
            "grader_trace": {"steps": trace_steps},
        }

    def _complete(self, messages: list[dict[str, Any]], registry: MCPRegistry, has_tools: bool) -> dict[str, Any]:
        if self.provider.provider == "anthropic":
            return self._complete_anthropic(messages, registry, has_tools)
        return self._complete_openai_compatible(messages, registry, has_tools)

    def _complete_openai_compatible(self, messages: list[dict[str, Any]], registry: MCPRegistry, has_tools: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.provider.model,
            "messages": [{"role": "system", "content": self.instructions}] + messages,
            "temperature": self.temperature,
        }
        if has_tools:
            body["tools"] = registry.to_openai_tools()
            body["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if self.provider.api_key:
            headers["Authorization"] = f"Bearer {self.provider.api_key}"
        data = _post_json(f"{self.provider.base_url}/chat/completions", body, headers=headers, timeout=self.timeout)
        message = data.get("choices", [{}])[0].get("message", {})
        return {
            "content": message.get("content") or "",
            "tool_calls": _openai_tool_calls(message.get("tool_calls", [])),
        }

    def _complete_anthropic(self, messages: list[dict[str, Any]], registry: MCPRegistry, has_tools: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.provider.model,
            "max_tokens": 2000,
            "temperature": self.temperature,
            "system": self.instructions,
            "messages": _to_anthropic_messages(messages),
        }
        if has_tools:
            body["tools"] = registry.to_anthropic_tools()
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.provider.api_key:
            headers["x-api-key"] = self.provider.api_key
        data = _post_json(f"{self.provider.base_url}/v1/messages", body, headers=headers, timeout=self.timeout)
        content = ""
        tool_calls: list[dict[str, Any]] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": block.get("input", {}),
                })
        return {"content": content, "tool_calls": tool_calls}


def _openai_tool_calls(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []
    for call in raw:
        function = call.get("function", {})
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        calls.append({
            "id": call.get("id", ""),
            "name": function.get("name", ""),
            "arguments": arguments,
        })
    return calls


def _to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            converted.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id", ""),
                    "content": message.get("content", ""),
                }],
            })
        elif role == "assistant" and message.get("tool_calls"):
            content: list[dict[str, Any]] = []
            if message.get("content"):
                content.append({"type": "text", "text": message["content"]})
            for call in message.get("tool_calls", []):
                content.append({
                    "type": "tool_use",
                    "id": call.get("id", ""),
                    "name": call.get("name", ""),
                    "input": call.get("arguments", {}),
                })
            converted.append({"role": "assistant", "content": content})
        else:
            converted.append({"role": role, "content": message.get("content", "")})
    return converted


def _parse_json_object(content: str) -> dict[str, Any] | None:
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _post_json(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers=headers or {"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())
