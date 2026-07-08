"""Environment configuration and lifecycle manager.

Handles:
- System prompt for the agent
- Docker container management (start/stop)
- MCP server connection (direct URL or Docker)
- Agent endpoint connection (direct URL or Docker)

Usage:
    # Simple (no Docker)
    env = te.Environment(system_prompt="...")

    # With Docker
    env = te.Environment(
        system_prompt="...",
        agent={"image": "python:3.12-slim", "port": 8000, "command": "python /app/agent.py"},
        mcp={"image": "node:18-slim", "port": 9000},
        env_file=".env",
    )

    # With direct URLs (no Docker)
    env = te.Environment(
        system_prompt="...",
        agent_url="http://localhost:8000",
        mcp_url="http://localhost:9000/mcp",
    )
"""

from __future__ import annotations

from typing import Any, Callable


class Environment:
    """Environment configuration and lifecycle manager.

    Args:
        system_prompt: System prompt for the agent.
        tools: Local Python callables to expose as tools.
        agent_url: Direct URL to an agent endpoint (no Docker).
        mcp_url: Direct URL to an MCP server (no Docker).
        mcp_servers: List of MCP server configs (dicts or MCPServer objects).
        agent: Docker config dict for the agent service.
        mcp: Docker config dict for the MCP server service.
        env_file: Path to a .env file for Docker containers.
        compose_yaml: Path to an existing compose.yaml file.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        tools: list[Callable] | None = None,
        agent_url: str | None = None,
        mcp_url: str | None = None,
        mcp_servers: list[Any] | None = None,
        agent: dict[str, Any] | None = None,
        mcp: dict[str, Any] | None = None,
        env_file: str | None = None,
        compose_yaml: str | None = None,
    ):
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.agent_url = agent_url
        self.mcp_url = mcp_url
        self.mcp_servers = mcp_servers or []
        self.agent = agent
        self.mcp = mcp
        self.env_file = env_file
        self.compose_yaml = compose_yaml
        self._started = False
        self._compose = None

    async def start(self) -> dict[str, str]:
        """Start Docker containers if configured.

        Returns:
            Dict of service_name → URL.
        """
        if self._started:
            return self._get_urls()

        # If direct URLs set and no Docker services needed, skip startup
        if self.agent_url and not self.agent and not self.mcp:
            self._started = True
            return self._get_urls()

        if self.agent or self.mcp:
            self._compose = self._build_compose()
            urls = await self._compose.up()
            if "agent" in urls:
                self.agent_url = urls["agent"]
            if "mcp-server" in urls:
                self.mcp_url = urls["mcp-server"] + "/mcp"

        self._started = True
        return self._get_urls()

    async def stop(self) -> None:
        """Stop Docker containers."""
        if self._compose:
            await self._compose.down()
            self._compose = None
        self._started = False

    def get_agent_url(self) -> str | None:
        """Get the configured or Docker-derived agent URL."""
        if self.agent_url:
            return self.agent_url
        if self.agent and "port" in self.agent:
            return f"http://localhost:{self.agent['port']}"
        return None

    def get_mcp_url(self) -> str | None:
        """Get the configured or Docker-derived MCP URL."""
        if self.mcp_url:
            return self.mcp_url
        if self.mcp and "port" in self.mcp:
            return f"http://localhost:{self.mcp['port']}/mcp"
        return None

    def _build_compose(self):
        from tensoreval.docker import DockerCompose

        services: dict[str, dict[str, Any]] = {}
        if self.agent:
            services["agent"] = self.agent
        if self.mcp:
            services["mcp-server"] = self.mcp
        return DockerCompose(
            services=services,
            compose_yaml=self.compose_yaml,
            env_file=self.env_file,
        )

    def _get_urls(self) -> dict[str, str]:
        urls: dict[str, str] = {}
        if self.agent_url:
            urls["agent"] = self.agent_url
        if self.mcp_url:
            urls["mcp"] = self.mcp_url
        return urls

    async def __aenter__(self) -> "Environment":
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()

    def __repr__(self) -> str:
        parts = []
        if self.system_prompt:
            parts.append(f"system_prompt={self.system_prompt[:30]}...")
        if self.agent_url:
            parts.append(f"agent_url={self.agent_url}")
        if self.mcp_url:
            parts.append(f"mcp_url={self.mcp_url}")
        if self.agent:
            parts.append("agent=docker")
        if self.mcp:
            parts.append("mcp=docker")
        return f"Environment({', '.join(parts)})"
