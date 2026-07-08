"""Environment configuration and lifecycle manager.

Handles:
- System prompt
- Docker container management (start/stop)
- API key injection
- MCP server connection
- Agent endpoint connection

Usage:
    # Simple (no Docker)
    env = te.Environment(system_prompt="...")

    # With Docker
    env = te.Environment(
        system_prompt="...",
        agent={"image": "python:3.12-slim", "port": 8000, "env": {"KEY": "val"}},
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

from typing import Any, Callable


class Environment:
    """Environment configuration and lifecycle manager."""

    def __init__(
        self,
        system_prompt: str | None = None,
        tools: list[Callable] | None = None,
        docker_image: str | None = None,
        compose_yaml: str | None = None,
        mcp_url: str | None = None,
        mcp_servers: list[Any] | None = None,
        agent_url: str | None = None,
        agent: dict[str, Any] | None = None,
        mcp: dict[str, Any] | None = None,
        env_file: str | None = None,
        config: dict[str, Any] | None = None,
        agent_port: int | None = None,
        mcp_port: int | None = None,
    ):
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.docker_image = docker_image
        self.compose_yaml = compose_yaml
        self.mcp_url = mcp_url
        self.mcp_servers = mcp_servers or []
        self.agent_url = agent_url
        self.agent = agent
        self.mcp = mcp
        self.env_file = env_file
        self.config = config or {}
        self.agent_port = agent_port
        self.mcp_port = mcp_port
        self._started = False
        self._compose = None

    # ------------------------------------------------------------------
    # Docker lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> dict[str, str]:
        """Start Docker containers if configured.

        Returns:
            Dict of service_name -> URL
        """
        if self._started:
            return self._get_urls()

        # If direct agent_url set AND no Docker services needed, skip startup
        if self.agent_url and not self.agent and not self.mcp:
            self._started = True
            return self._get_urls()

        # Build Docker compose from config
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
        from tensoreval.tools.docker import DockerCompose

        services = {}
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
        urls = {}
        if self.agent_url:
            urls["agent"] = self.agent_url
        if self.mcp_url:
            urls["mcp"] = self.mcp_url
        return urls

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
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
