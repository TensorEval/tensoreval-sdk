"""Env — Environment configuration and lifecycle manager.

Handles:
- System prompt
- Docker container management (start/stop)
- API key injection
- MCP server connection
- Agent endpoint connection

Usage:
    # Simple (no Docker)
    env = te.Env.from_dict({"system_prompt": "..."})

    # With Docker
    env = te.Env.from_dict({
        "system_prompt": "...",
        "agent": {"image": "python:3.12-slim", "port": 8000, "env": {"KEY": "val"}},
        "mcp": {"image": "node:18-slim", "port": 9000},
        "env_file": ".env",
    })

    # With direct URLs (no Docker)
    env = te.Env.from_dict({
        "system_prompt": "...",
        "agent_url": "http://localhost:8000",
        "mcp_url": "http://localhost:9000/mcp",
    })
"""

import os
from pathlib import Path
from typing import Any, Callable


class Env:
    """Environment configuration and lifecycle manager."""

    def __init__(
        self,
        system_prompt: str | None = None,
        tools: list[Callable] | None = None,
        docker_image: str | None = None,
        dockerfile: str | None = None,
        compose_yaml: str | None = None,
        mcp_url: str | None = None,
        agent_url: str | None = None,
        agent: dict[str, Any] | None = None,
        mcp: dict[str, Any] | None = None,
        env_file: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.docker_image = docker_image
        self.dockerfile = dockerfile
        self.compose_yaml = compose_yaml
        self.mcp_url = mcp_url
        self.agent_url = agent_url
        self.agent = agent
        self.mcp = mcp
        self.env_file = env_file
        self.config = config or {}
        self._compose = None
        self._started = False

    @classmethod
    def load_from_file(cls, path: str | Path) -> "Env":
        """Load environment from YAML, JSON, or Python file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return cls._load_yaml(path)
        elif suffix == ".json":
            return cls._load_json(path)
        elif suffix == ".py":
            return cls._load_python(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Env":
        """Create from dict config."""
        return cls(
            system_prompt=config.get("system_prompt"),
            tools=config.get("tools"),
            docker_image=config.get("docker_image"),
            dockerfile=config.get("dockerfile"),
            compose_yaml=config.get("compose_yaml"),
            mcp_url=config.get("mcp_url"),
            agent_url=config.get("agent_url"),
            agent=config.get("agent"),
            mcp=config.get("mcp"),
            env_file=config.get("env_file"),
            config=config,
        )

    @classmethod
    def _load_yaml(cls, path: Path) -> "Env":
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required. Install: pip install pyyaml")
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def _load_json(cls, path: Path) -> "Env":
        import json
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def _load_python(cls, path: Path) -> "Env":
        import importlib.util
        spec = importlib.util.spec_from_file_location("env_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "load_environment"):
            result = module.load_environment()
            if isinstance(result, cls):
                return result
            elif isinstance(result, dict):
                return cls.from_dict(result)
        raise ValueError(f"Python file must define load_environment(): {path}")

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

        # If direct URLs set, no Docker needed
        if self.agent_url and not self.agent:
            self._started = True
            return self._get_urls()

        # Build Docker compose from config
        if self.agent or self.mcp:
            from tensoreval.tools.docker import DockerCompose

            services = {}
            if self.agent:
                services["agent"] = self.agent
            if self.mcp:
                services["mcp-server"] = self.mcp

            self._compose = DockerCompose(
                services=services,
                compose_yaml=self.compose_yaml,
                env_file=self.env_file,
            )

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
        return f"Env({', '.join(parts)})"
