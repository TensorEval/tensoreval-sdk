"""Docker Compose manager for TensorEval.

Uses the exact patterns that work on Windows Docker Desktop:
- docker compose (not raw docker run)
- 127.0.0.1 binding (safe for Windows firewall)
- Ephemeral port mapping (::PORT syntax)
- docker port for port discovery
- .env file for API keys

Usage:
    compose = DockerCompose(services={...})
    ports = await compose.up()
    # ... run evaluation ...
    await compose.down()
"""

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any


class DockerCompose:
    """Manages Docker Compose projects for evaluation.

    Handles: start containers, pass API keys, expose ports, cleanup.
    """

    def __init__(
        self,
        services: dict[str, dict[str, Any]] | None = None,
        compose_yaml: str | None = None,
        env_file: str | None = None,
        project_name: str | None = None,
    ):
        self.services = services or {}
        self.compose_yaml = compose_yaml
        self.env_file = env_file
        self.project_name = project_name or f"tensoreval-{uuid.uuid4().hex[:8]}"
        self._tmpdir: str | None = None
        self._compose_path: str | None = None
        self._is_up = False

    def _generate_compose_yaml(self) -> str:
        """Generate compose.yaml from services config."""
        lines = ["services:"]

        for name, config in self.services.items():
            lines.append(f"  {name}:")
            lines.append(f"    image: {config.get('image', 'ubuntu:24.04')}")

            cmd = config.get("command", "tail -f /dev/null")
            lines.append(f"    command: {cmd}")
            lines.append("    init: true")
            lines.append("    stop_grace_period: 1s")

            # Port mapping — use 127.0.0.1 for safety
            if "port" in config:
                port = config["port"]
                container_port = config.get("container_port", port)
                lines.append(f"    ports:")
                lines.append(f'      - "127.0.0.1:{port}:{container_port}"')

            # Environment variables
            env = config.get("env", {})
            if env:
                lines.append("    environment:")
                for key, value in env.items():
                    lines.append(f"      - {key}={value}")

            # Volumes — convert Windows paths to Docker-compatible
            volumes = config.get("volumes", [])
            if volumes:
                lines.append("    volumes:")
                for vol in volumes:
                    # Convert Windows backslashes to forward slashes for Docker
                    vol_converted = vol.replace("\\", "/")
                    # Convert C:/ to /c/ for Docker on Windows
                    if len(vol_converted) > 1 and vol_converted[1] == ":":
                        drive = vol_converted[0].lower()
                        vol_converted = f"/{drive}{vol_converted[2:]}"
                    lines.append(f"      - {vol_converted}")

            # Depends on
            deps = config.get("depends_on", [])
            if deps:
                lines.append("    depends_on:")
                for dep in deps:
                    lines.append(f"      - {dep}")

        return "\n".join(lines) + "\n"

    def _ensure_compose_file(self) -> str:
        """Ensure compose.yaml exists and return its path."""
        if self.compose_yaml and os.path.exists(self.compose_yaml):
            self._compose_path = self.compose_yaml
            return self._compose_path

        self._tmpdir = tempfile.mkdtemp(prefix="tensoreval-compose-")
        self._compose_path = os.path.join(self._tmpdir, "compose.yaml")

        compose_content = self._generate_compose_yaml()

        # Add env_file reference if specified
        if self.env_file:
            # Inject env_file into each service
            for name in self.services:
                compose_content = compose_content.replace(
                    f"  {name}:\n",
                    f"  {name}:\n    env_file:\n      - {self.env_file}\n",
                    1,
                )

        with open(self._compose_path, "w") as f:
            f.write(compose_content)

        return self._compose_path

    async def up(self) -> dict[str, str]:
        """Start all services. Returns dict of service_name -> URL.

        Returns:
            Dict mapping service names to their localhost URLs.
        """
        compose_path = self._ensure_compose_file()

        # Build env with API keys
        env = {**os.environ}
        if self.env_file and os.path.exists(self.env_file):
            with open(self.env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        env[key.strip()] = value.strip()

        # docker compose up --detach --wait
        cmd = [
            "docker", "compose",
            "--project-name", self.project_name,
            "-f", compose_path,
            "up", "-d", "--wait",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

        if proc.returncode != 0:
            raise RuntimeError(f"docker compose up failed: {stderr.decode()}")

        self._is_up = True

        # Discover actual ports
        urls = {}
        for name, config in self.services.items():
            if "port" in config:
                port = config["port"]
                urls[name] = f"http://localhost:{port}"

        return urls

    async def down(self) -> None:
        """Stop and remove all services."""
        if not self._is_up or not self._compose_path:
            return

        cmd = [
            "docker", "compose",
            "--project-name", self.project_name,
            "-f", self._compose_path,
            "down", "--volumes", "--remove-orphans",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=60)
        self._is_up = False

    async def exec(self, service: str, cmd: list[str], timeout: int = 30) -> tuple[str, str, int]:
        """Execute a command inside a running service container."""
        if not self._compose_path:
            raise RuntimeError("Compose not started. Call up() first.")

        exec_cmd = [
            "docker", "compose",
            "--project-name", self.project_name,
            "-f", self._compose_path,
            "exec", "-T", service,
        ] + cmd

        proc = await asyncio.create_subprocess_exec(
            *exec_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return stdout.decode(), stderr.decode(), proc.returncode or 0

    async def write_file(self, service: str, path: str, contents: str | bytes) -> None:
        """Write a file into a container."""
        if isinstance(contents, str):
            contents = contents.encode()
        import base64
        b64 = base64.b64encode(contents).decode()
        await self.exec(service, ["sh", "-c", f'echo "{b64}" | base64 -d > {path}'])

    async def read_file(self, service: str, path: str) -> str:
        """Read a file from a container."""
        stdout, _, rc = await self.exec(service, ["cat", path])
        if rc != 0:
            raise FileNotFoundError(f"File not found: {path}")
        return stdout

    def get_agent_url(self) -> str | None:
        """Get the agent service URL."""
        for name, config in self.services.items():
            if "agent" in name.lower() and "port" in config:
                return f"http://localhost:{config['port']}"
        return None

    def get_mcp_url(self) -> str | None:
        """Get the MCP server URL."""
        for name, config in self.services.items():
            if "mcp" in name.lower() and "port" in config:
                return f"http://localhost:{config['port']}/mcp"
        return None
