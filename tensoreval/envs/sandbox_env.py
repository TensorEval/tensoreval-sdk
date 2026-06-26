"""Docker sandbox environment for TensorEval.

Based on Inspect AI's Docker sandbox pattern (MIT License).
Provides per-sample container isolation using Docker Compose.

Key patterns adopted from Inspect AI:
- Per-sample isolation via unique Compose project names
- compose.yaml auto-generation when not provided
- exec() for running commands inside containers
- write_file()/read_file() for file I/O
- Proper cleanup on exit

Usage:
    env = DockerSandboxEnv(
        rubric=my_rubric,
        compose_yaml="compose.yaml",  # or None for auto-generated
        system_prompt="You are a coding agent.",
    )
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from tensoreval.core.types import (
    AssistantMessage,
    Messages,
    RolloutInput,
    SamplingArgs,
    State,
    SystemMessage,
    TrajectoryStep,
    UserMessage,
)
from tensoreval.envs.environment import Environment


# ---------------------------------------------------------------------------
# Default compose.yaml for auto-generation
# ---------------------------------------------------------------------------

DEFAULT_COMPOSE_YAML = """\
services:
  default:
    image: ubuntu:24.04
    command: tail -f /dev/null
    init: true
    stop_grace_period: 1s
"""


# ---------------------------------------------------------------------------
# ComposeProject — manages a Docker Compose project
# ---------------------------------------------------------------------------

class ComposeProject:
    """Manages a Docker Compose project for sandbox isolation."""

    def __init__(self, name: str, config_path: str | None = None, env: dict[str, str] | None = None):
        self.name = name
        self.config_path = config_path
        self.env = env or {}

    async def up(self, timeout: int = 600) -> None:
        """Start the compose project."""
        cmd = ["docker", "compose", "--project-name", self.name]
        if self.config_path:
            cmd.extend(["-f", self.config_path])
        cmd.extend(["up", "--detach", "--wait", "--wait-timeout", str(timeout)])

        env = {**os.environ, **self.env}
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout + 10)
        if proc.returncode != 0:
            raise RuntimeError(f"docker compose up failed: {stderr.decode()}")

    async def down(self) -> None:
        """Destroy the compose project and all its resources."""
        cmd = ["docker", "compose", "--project-name", self.name, "down", "--volumes"]
        if self.config_path:
            cmd.extend(["-f", self.config_path])

        env = {**os.environ, **self.env}
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        await asyncio.wait_for(proc.communicate(), timeout=120)

    async def exec(
        self,
        service: str,
        cmd: list[str],
        workdir: str | None = None,
        timeout: int | None = None,
        input_data: str | bytes | None = None,
    ) -> tuple[str, str, int]:
        """Execute a command inside a running service container."""
        exec_cmd = ["docker", "compose", "--project-name", self.name]
        if self.config_path:
            exec_cmd.extend(["-f", self.config_path])
        exec_cmd.append("exec")
        if workdir:
            exec_cmd.extend(["--workdir", workdir])
        exec_cmd.append(service)
        exec_cmd.extend(cmd)

        env = {**os.environ, **self.env}
        host_timeout = (timeout + 10) if timeout else None

        proc = await asyncio.create_subprocess_exec(
            *exec_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=input_data if isinstance(input_data, bytes) else input_data.encode() if input_data else None),
            timeout=host_timeout,
        )
        return stdout.decode(), stderr.decode(), proc.returncode or 0

    async def write_file(self, service: str, path: str, contents: str | bytes) -> None:
        """Write a file into a container."""
        parent = str(Path(path).parent)
        if parent != ".":
            await self.exec(service, ["mkdir", "-p", parent])

        if isinstance(contents, str):
            await self.exec(
                service,
                ["sh", "-e", "-c", 'tee -- "$1" > /dev/null', "write_file", path],
                input_data=contents,
            )
        else:
            import base64
            b64 = base64.b64encode(contents).decode("US-ASCII")
            await self.exec(
                service,
                ["sh", "-e", "-c", 'base64 -d | tee -- "$1" > /dev/null', "write_file", path],
                input_data=b64,
            )

    async def read_file(self, service: str, path: str) -> str:
        """Read a file from a container."""
        stdout, _, rc = await self.exec(service, ["cat", path])
        if rc != 0:
            raise FileNotFoundError(f"File not found: {path}")
        return stdout


# ---------------------------------------------------------------------------
# DockerSandboxEnv — environment with Docker sandbox support
# ---------------------------------------------------------------------------

class DockerSandboxEnv(Environment):
    """Environment with Docker sandbox for code execution.

    Each evaluation sample gets its own Docker Compose project,
    providing complete isolation between samples.

    Usage:
        # With explicit compose.yaml
        env = DockerSandboxEnv(
            rubric=my_rubric,
            compose_yaml="path/to/compose.yaml",
        )

        # With auto-generated compose (default ubuntu image)
        env = DockerSandboxEnv(
            rubric=my_rubric,
            image="python:3.12-slim",
        )

        # With custom Dockerfile
        env = DockerSandboxEnv(
            rubric=my_rubric,
            dockerfile="Dockerfile",
        )
    """

    def __init__(
        self,
        compose_yaml: str | None = None,
        dockerfile: str | None = None,
        image: str = "ubuntu:24.04",
        service_name: str = "default",
        workdir: str = "/workspace",
        **kwargs,
    ):
        super__(**kwargs)
        self.compose_yaml = compose_yaml
        self.dockerfile = dockerfile
        self.image = image
        self.service_name = service_name
        self.workdir = workdir
        self._project: ComposeProject | None = None

    def _generate_compose_yaml(self) -> str:
        """Generate a compose.yaml based on configuration."""
        if self.compose_yaml and os.path.exists(self.compose_yaml):
            return Path(self.compose_yaml).read_text()

        if self.dockerfile:
            return f"""\
services:
  default:
    build:
      context: "."
      dockerfile: "{self.dockerfile}"
    command: tail -f /dev/null
    init: true
    stop_grace_period: 1s
"""

        return f"""\
services:
  default:
    image: {self.image}
    command: tail -f /dev/null
    init: true
    stop_grace_period: 1s
"""

    async def _setup_sandbox(self) -> ComposeProject:
        """Create and start a Docker Compose project for this evaluation."""
        project_name = f"tensoreval-{uuid.uuid4().hex[:8]}"

        # Write compose.yaml to temp directory
        compose_content = self._generate_compose_yaml()
        tmpdir = tempfile.mkdtemp(prefix="tensoreval-")
        compose_path = os.path.join(tmpdir, "compose.yaml")
        with open(compose_path, "w") as f:
            f.write(compose_content)

        project = ComposeProject(name=project_name, config_path=compose_path)
        await project.up()
        self._project = project
        return project

    async def _cleanup_sandbox(self) -> None:
        """Destroy the Docker Compose project."""
        if self._project:
            try:
                await self._project.down()
            except Exception as e:
                self.logger.warning(f"Failed to cleanup sandbox: {e}")
            self._project = None

    async def rollout(
        self,
        input: RolloutInput,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """Run a rollout with Docker sandbox support.

        The agent can use tools to execute commands inside the container.
        """
        state = self._create_state(input)
        state["timing"].generation.start = time.time()

        # Build prompt
        prompt = state.get("prompt", [])
        if isinstance(prompt, str):
            prompt = [UserMessage(content=prompt)]
        if self.system_prompt:
            prompt = [SystemMessage(content=self.system_prompt)] + prompt

        # Set up sandbox
        try:
            project = await self._setup_sandbox()
            state["sandbox_project"] = project
            state["sandbox_service"] = self.service_name
            state["sandbox_workdir"] = self.workdir
        except Exception as e:
            state["error"] = e
            state["is_completed"] = True
            state["timing"].generation.end = time.time()
            return state

        # Call model
        try:
            from tensoreval.envs.singleturn_env import _call_anthropic, _call_openai, _detect_api_type, _messages_to_dicts

            messages = _messages_to_dicts(prompt)
            api_type = _detect_api_type(base_url)
            resolved_key = api_key or "dummy"
            resolved_url = base_url or "https://api.openai.com/v1"

            merged_args = {**self.sampling_args, **(sampling_args or {})}
            max_tokens = merged_args.get("max_tokens", 1024)
            temperature = merged_args.get("temperature", 0.7)

            if api_type == "anthropic":
                content = await _call_anthropic(messages, model, resolved_key, resolved_url, max_tokens, temperature)
            else:
                content = await _call_openai(messages, model, resolved_key, resolved_url, max_tokens, temperature)

            completion = [AssistantMessage(content=content)]
            state["completion"] = completion
            state["trajectory"] = [TrajectoryStep(
                prompt=prompt,
                completion=completion,
                trajectory_id=state["trajectory_id"],
            )]
        except Exception as e:
            state["error"] = e
            state["is_completed"] = True

        # Cleanup sandbox
        await self._cleanup_sandbox()

        state["timing"].generation.end = time.time()
        state["is_completed"] = True
        return state

    async def exec_in_sandbox(self, cmd: list[str], timeout: int = 30) -> tuple[str, str, int]:
        """Execute a command in the sandbox container."""
        if not self._project:
            raise RuntimeError("No sandbox active. Call rollout() first.")
        return await self._project.exec(self.service_name, cmd, workdir=self.workdir, timeout=timeout)

    async def write_to_sandbox(self, path: str, contents: str | bytes) -> None:
        """Write a file to the sandbox container."""
        if not self._project:
            raise RuntimeError("No sandbox active. Call rollout() first.")
        await self._project.write_file(self.service_name, path, contents)

    async def read_from_sandbox(self, path: str) -> str:
        """Read a file from the sandbox container."""
        if not self._project:
            raise RuntimeError("No sandbox active. Call rollout() first.")
        return await self._project.read_file(self.service_name, path)


# ---------------------------------------------------------------------------
# SandboxToolEnv — ToolEnv with Docker sandbox for code execution
# ---------------------------------------------------------------------------

class SandboxToolEnv(Environment):
    """Environment with tools that execute inside a Docker sandbox.

    The model can call tools (bash, python) that execute inside
    an isolated Docker container.

    Usage:
        def run_code(code: str, sandbox) -> str:
            # Execute code inside the sandbox
            ...

        env = SandboxToolEnv(
            tools=[run_code],
            rubric=my_rubric,
            image="python:3.12-slim",
        )
    """

    def __init__(
        self,
        tools: list[Any] | None = None,
        image: str = "python:3.12-slim",
        max_turns: int = 10,
        **kwargs,
    ):
        super__(max_turns=max_turns, **kwargs)
        self.tools = tools or []
        self.image = image

    async def rollout(
        self,
        input: RolloutInput,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """Run a multi-turn rollout with sandboxed tool execution."""
        from tensoreval.envs.multiturn_env import MultiTurnEnv

        # Create a MultiTurnEnv with tools
        multiturn = MultiTurnEnv(
            max_turns=self.max_turns,
            rubric=self.rubric,
            system_prompt=self.system_prompt,
        )

        # Set up sandbox for tool execution
        project_name = f"tensoreval-{uuid.uuid4().hex[:8]}"
        compose_content = f"""\
services:
  default:
    image: {self.image}
    command: tail -f /dev/null
    init: true
    stop_grace_period: 1s
"""
        tmpdir = tempfile.mkdtemp(prefix="tensoreval-")
        compose_path = os.path.join(tmpdir, "compose.yaml")
        with open(compose_path, "w") as f:
            f.write(compose_content)

        project = ComposeProject(name=project_name, config_path=compose_path)
        await project.up()

        try:
            # Inject sandbox into tools
            async def sandbox_exec(command: str) -> str:
                stdout, stderr, rc = await project.exec("default", ["bash", "-c", command], timeout=30)
                if rc != 0:
                    return f"Error (exit {rc}): {stderr}"
                return stdout

            # Add sandbox_exec as a tool
            sandbox_exec.__name__ = "bash"
            sandbox_exec.__doc__ = "Execute a bash command in the sandbox container"

            multiturn.tools = [sandbox_exec] + self.tools
            multiturn.tool_map = {t.__name__: t for t in multiturn.tools}

            state = await multiturn.rollout(input, model, api_key, base_url, sampling_args)
        finally:
            await project.down()

        return state
