"""Runtime environment definitions for TensorEval."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Env:
    """Runtime environment for an evaluation — where the agent lives and what tools it can reach.

    Construct via ``from_endpoint`` (remote HTTP), ``from_dockerfile`` (local container),
    or ``from_yaml`` (config file).  ``from_registry`` is reserved for future use.
    """
    kind: str
    agent_url: str | None = None
    mcp_urls: list[str] = field(default_factory=list)
    dockerfile: str | None = None
    agent_port: int | None = None
    mcp_ports: list[int] = field(default_factory=list)
    registry_name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def supports_parallel_workers(self) -> bool:
        return self.kind == "dockerfile"

    @classmethod
    def from_endpoint(
        cls,
        agent_url: str,
        mcp_urls: str | list[str] | None = None,
        **config: Any,
    ) -> "Env":
        return cls(
            kind="endpoint",
            agent_url=agent_url,
            mcp_urls=_as_list(mcp_urls),
            config=config,
        )

    @classmethod
    def from_dockerfile(
        cls,
        path: str = "Dockerfile",
        agent_port: int = 8000,
        mcp_ports: int | list[int] | None = None,
        **config: Any,
    ) -> "Env":
        return cls(
            kind="dockerfile",
            dockerfile=path,
            agent_port=agent_port,
            mcp_ports=[int(p) for p in _as_list(mcp_ports)],
            config=config,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Env":
        path = Path(path)
        try:
            import yaml
            data = yaml.safe_load(path.read_text()) or {}
        except ImportError:
            data = json.loads(path.read_text())

        kind = data.pop("type", data.pop("kind", "endpoint"))
        if kind == "endpoint":
            return cls.from_endpoint(**data)
        if kind == "dockerfile":
            dockerfile = data.pop("path", data.pop("dockerfile", "Dockerfile"))
            return cls.from_dockerfile(path=dockerfile, **data)
        raise ValueError(f"Unsupported env type: {kind}")

    @classmethod
    def from_registry(cls, name: str, **config: Any) -> "Env":
        raise NotImplementedError(f"Registry env '{name}' is not available in the local SDK yet")

    def mcp_servers(self) -> list[dict[str, str]]:
        return [{"name": f"mcp_{i}", "url": url} for i, url in enumerate(self.mcp_urls)]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]
