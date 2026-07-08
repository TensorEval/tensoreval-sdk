"""Evaluation runner for TensorEval.

Core API:
    results = Evaluation.run(dataset, grader, agent=my_agent)
    print(results.summary())
    results.save("results.json")

The agent is a plain async callable:
    async def my_agent(query: str) -> str | AgentResult:
        return "answer"

    # Or with tool trace:
    async def my_agent(query: str) -> AgentResult:
        return AgentResult(response="answer", tool_trace=[...])
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Union

from tensoreval.dataset import Dataset
from tensoreval.grader import Grader
from tensoreval.types import AgentResult, EvalConfig, Run, Summary


# Agent can be:
# - async callable: (query: str) -> str | AgentResult
# - sync callable:  (query: str) -> str | AgentResult
AgentCallable = Union[Callable[[str], Awaitable[Any]], Callable[[str], Any]]


class EvaluationResult:
    """Results from an evaluation run.

    Attributes:
        runs: List of :class:`Run` objects (one per sample).
        dataset: The dataset that was evaluated.
        config: The evaluation config used.
    """

    def __init__(self, runs: list[Run], dataset: Dataset, config: EvalConfig):
        self.runs = runs
        self.dataset = dataset
        self.config = config

    @property
    def pass_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.reward >= self.config.pass_threshold) / len(self.runs)

    @property
    def avg_reward(self) -> float:
        if not self.runs:
            return 0.0
        return sum(r.reward for r in self.runs) / len(self.runs)

    @property
    def avg_latency_ms(self) -> float:
        if not self.runs:
            return 0.0
        return sum(r.latency_ms for r in self.runs) / len(self.runs)

    def summary(self) -> Summary:
        """Get aggregate summary."""
        return Summary(
            model=self.config.model,
            num_runs=len(self.runs),
            avg_reward=self.avg_reward,
            pass_rate=self.pass_rate,
            pass_count=sum(1 for r in self.runs if r.reward >= self.config.pass_threshold),
            fail_count=sum(1 for r in self.runs if r.reward < self.config.pass_threshold),
            total_latency_ms=sum(r.latency_ms for r in self.runs),
        )

    def per_query(self) -> list[dict[str, Any]]:
        """Get per-query details."""
        return [
            {
                "query_id": r.sample_id,
                "query": r.query,
                "reward": r.reward,
                "passed": r.reward >= self.config.pass_threshold,
                "response": r.response,
                "latency_ms": r.latency_ms,
            }
            for r in self.runs
        ]

    def save(self, path: str | Path) -> None:
        """Save results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        from tensoreval.types import Rubric, Sample

        data = {
            "config": {
                "model": self.config.model,
                "pass_threshold": self.config.pass_threshold,
                "workers": self.config.workers,
            },
            "summary": self.summary().to_dict(),
            "runs": [
                {
                    "sample_id": r.sample_id,
                    "query": r.query,
                    "answer": r.answer,
                    "response": r.response,
                    "reward": r.reward,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                    "grader_result": r.metadata.get("grader_result") if r.metadata else None,
                    "tool_trace": r.metadata.get("tool_trace", []) if r.metadata else [],
                }
                for r in self.runs
            ],
            "dataset": [
                {
                    "input": s.input,
                    "target": s.target,
                    "rubrics": [{"name": r.name, "criteria": r.criteria, "weight": r.weight} for r in s.rubrics],
                }
                for s in self.dataset
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationResult":
        """Load results from JSON."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        from tensoreval.types import Rubric, Sample

        samples = [
            Sample(
                input=s["input"],
                target=s.get("target", ""),
                rubrics=[Rubric.from_dict(r) for r in s.get("rubrics", [])],
            )
            for s in data.get("dataset", [])
        ]
        config_data = data.get("config", {})
        config = EvalConfig(
            model=config_data.get("model", "unknown"),
            pass_threshold=config_data.get("pass_threshold", 0.8),
        )
        runs = [
            Run(
                sample_id=r.get("sample_id", ""),
                query=r.get("query", ""),
                answer=r.get("answer", ""),
                response=r.get("response", ""),
                reward=r.get("reward", 0.0),
                latency_ms=r.get("latency_ms", 0.0),
                error=r.get("error"),
                metadata={
                    "grader_result": r.get("grader_result"),
                    "tool_trace": r.get("tool_trace", []),
                },
            )
            for r in data.get("runs", [])
        ]
        return cls(runs=runs, dataset=Dataset(samples), config=config)


class Evaluation:
    """Evaluation runner.

    Usage:
        async def my_agent(query: str) -> str:
            return "answer"

        results = Evaluation.run(dataset, grader, agent=my_agent)
        print(results.summary())
    """

    @staticmethod
    async def run_async(
        dataset: Dataset,
        grader: Grader | None = None,
        agent: AgentCallable | None = None,
        env: Any = None,
        config: EvalConfig | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Run evaluation asynchronously.

        Args:
            dataset: Samples with queries, answers, and rubrics.
            grader: Grader instance (default: :class:`Grader` with fallback).
            agent: Async callable ``(query: str) -> str | AgentResult``.
            env: :class:`Environment` for Docker/MCP config.
            config: Evaluation configuration.
            **kwargs: Override config fields (model, api_key, base_url, etc.).

        Returns:
            :class:`EvaluationResult` with per-query scores and summary.
        """
        if agent is None:
            raise ValueError("agent is required — pass an async callable: agent=my_agent_fn")

        # Build config
        if config is None:
            config = EvalConfig(**{k: v for k, v in kwargs.items() if k in EvalConfig.__dataclass_fields__})
        else:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        # Pull config from environment
        if env is not None:
            if config.system_prompt is None:
                config.system_prompt = env.system_prompt
            if not config.mcp_servers and getattr(env, "mcp_servers", None):
                config.mcp_servers = list(env.mcp_servers)
            if not config.mcp_servers and getattr(env, "mcp_url", None):
                config.mcp_servers = [{"name": "default", "url": env.mcp_url}]

        # Model config from env vars
        if not config.api_key:
            config.api_key = (
                os.environ.get("TENSOREVAL_MODEL_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("TENSOREVAL_API_KEY", "")
            )
        if not config.base_url:
            config.base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"

        # Default grader
        if grader is None:
            grader = Grader(
                model=config.model,
                api_key=config.api_key,
                base_url=config.base_url,
                fallback_on_error=True,
                timeout=config.timeout,
            )

        # Start Docker environment if needed
        env_started = False
        if env is not None and hasattr(env, "start") and (getattr(env, "agent", None) or getattr(env, "mcp", None)):
            try:
                await env.start()
                env_started = True
            except Exception as exc:
                import sys
                print(f"[tensoreval] warning: env.start() failed: {exc}", file=sys.stderr)

        # Build MCP registry from config + env
        mcp_registry = _build_mcp_registry(config, env)
        mcp_urls = _extract_mcp_urls(config, env)

        try:
            results = await _run_eval(dataset, grader, agent, config, mcp_registry, mcp_urls)
            _maybe_push_to_backend(results)
            return results
        finally:
            if env_started and hasattr(env, "stop"):
                try:
                    await env.stop()
                except Exception:
                    pass

    @staticmethod
    def run(
        dataset: Dataset,
        grader: Grader | None = None,
        agent: AgentCallable | None = None,
        env: Any = None,
        config: EvalConfig | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Run evaluation synchronously.

        Same args as :meth:`run_async`. Wraps the async call for sync usage.
        """
        coro = Evaluation.run_async(
            dataset=dataset, grader=grader, agent=agent,
            env=env, config=config, **kwargs,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Internal: evaluation loop
# ---------------------------------------------------------------------------

async def _run_eval(
    dataset: Dataset,
    grader: Grader,
    agent: AgentCallable,
    config: EvalConfig,
    mcp_registry: Any = None,
    mcp_urls: list[str] | None = None,
) -> EvaluationResult:
    """Run the evaluation loop with concurrent workers."""
    sem = asyncio.Semaphore(config.workers)
    mcp_urls = mcp_urls or []

    # Pre-fetch MCP tools (non-fatal if a server is down)
    mcp_tools: list[dict[str, Any]] = []
    if mcp_registry:
        try:
            await mcp_registry.list_all_tools()
            mcp_tools = mcp_registry.to_openai_tools()
        except Exception:
            pass

    async def eval_one(idx: int) -> Run:
        async with sem:
            return await _evaluate_single(idx, dataset, grader, agent, config, mcp_tools, mcp_registry, mcp_urls)

    tasks = [eval_one(i) for i in range(len(dataset))]
    runs = await asyncio.gather(*tasks)
    return EvaluationResult(list(runs), dataset, config)


async def _evaluate_single(
    idx: int,
    dataset: Dataset,
    grader: Grader,
    agent: AgentCallable,
    config: EvalConfig,
    mcp_tools: list[dict[str, Any]],
    mcp_registry: Any,
    mcp_urls: list[str],
) -> Run:
    """Evaluate a single sample: call agent → grade → return Run."""
    sample = dataset[idx]
    start_time = time.monotonic()

    try:
        # Call the agent
        raw_result = agent(sample.input)
        if asyncio.iscoroutine(raw_result):
            raw_result = await raw_result

        agent_result = AgentResult.coerce(raw_result)
        response = agent_result.response
        tool_trace = agent_result.tool_trace
        latency_ms = (time.monotonic() - start_time) * 1000

        # Build grader state
        sample_metadata = sample.metadata or {}
        state = {
            "query": sample.input,
            "answer": sample.target,
            "completion": [{"role": "assistant", "content": response}],
            "info": {
                "rubrics": [{"name": r.name, "criteria": r.criteria, "weight": r.weight} for r in sample.rubrics],
                "mcp_tools": mcp_tools,
                "tool_trace": tool_trace,
                "attachments": sample_metadata.get("attachments", []),
            },
            "tools": mcp_tools,
            "tool_registry": mcp_registry,
            "prompt": [{"role": "user", "content": sample.input}],
            "index": idx,
        }

        reward = await grader.score(state, mcp_urls=mcp_urls)

        return Run(
            sample_id=sample.id,
            query=sample.input,
            answer=sample.target,
            response=response,
            reward=reward,
            latency_ms=latency_ms,
            metadata={
                "grader_result": state.get("grader_result"),
                "tool_trace": tool_trace,
            },
        )
    except Exception as exc:
        latency_ms = (time.monotonic() - start_time) * 1000
        return Run(
            sample_id=sample.id,
            query=sample.input,
            answer=sample.target,
            response="",
            reward=0.0,
            latency_ms=latency_ms,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Internal: MCP registry construction
# ---------------------------------------------------------------------------

def _build_mcp_registry(config: EvalConfig, env: Any = None) -> Any:
    """Build an MCP registry from local tools + MCP servers."""
    from tensoreval.mcp import MCPServer, MCPRegistry

    registry = MCPRegistry()

    # Local Python tools
    for tool_fn in getattr(config, "tools", None) or []:
        if callable(tool_fn):
            registry.add_local_tool(tool_fn)

    # Collect server specs from config + env
    server_specs: list[Any] = []
    if config.mcp_servers:
        server_specs.extend(config.mcp_servers if isinstance(config.mcp_servers, list) else [config.mcp_servers])
    if env is not None and getattr(env, "mcp_servers", None):
        server_specs.extend(env.mcp_servers if isinstance(env.mcp_servers, list) else [env.mcp_servers])
    if env is not None and getattr(env, "mcp_url", None):
        server_specs.append({"name": "default", "url": env.mcp_url})

    seen_urls: set[str] = set()
    for idx, spec in enumerate(server_specs):
        server = _mcp_server_from_spec(spec, idx)
        if not server or not server.url or server.url in seen_urls:
            continue
        seen_urls.add(server.url)
        registry.add_server(server.name or f"server_{idx + 1}", server)

    if registry.local_tools or registry.servers:
        return registry
    return None


def _mcp_server_from_spec(spec: Any, idx: int) -> MCPServer | None:
    from tensoreval.mcp import MCPServer

    if isinstance(spec, MCPServer):
        return spec
    if isinstance(spec, str):
        return MCPServer(url=spec, name=f"server_{idx + 1}")
    if isinstance(spec, dict):
        url = str(spec.get("url") or spec.get("mcp_url") or "")
        if not url:
            return None
        return MCPServer(
            url=url,
            name=str(spec.get("name") or f"server_{idx + 1}"),
            auth_token=spec.get("auth_token") or spec.get("token"),
        )
    return None


def _extract_mcp_urls(config: EvalConfig, env: Any = None) -> list[str]:
    """Extract MCP URLs for grader verification."""
    urls: list[str] = []
    for s in config.mcp_servers or []:
        url = getattr(s, "url", None) or (s.get("url") if isinstance(s, dict) else None)
        if url:
            urls.append(url)
    if env is not None and getattr(env, "mcp_url", None):
        if env.mcp_url not in urls:
            urls.append(env.mcp_url)
    return urls


# ---------------------------------------------------------------------------
# Internal: backend push (optional, non-fatal)
# ---------------------------------------------------------------------------

def _maybe_push_to_backend(results: EvaluationResult) -> None:
    """Push results to TensorEval backend if TENSOREVAL_API_KEY is set.

    Failures are non-fatal — local evals never break because the backend
    is unreachable.
    """
    api_key = os.environ.get("TENSOREVAL_API_KEY")
    if not api_key:
        return

    import sys
    from tensoreval.client import TensorEvalClient, TensorEvalError

    base_url = os.environ.get("TENSOREVAL_BASE_URL", "http://localhost:4000")
    try:
        client = TensorEvalClient(api_key=api_key, base_url=base_url, timeout=3.0)
        client.ingest_evaluation(results)
    except TensorEvalError as exc:
        print(f"[tensoreval] backend push skipped: {exc}", file=sys.stderr)
    except Exception:
        pass  # Backend down — local evals must still work
