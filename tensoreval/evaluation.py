"""Evaluation runner for TensorEval.

Core API:
    results = Evaluation.run(dataset, grader, agent=my_agent, model="gpt-4o")
    print(results.summary())
    results.save("results.json")

Agent can be:
    - An Agent instance (any class extending Agent)
    - An async function (async def my_agent(query) -> str)
    - A string URL ("http://localhost:8000")
    - A model name ("gpt-4o" with api_key)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from tensoreval.datasets import Datasets
from tensoreval.graders.base import Grader
from tensoreval.types import EvalConfig, Run, Score, Summary


class EvaluationResult:
    """Results from an evaluation run."""

    def __init__(self, runs: list[Run], datasets: Datasets, config: EvalConfig):
        self.runs = runs
        self.datasets = datasets
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

    @property
    def total_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.runs)

    def summary(self) -> Summary:
        """Get aggregate summary."""
        return Summary(
            model=self.config.model,
            num_runs=len(self.runs),
            avg_reward=self.avg_reward,
            pass_rate=self.pass_rate,
            pass_count=sum(1 for r in self.runs if r.reward >= self.config.pass_threshold),
            fail_count=sum(1 for r in self.runs if r.reward < self.config.pass_threshold),
            total_latency_ms=self.total_latency_ms,
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
                    "metadata": {k: v for k, v in (r.metadata or {}).items() if k not in ("grader_result", "tool_trace")} if r.metadata else {},
                }
                for r in self.runs
            ],
            "datasets": [
                {
                    "input": s.input,
                    "target": s.target,
                    "rubrics": [{"name": r.name, "criteria": r.criteria, "weight": r.weight} for r in s.rubrics],
                }
                for s in self.datasets
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> EvaluationResult:
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
            for s in data.get("datasets", [])
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
                    **{k: v for k, v in r.get("metadata", {}).items() if k not in ("grader_result", "tool_trace")},
                },
            )
            for r in data.get("runs", [])
        ]

        return cls(runs=runs, datasets=Datasets(samples), config=config)


class Evaluation:
    """Evaluation runner.

    Usage:
        # With a function
        async def my_agent(query: str) -> str:
            return "answer"

        results = Evaluation.run(dataset, grader, agent=my_agent)

        # With an Agent class
        class MyAgent(Agent):
            async def run(self, query, context):
                return "answer"

        results = Evaluation.run(dataset, grader, agent=MyAgent())

        # With an endpoint
        results = Evaluation.run(dataset, grader, agent="http://localhost:8000")

        # With a model name
        results = Evaluation.run(dataset, grader, agent="gpt-4o", api_key="sk-...")
    """

    @staticmethod
    async def run_async(
        dataset: Datasets,
        grader: Grader | None = None,
        agent: Any = None,
        env: Any = None,
        config: EvalConfig | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Run evaluation asynchronously.

        Args:
            dataset: Samples with queries, answers, rubrics.
            grader: Scorer (default: AgentGrader).
            agent: Agent to evaluate. Can be:
                - Agent instance
                - async function (query: str) -> str
                - String URL ("http://localhost:8000")
                - Model name ("gpt-4o")
            env: Env config (system_prompt, Docker config).
            config: Evaluation configuration.
            **kwargs: Override config fields (model, api_key, base_url, etc.)

        Returns:
            EvaluationResult with per-query scores and summary.
        """
        from tensoreval.agents import Agent, Context, resolve_agent

        run_name = kwargs.get("experiment_name") or getattr(dataset, "name", "") or None

        # Build config
        if config is None:
            config = EvalConfig(**{k: v for k, v in kwargs.items() if k in EvalConfig.__dataclass_fields__})
        else:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        # Resolve env config
        if env is not None:
            if config.system_prompt is None:
                config.system_prompt = env.system_prompt
            if not config.tools and getattr(env, "tools", None):
                config.tools = list(env.tools)
            if not config.mcp_servers and getattr(env, "mcp_servers", None):
                config.mcp_servers = list(env.mcp_servers)
            if config.mcp_port is None and getattr(env, "mcp_port", None):
                config.mcp_port = env.mcp_port
            # Extract agent port from env.agent_url if not explicitly set
            if config.agent_port is None and env.agent_url:
                try:
                    config.agent_port = int(env.agent_url.rsplit(":", 1)[-1].split("/")[0])
                except (ValueError, IndexError):
                    pass

        # ── Model config from env (TENSOREVAL_MODEL_*) ──────────────
        # Separate from backend API key (TENSOREVAL_API_KEY for pushing results)
        if not config.api_key:
            config.api_key = (
                os.environ.get("TENSOREVAL_MODEL_API_KEY")
                or os.environ.get("TENSOREVAL_API_KEY")
                or os.environ.get("OPENAI_API_KEY", "")
            )
        if not config.base_url:
            config.base_url = (
                os.environ.get("TENSOREVAL_MODEL_BASE_URL")
                or os.environ.get("TENSOREVAL_BASE_URL")
                or "https://api.openai.com/v1"
            )
        if config.model == "gpt-4o":  # still default → check env
            env_model = os.environ.get("TENSOREVAL_MODEL_NAME")
            if env_model:
                config.model = env_model

        # Resolve agent — uses model config from above
        resolved_agent = resolve_agent(
            agent=agent,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            agent_port=config.agent_port,
        )

        # Default grader: Vercel AI Gateway judge with deterministic offline fallback.
        if grader is None:
            from tensoreval.graders.agent_grader import AgentGrader
            grader = AgentGrader(fallback_on_error=True)

        from tensoreval.observability import current_run, get_tracer

        tracer = get_tracer()
        trace_run = None
        if current_run() is None:
            trace_run = tracer.begin_run(
                "evaluation",
                model=config.model,
                samples=len(dataset),
                workers=config.workers,
            )

        # Start Docker if needed
        env_started = False
        if env is not None and hasattr(env, 'start') and (getattr(env, 'agent', None) or getattr(env, 'mcp', None)):
            try:
                await env.start()
                env_started = True
            except Exception as e:
                import sys
                print(f"[tensoreval] warning: env.start() failed: {e}", file=sys.stderr)

        # Build one tool registry from local Python tools plus optional MCP servers.
        mcp_registry = _build_tool_registry(config, env)
        platform_client = None
        live_run_id = None
        if os.environ.get("TENSOREVAL_API_KEY"):
            platform_client, live_run_id = _start_live_dashboard_run(config, len(dataset), name=run_name)

        try:
            results = await _run_eval(
                dataset,
                grader,
                resolved_agent,
                config,
                mcp_registry,
                platform_client=platform_client,
                live_run_id=live_run_id,
            )
            if trace_run is not None:
                trace_run.set_summary(status="ok", **results.summary().to_dict())
            if platform_client and live_run_id:
                try:
                    platform_client.complete_evaluation(live_run_id, results.summary().to_dict())
                except Exception:
                    pass
            else:
                # Push to TensorEval backend if an API key is configured
                _maybe_push_to_backend(results, trace_run)
            return results
        except Exception as e:
            if platform_client and live_run_id:
                try:
                    platform_client.complete_evaluation(live_run_id, failed=True, progress=str(e))
                except Exception:
                    pass
            if trace_run is not None:
                trace_run.set_summary(status="error", error=str(e))
            raise
        finally:
            if env_started and hasattr(env, 'stop'):
                try:
                    await env.stop()
                except Exception:
                    pass
            if trace_run is not None:
                tracer.end_run(trace_run)

    @staticmethod
    def run(
        dataset: Datasets,
        grader: Grader | None = None,
        agent: Any = None,
        env: Any = None,
        config: EvalConfig | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Run evaluation synchronously.

        Args:
            dataset: Samples with queries, answers, rubrics.
            grader: Scorer (default: AgentGrader).
            agent: Agent to evaluate (see run_async for options).
            env: Env config.
            config: Evaluation configuration.
            **kwargs: Override config fields.

        Returns:
            EvaluationResult with per-query scores and summary.
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


async def _run_eval(
    dataset: Datasets,
    grader: Grader,
    agent: Any,
    config: EvalConfig,
    mcp_registry: Any = None,
    platform_client: Any = None,
    live_run_id: str | None = None,
) -> EvaluationResult:
    """Internal: run the evaluation loop.

    If a custom grader overrides ``score_group`` and there are multiple samples,
    responses are collected first then batch-scored. Otherwise each sample is
    scored individually.
    """
    sem = asyncio.Semaphore(config.workers)

    # Pre-fetch tools once (non-fatal if an optional MCP server is down)
    mcp_tools: list[dict[str, Any]] = []
    if mcp_registry:
        try:
            await mcp_registry.list_all_tools()
            mcp_tools = mcp_registry.to_openai_tools()
        except Exception:
            pass

    base_score_group = getattr(Grader, "score_group", None)
    grader_score_group = getattr(type(grader), "score_group", None)
    use_group_scoring = grader_score_group is not base_score_group and len(dataset) > 1

    if use_group_scoring:
        # Phase 1: collect all responses in parallel
        async def collect_one(idx: int) -> dict:
            async with sem:
                return await _collect_response(idx, dataset, agent, config, mcp_tools, mcp_registry)

        collected = await asyncio.gather(*[collect_one(i) for i in range(len(dataset))])

        # Phase 2: batch-score via score_group (relative ranking)
        states = [c["state"] for c in collected]
        try:
            scores = await grader.score_group(states)
        except Exception:
            scores = [0.5] * len(collected)

        runs = [
            Run(
                sample_id=dataset[i].id,
                query=dataset[i].input,
                answer=dataset[i].target,
                response=collected[i]["response"],
                reward=scores[i] if i < len(scores) else 0.0,
                latency_ms=collected[i]["latency_ms"],
            )
            for i in range(len(dataset))
        ]
        if platform_client and live_run_id:
            for run in runs:
                try:
                    await asyncio.to_thread(platform_client.append_evaluation_result, live_run_id, run, len(dataset))
                except Exception:
                    pass
        return EvaluationResult(runs, dataset, config)

    # Standard per-sample scoring
    async def eval_one(idx: int) -> Run:
        async with sem:
            run = await _evaluate_single(idx, dataset, grader, agent, config, mcp_tools, mcp_registry)
            if platform_client and live_run_id:
                try:
                    await asyncio.to_thread(platform_client.append_evaluation_result, live_run_id, run, len(dataset))
                except Exception:
                    pass
            return run

    tasks = [eval_one(i) for i in range(len(dataset))]
    runs = await asyncio.gather(*tasks)
    return EvaluationResult(list(runs), dataset, config)


async def _collect_response(
    idx: int,
    dataset: Datasets,
    agent: Any,
    config: EvalConfig,
    mcp_tools: list[dict[str, Any]] | None = None,
    mcp_registry: Any = None,
) -> dict[str, Any]:
    """Collect an agent response + grader state without scoring. Used for group scoring."""
    from tensoreval.agents import Context

    sample = dataset[idx]
    start_time = time.monotonic()

    context = Context(
        query=sample.input,
        system_prompt=config.system_prompt,
        tools=mcp_tools or [],
        metadata={"sample_id": sample.id, "index": idx},
        mcp_registry=mcp_registry,
    )

    response = await agent.run(sample.input, context)
    latency_ms = (time.monotonic() - start_time) * 1000

    state = {
        "query": sample.input,
        "answer": sample.target,
        "completion": [{"role": "assistant", "content": response}],
        "info": {
            "rubrics": [{"name": r.name, "criteria": r.criteria, "weight": r.weight} for r in sample.rubrics],
            "mcp_tools": mcp_tools or [],
        },
        "tools": mcp_tools or [],
        "tool_registry": mcp_registry,
        "prompt": [{"role": "user", "content": sample.input}],
        "index": idx,
    }

    return {"response": response, "latency_ms": latency_ms, "state": state}


async def _evaluate_single(
    idx: int,
    dataset: Datasets,
    grader: Grader,
    agent: Any,
    config: EvalConfig,
    mcp_tools: list[dict[str, Any]] | None = None,
    mcp_registry: Any = None,
) -> Run:
    """Internal: evaluate a single sample."""
    from tensoreval.agents import Context

    sample = dataset[idx]
    start_time = time.monotonic()

    try:
        # Build context — include MCP tools so agents can call them
        context = Context(
            query=sample.input,
            system_prompt=config.system_prompt,
            tools=mcp_tools or [],
            metadata={"sample_id": sample.id, "index": idx},
            mcp_registry=mcp_registry,
        )

        from tensoreval.observability import observe

        @observe("agent.run", kind="agent")
        async def call_agent(input_text: str, model: str) -> str:
            return await agent.run(input_text, context)

        @observe("grader.score", kind="grader")
        async def call_grader(input_text: str, model: str, state: dict[str, Any], **kw: Any) -> float:
            return await grader.score(state, **kw)

        # Get response from agent
        response = await call_agent(input_text=sample.input, model=config.model)
        latency_ms = (time.monotonic() - start_time) * 1000
        tool_trace = context.metadata.get("tool_trace", [])

        # Build state for grader
        sample_metadata = sample.metadata or {}
        state = {
            "query": sample.input,
            "answer": sample.target,
            "completion": [{"role": "assistant", "content": response}],
            "info": {
                "rubrics": [{"name": r.name, "criteria": r.criteria, "weight": r.weight} for r in sample.rubrics],
                "mcp_tools": mcp_tools or [],
                "tool_trace": tool_trace,
                "difficulty": sample_metadata.get("difficulty", ""),
                "attachments": sample_metadata.get("attachments", []),
            },
            "tools": mcp_tools or [],
            "tool_registry": mcp_registry,
            "prompt": [{"role": "user", "content": sample.input}],
            "index": idx,
        }

        # Pass MCP URLs to grader for active verification if available
        mcp_urls = []
        if hasattr(config, 'mcp_servers') and config.mcp_servers:
            for s in config.mcp_servers:
                url = getattr(s, 'url', None) or (s.get('url') if isinstance(s, dict) else None)
                if url:
                    mcp_urls.append(url)

        reward = await call_grader(input_text=response, model=config.model, state=state, mcp_urls=mcp_urls)

        return Run(
            sample_id=sample.id,
            query=sample.input,
            answer=sample.target,
            response=response,
            reward=reward,
            latency_ms=latency_ms,
            metadata={
                "grader_result": state.get("grader_result"),
                "category": sample.metadata.get("category", "sdk") if sample.metadata else "sdk",
                "tool_trace": tool_trace,
            },
        )

    except Exception as e:
        latency_ms = (time.monotonic() - start_time) * 1000
        return Run(
            sample_id=sample.id,
            query=sample.input,
            answer=sample.target,
            response="",
            reward=0.0,
            latency_ms=latency_ms,
            error=str(e),
            metadata={"category": sample.metadata.get("category", "sdk") if sample.metadata else "sdk"},
        )


def _start_live_dashboard_run(config: EvalConfig, total_count: int, name: str | None = None) -> tuple[Any, str | None]:
    """Start a live dashboard run when TensorEval backend credentials exist."""
    import sys
    from tensoreval.client import TensorEvalClient, TensorEvalError

    try:
        client = TensorEvalClient(timeout=3.0)
        response = client.start_evaluation(model=config.model, total_count=total_count, name=name)
        return client, response.get("evaluation_run_id")
    except (TensorEvalError, Exception) as exc:
        print(f"[tensoreval] live dashboard updates skipped: {exc}", file=sys.stderr)
        return None, None


def _build_tool_registry(config: EvalConfig, env: Any = None) -> Any:
    """Build a registry for local Python tools and optional MCP servers."""
    from tensoreval.tools.mcp import MCPServer, MCPToolRegistry

    registry = MCPToolRegistry()

    for tool in config.tools or []:
        if callable(tool):
            registry.add_local_tool(tool)

    server_specs: list[Any] = []
    if config.mcp_servers:
        server_specs.extend(config.mcp_servers if isinstance(config.mcp_servers, list) else [config.mcp_servers])
    if env is not None and getattr(env, "mcp_servers", None):
        server_specs.extend(env.mcp_servers if isinstance(env.mcp_servers, list) else [env.mcp_servers])
    if env is not None and getattr(env, "mcp_url", None):
        server_specs.append({"name": "default", "url": env.mcp_url})
    if config.mcp_port:
        server_specs.append({"name": "default", "url": f"http://localhost:{config.mcp_port}/mcp"})

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


def _mcp_server_from_spec(spec: Any, idx: int) -> Any:
    from tensoreval.tools.mcp import MCPServer

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
            transport=str(spec.get("transport") or "streamable-http"),
            auth_token=spec.get("auth_token") or spec.get("token"),
        )
    return None


def _maybe_push_to_backend(results: EvaluationResult, trace_run: Any) -> None:
    """Post results + traces to the TensorEval backend if TENSOREVAL_API_KEY is set.

    Failures are non-fatal (printed to stderr) so local evals never break
    because the backend is unreachable. Uses a short timeout (3s) so a down
    backend doesn't stall the evaluation.
    """
    api_key = os.environ.get("TENSOREVAL_API_KEY")
    if not api_key:
        return

    from tensoreval.client import TensorEvalClient, TensorEvalError

    base_url = os.environ.get("TENSOREVAL_BASE_URL", "http://localhost:4000")
    import sys

    client: TensorEvalClient | None = None
    try:
        client = TensorEvalClient(api_key=api_key, base_url=base_url, timeout=3.0)
        client.ingest_evaluation(results)
    except TensorEvalError as e:
        print(f"[tensoreval] backend push skipped: {e}", file=sys.stderr)
    except Exception:
        client = None  # Backend down/unreachable — local evals must still work

    # Push trace events if we have a run context with spans
    if client and trace_run is not None and trace_run.spans:
        events = [s.to_dict() for s in trace_run.spans]
        try:
            client.ingest_trace(trace_run.name, events)
        except Exception:
            pass  # Non-fatal
