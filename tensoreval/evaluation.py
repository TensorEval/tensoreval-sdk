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
from typing import Any, Callable, Awaitable

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

    def summary(self) -> Summary:
        """Get aggregate summary."""
        return Summary(
            model=self.config.model,
            num_runs=len(self.runs),
            avg_reward=self.avg_reward,
            pass_rate=self.pass_rate,
            pass_count=sum(1 for r in self.runs if r.reward >= self.config.pass_threshold),
            fail_count=sum(1 for r in self.runs if r.reward < self.config.pass_threshold),
            total_latency_ms=self.avg_latency_ms,
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
            grader: Scorer (default: RubricGrader).
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
            # Extract agent port from env.agent_url if not explicitly set
            if config.agent_port is None and env.agent_url:
                try:
                    config.agent_port = int(env.agent_url.rsplit(":", 1)[-1].split("/")[0])
                except (ValueError, IndexError):
                    pass

        # Resolve agent
        resolved_agent = resolve_agent(
            agent=agent,
            model=config.model,
            api_key=config.api_key or os.environ.get("TENSOREVAL_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            base_url=config.base_url or os.environ.get("TENSOREVAL_BASE_URL", ""),
            agent_port=config.agent_port,
        )

        # Default grader
        if grader is None:
            from tensoreval.graders.rubric_grader import RubricGrader
            grader = RubricGrader()

        # Resolve API key
        if not config.api_key:
            config.api_key = os.environ.get("TENSOREVAL_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        if not config.base_url:
            config.base_url = os.environ.get("TENSOREVAL_BASE_URL", "https://api.openai.com/v1")

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
        if env is not None and hasattr(env, 'start') and hasattr(env, 'agent') and env.agent:
            try:
                await env.start()
                env_started = True
            except Exception:
                pass

        # Build MCP tool registry from env or config
        mcp_registry = None
        mcp_url = None
        if env is not None and getattr(env, 'mcp_url', None):
            mcp_url = env.mcp_url
        elif config.mcp_port:
            mcp_url = f"http://localhost:{config.mcp_port}/mcp"

        if mcp_url:
            from tensoreval.tools.mcp import MCPServer, MCPToolRegistry
            mcp_registry = MCPToolRegistry()
            mcp_registry.add_server("default", MCPServer(url=mcp_url))

        try:
            results = await _run_eval(dataset, grader, resolved_agent, config, mcp_registry)
            if trace_run is not None:
                trace_run.set_summary(status="ok", **results.summary().to_dict())
            # Push to TensorEval backend if an API key is configured
            _maybe_push_to_backend(results, trace_run)
            return results
        except Exception as e:
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
            grader: Scorer (default: RubricGrader).
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
    agent: Any,  # Agent instance
    config: EvalConfig,
    mcp_registry: Any = None,
) -> EvaluationResult:
    """Internal: run the evaluation loop."""
    sem = asyncio.Semaphore(config.workers)

    # Pre-fetch MCP tools once (non-fatal if server is down)
    mcp_tools: list[dict[str, Any]] = []
    if mcp_registry:
        try:
            await mcp_registry.list_all_tools()
            mcp_tools = mcp_registry.to_openai_tools()
        except Exception:
            pass  # MCP server down — eval continues without tools

    async def eval_one(idx: int) -> Run:
        async with sem:
            return await _evaluate_single(idx, dataset, grader, agent, config, mcp_tools, mcp_registry)

    tasks = [eval_one(i) for i in range(len(dataset))]
    runs = await asyncio.gather(*tasks)

    return EvaluationResult(list(runs), dataset, config)


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
        async def call_grader(input_text: str, model: str, state: dict[str, Any]) -> float:
            return await grader.score(state)

        # Get response from agent
        response = await call_agent(input_text=sample.input, model=config.model)
        latency_ms = (time.monotonic() - start_time) * 1000

        # Build state for grader
        state = {
            "query": sample.input,
            "answer": sample.target,
            "completion": [{"role": "assistant", "content": response}],
            "info": {"rubrics": [{"name": r.name, "criteria": r.criteria, "weight": r.weight} for r in sample.rubrics]},
            "prompt": [{"role": "user", "content": sample.input}],
            "index": idx,
        }

        reward = await call_grader(input_text=response, model=config.model, state=state)

        return Run(
            sample_id=sample.id,
            query=sample.input,
            answer=sample.target,
            response=response,
            reward=reward,
            latency_ms=latency_ms,
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
        )


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

    # Push eval results (short timeout — backend being down must not block)
    try:
        client = TensorEvalClient(api_key=api_key, base_url=base_url, timeout=3.0)
        client.ingest_evaluation(results)
    except TensorEvalError as e:
        print(f"[tensoreval] backend push skipped: {e}", file=sys.stderr)
    except Exception:
        pass  # Backend down/unreachable — local evals must still work

    # Push trace events if we have a run context with spans
    if trace_run is not None and trace_run.spans:
        events = [s.to_dict() for s in trace_run.spans]
        try:
            client = TensorEvalClient(api_key=api_key, base_url=base_url, timeout=3.0)
            client.ingest_trace(trace_run.name, events)
        except Exception:
            pass  # Non-fatal
