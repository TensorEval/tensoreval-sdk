"""Evaluation runner for TensorEval.

Core API:
    results = Evaluation.run(dataset, grader, agent=my_agent)
    print(results.summary())
    results.save("results.json")

The agent is a plain async callable:
    async def my_agent(query: str) -> str | AgentResult:
        return "answer"

If ``TENSOREVAL_API_KEY`` is set, results are pushed to the TensorEval
dashboard in real time (start → per-query updates → complete).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Union

from tensoreval.dataset import Dataset
from tensoreval.grader import Grader
from tensoreval.types import AgentResult, EvalConfig, Run, Summary


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

        if config is None:
            config = EvalConfig(**{k: v for k, v in kwargs.items() if k in EvalConfig.__dataclass_fields__})
        else:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        if env is not None:
            if config.system_prompt is None:
                config.system_prompt = env.system_prompt
            if not config.mcp_servers and getattr(env, "mcp_servers", None):
                config.mcp_servers = list(env.mcp_servers)
            if not config.mcp_servers and getattr(env, "mcp_url", None):
                config.mcp_servers = [{"name": "default", "url": env.mcp_url}]

        if not config.api_key:
            config.api_key = (
                os.environ.get("TENSOREVAL_MODEL_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("TENSOREVAL_API_KEY", "")
            )
        if not config.base_url:
            config.base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"

        if grader is None:
            grader = Grader(
                model=config.model,
                api_key=config.api_key,
                base_url=config.base_url,
                fallback_on_error=True,
                timeout=config.timeout,
            )

        env_started = False
        if env is not None and hasattr(env, "start") and (getattr(env, "agent", None) or getattr(env, "mcp", None)):
            try:
                await env.start()
                env_started = True
            except Exception as exc:
                print(f"[tensoreval] warning: env.start() failed: {exc}", file=sys.stderr)

        mcp_urls = extract_mcp_urls(config, env)

        # Live dashboard updates (if TENSOREVAL_API_KEY is set)
        platform_client, live_run_id = start_live_run(config, len(dataset))

        try:
            results = await run_eval(
                dataset, grader, agent, config, mcp_urls,
                platform_client, live_run_id,
            )
            complete_live_run(platform_client, live_run_id, results.summary().to_dict())
            return results
        except Exception as exc:
            complete_live_run(platform_client, live_run_id, failed=True, progress=str(exc))
            raise
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
        """Run evaluation synchronously. Same args as :meth:`run_async`."""
        return asyncio.run(Evaluation.run_async(
            dataset=dataset, grader=grader, agent=agent,
            env=env, config=config, **kwargs,
        ))


# ---------------------------------------------------------------------------
# Internal: evaluation loop
# ---------------------------------------------------------------------------

async def run_eval(
    dataset: Dataset,
    grader: Grader,
    agent: AgentCallable,
    config: EvalConfig,
    mcp_urls: list[str] | None = None,
    platform_client: Any = None,
    live_run_id: str | None = None,
) -> EvaluationResult:
    sem = asyncio.Semaphore(config.workers)
    mcp_urls = mcp_urls or []

    async def eval_one(idx: int) -> Run:
        async with sem:
            run = await evaluate_single(idx, dataset, grader, agent, config, mcp_urls)
            if platform_client and live_run_id:
                try:
                    await asyncio.to_thread(
                        platform_client.append_evaluation_result,
                        live_run_id, run, len(dataset),
                    )
                except Exception:
                    pass
            return run

    tasks = [eval_one(i) for i in range(len(dataset))]
    runs = await asyncio.gather(*tasks)
    return EvaluationResult(list(runs), dataset, config)


async def evaluate_single(
    idx: int,
    dataset: Dataset,
    grader: Grader,
    agent: AgentCallable,
    config: EvalConfig,
    mcp_urls: list[str],
) -> Run:
    sample = dataset[idx]
    start_time = time.monotonic()

    try:
        raw_result = agent(sample.input)
        if asyncio.iscoroutine(raw_result):
            raw_result = await raw_result

        agent_result = AgentResult.coerce(raw_result)
        response = agent_result.response
        tool_trace = agent_result.tool_trace
        latency_ms = (time.monotonic() - start_time) * 1000

        sample_metadata = sample.metadata or {}
        state = {
            "query": sample.input,
            "answer": sample.target,
            "completion": [{"role": "assistant", "content": response}],
            "info": {
                "rubrics": [{"name": r.name, "criteria": r.criteria, "weight": r.weight} for r in sample.rubrics],
                "tool_trace": tool_trace,
                "attachments": sample_metadata.get("attachments", []),
            },
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
# Internal: MCP URL extraction
# ---------------------------------------------------------------------------

def extract_mcp_urls(config: EvalConfig, env: Any = None) -> list[str]:
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
# Internal: live dashboard updates
# ---------------------------------------------------------------------------

def start_live_run(config: EvalConfig, total_count: int) -> tuple[Any, str | None]:
    """Start a live dashboard run when TENSOREVAL_API_KEY is set."""
    if not os.environ.get("TENSOREVAL_API_KEY"):
        return None, None

    from tensoreval.client import TensorEvalClient, TensorEvalError

    try:
        client = TensorEvalClient(timeout=3.0)
        response = client.start_evaluation(model=config.model, total_count=total_count)
        return client, response.get("evaluation_run_id")
    except (TensorEvalError, Exception) as exc:
        print(f"[tensoreval] live dashboard updates skipped: {exc}", file=sys.stderr)
        return None, None


def complete_live_run(
    client: Any,
    run_id: str | None,
    summary: dict | None = None,
    failed: bool = False,
    progress: str | None = None,
) -> None:
    if not client or not run_id:
        return
    try:
        client.complete_evaluation(run_id, summary=summary, failed=failed, progress=progress)
    except Exception:
        pass
