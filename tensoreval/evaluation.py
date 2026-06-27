"""Evaluation runner for TensorEval.

Fully wired: Env config, Docker, voice metrics, agent endpoints, MCP, persistence.

Usage:
    # Simple (model API)
    results = te.Evaluation.run(ds, grader, model="mimo-v2.5-pro", api_key="...", base_url="...")

    # With Docker environment
    env = te.Env.from_dict({
        "system_prompt": "You are a support agent...",
        "agent": {"image": "my-agent:latest", "port": 8000},
        "mcp": {"image": "my-mcp:latest", "port": 9000},
    })
    results = te.Evaluation.run(ds, grader, env=env)

    # With direct URLs (no Docker)
    env = te.Env.from_dict({
        "system_prompt": "...",
        "agent_url": "http://localhost:8000",
        "mcp_url": "http://localhost:9000/mcp",
    })
    results = te.Evaluation.run(ds, grader, env=env)

    # Save results
    results.save("results.json")
    loaded = te.EvaluationResult.load("results.json")
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from tensoreval.graders.base import Grader
from tensoreval.datasets import Datasets


class EvaluationResult:
    """Results from an evaluation run. Supports save/load."""

    def __init__(
        self,
        runs: list[dict],
        datasets: Datasets,
        model: str,
        voice_metrics: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.runs = runs
        self.datasets = datasets
        self.model = model
        self.voice_metrics = voice_metrics or {}
        self.config = config or {}

    @property
    def pass_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.get("reward", 0) >= 0.8) / len(self.runs)

    @property
    def avg_reward(self) -> float:
        if not self.runs:
            return 0.0
        return sum(r.get("reward", 0) for r in self.runs) / len(self.runs)

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "num_runs": len(self.runs),
            "avg_reward": round(self.avg_reward, 4),
            "pass_rate": round(self.pass_rate, 4),
            "pass_count": sum(1 for r in self.runs if r.get("reward", 0) >= 0.8),
            "fail_count": sum(1 for r in self.runs if r.get("reward", 0) < 0.8),
            "voice_metrics": self.voice_metrics,
        }

    def per_query(self) -> list[dict[str, Any]]:
        results = []
        for i, run in enumerate(self.runs):
            sample = self.datasets[i] if i < len(self.datasets) else None
            results.append({
                "query_id": sample.id if sample else f"q_{i+1}",
                "query": sample.input if sample else "",
                "reward": run.get("reward", 0),
                "passed": run.get("reward", 0) >= 0.8,
                "response": run.get("response", ""),
            })
        return results

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self.model,
            "config": self.config,
            "voice_metrics": self.voice_metrics,
            "summary": self.summary(),
            "runs": self.runs,
            "datasets": [
                {
                    "input": s.input,
                    "target": s.target,
                    "rubrics": s.rubrics if hasattr(s, "rubrics") else [],
                    "metadata": s.metadata if hasattr(s, "metadata") else {},
                }
                for s in self.datasets
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationResult":
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        from tensoreval.datasets import Sample
        samples = [
            Sample(
                input=s.get("input", ""),
                target=s.get("target", ""),
                rubrics=s.get("rubrics", []),
                metadata=s.get("metadata", {}),
            )
            for s in data.get("datasets", [])
        ]
        return cls(
            runs=data.get("runs", []),
            datasets=Datasets(samples),
            model=data.get("model", "unknown"),
            voice_metrics=data.get("voice_metrics", {}),
            config=data.get("config", {}),
        )


class Evaluation:
    """Evaluation runner. Fully wired for Docker, MCP, agent endpoints."""

    @staticmethod
    async def run_async(
        datasets: Datasets,
        grader: Grader | None = None,
        env: Any = None,
        model: str = "mimo-v2.5-pro",
        api_key: str | None = None,
        base_url: str | None = None,
        workers: int = 4,
        agent_port: int | None = None,
        mcp_port: int | None = None,
        system_prompt: str | None = None,
        voice_metrics: bool = False,
        output: str | None = None,
    ) -> EvaluationResult:
        """Run evaluation.

        Args:
            datasets: Samples with queries, answers, rubrics.
            grader: Scorer (default: RubricGrader).
            env: Env config (starts Docker if configured).
            model: Model name.
            api_key: API key.
            base_url: Base URL.
            workers: Concurrent workers.
            agent_port: Override agent port.
            mcp_port: Override MCP port.
            system_prompt: Override system prompt.
            voice_metrics: Compute voice metrics.
            output: Save results to this path.
        """
        # ── Start Docker if env has containers configured ────────
        docker_started = False
        if env is not None:
            if env.agent or env.mcp:
                urls = await env.start()
                docker_started = True

        # ── Wire env config ──────────────────────────────────────
        if env is not None:
            if system_prompt is None:
                system_prompt = env.system_prompt
            if agent_port is None and env.agent_url:
                # Only use agent_url if no API key provided (user wants agent testing)
                if not api_key and not base_url:
                    try:
                        agent_port = int(env.agent_url.rsplit(":", 1)[-1])
                    except ValueError:
                        pass
            if mcp_port is None and env.mcp_url:
                try:
                    mcp_port = int(env.mcp_url.rsplit(":", 1)[-1])
                except ValueError:
                    pass

        if grader is None:
            from tensoreval.graders.rubric_grader import RubricGrader
            grader = RubricGrader()

        resolved_key = api_key or _default_api_key()
        resolved_url = base_url or _default_base_url()

        # ── Build inputs ─────────────────────────────────────────
        inputs = []
        for i, sample in enumerate(datasets):
            query = sample.input
            answer = sample.target if isinstance(sample.target, str) else str(sample.target)
            inputs.append({
                "query": query,
                "answer": answer,
                "rubrics": sample.rubrics if hasattr(sample, "rubrics") else [],
                "info": sample.metadata if hasattr(sample, "metadata") and sample.metadata else {},
                "index": i,
            })

        # ── Run evaluations ──────────────────────────────────────
        sem = asyncio.Semaphore(workers)

        async def eval_one(inp: dict) -> dict:
            async with sem:
                return await _evaluate_single(
                    inp, grader, model, resolved_key, resolved_url,
                    agent_port, mcp_port, system_prompt, voice_metrics,
                )

        tasks = [eval_one(inp) for inp in inputs]
        runs = await asyncio.gather(*tasks)

        # ── Aggregate voice metrics ──────────────────────────────
        agg_voice = {}
        if voice_metrics:
            all_ttft = [r.get("voice_metrics", {}).get("ttft", 0) for r in runs if r.get("voice_metrics", {}).get("ttft")]
            all_wpm = [r.get("voice_metrics", {}).get("wpm", 0) for r in runs if r.get("voice_metrics", {}).get("wpm")]
            if all_ttft:
                agg_voice["avg_ttft_ms"] = round(sum(all_ttft) / len(all_ttft), 1)
            if all_wpm:
                agg_voice["avg_wpm"] = round(sum(all_wpm) / len(all_wpm), 1)

        result = EvaluationResult(
            runs=list(runs),
            datasets=datasets,
            model=model,
            voice_metrics=agg_voice,
            config={"model": model, "workers": workers, "voice_metrics": voice_metrics},
        )

        if output:
            result.save(output)

        return result

    @staticmethod
    def run(
        datasets: Datasets,
        grader: Grader | None = None,
        env: Any = None,
        model: str = "mimo-v2.5-pro",
        api_key: str | None = None,
        base_url: str | None = None,
        workers: int = 4,
        agent_port: int | None = None,
        mcp_port: int | None = None,
        system_prompt: str | None = None,
        voice_metrics: bool = False,
        output: str | None = None,
    ) -> EvaluationResult:
        """Run evaluation synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = Evaluation.run_async(
            datasets=datasets, grader=grader, env=env, model=model,
            api_key=api_key, base_url=base_url, workers=workers,
            agent_port=agent_port, mcp_port=mcp_port,
            system_prompt=system_prompt, voice_metrics=voice_metrics,
            output=output,
        )

        if loop is not None:
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Internal: single evaluation
# ---------------------------------------------------------------------------

async def _evaluate_single(
    inp: dict,
    grader: Grader,
    model: str,
    api_key: str,
    base_url: str,
    agent_port: int | None,
    mcp_port: int | None,
    system_prompt: str | None,
    voice_metrics: bool,
) -> dict:
    query = inp["query"]
    answer = inp["answer"]
    start_time = time.time()

    if agent_port:
        response = await _call_agent(query, agent_port, system_prompt)
    else:
        response = await _call_model(query, model, api_key, base_url, system_prompt)

    latency_ms = (time.time() - start_time) * 1000

    state = {
        "prompt": [{"role": "user", "content": query}],
        "completion": [{"role": "assistant", "content": response}],
        "answer": answer,
        "info": {**inp.get("info", {}), "rubrics": inp.get("rubrics", [])},
    }

    reward = await grader.score(state)

    voice = {}
    if voice_metrics:
        word_count = len(response.split())
        voice = {
            "ttft": latency_ms,
            "wpm": (word_count / (latency_ms / 1000)) * 60 if latency_ms > 0 else 0,
            "word_count": word_count,
            "latency_ms": latency_ms,
        }

    return {
        "query_id": f"q_{inp['index'] + 1}",
        "query": query,
        "answer": answer,
        "response": response,
        "reward": reward,
        "completion": state["completion"],
        "voice_metrics": voice,
    }


# ---------------------------------------------------------------------------
# Model/Agent calling
# ---------------------------------------------------------------------------

async def _call_model(query, model, api_key, base_url, system_prompt):
    is_anthropic = base_url and "anthropic" in base_url.lower()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    if is_anthropic:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)
        response = await client.messages.create(model=model, max_tokens=2000, messages=messages)
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""
    else:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        response = await client.chat.completions.create(model=model, max_tokens=2000, messages=messages)
        return response.choices[0].message.content or ""


async def _call_agent(query, agent_port, system_prompt):
    import httpx
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://localhost:{agent_port}/v1/chat/completions",
            json={"messages": messages},
            timeout=120.0,
        )
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def _default_api_key():
    return os.environ.get("TENSOREVAL_API_KEY", os.environ.get("OPENAI_API_KEY", ""))


def _default_base_url():
    return os.environ.get("TENSOREVAL_BASE_URL", "https://api.openai.com/v1")


import os
