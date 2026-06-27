"""Evaluation runner for TensorEval.

Usage:
    env = te.Env.load_from_file("config.yaml")
    ds = te.Datasets.load_from_dict([...])
    grader = te.RubricGrader()
    results = te.Evaluation.run(ds, env, grader, workers=4, agent_port=8000, mcp_port=9000)
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from tensoreval.graders.base import Grader
from tensoreval.datasets import Datasets


class EvaluationResult:
    """Results from an evaluation run."""

    def __init__(self, runs: list[dict], datasets: Datasets, model: str):
        self.runs = runs
        self.datasets = datasets
        self.model = model

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
            "summary": self.summary(),
            "runs": self.runs,
            "datasets": [
                {
                    "input": s.input,
                    "target": s.target,
                    "rubrics": s.rubrics if hasattr(s, "rubrics") else [],
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
            Sample(input=s["input"], target=s.get("target", ""), rubrics=s.get("rubrics", []))
            for s in data.get("datasets", [])
        ]
        return cls(runs=data.get("runs", []), datasets=Datasets(samples), model=data.get("model", "unknown"))


class Evaluation:
    """Evaluation runner.

    Usage:
        env = te.Env.load_from_file("config.yaml")
        ds = te.Datasets.load_from_dict([...])
        grader = te.RubricGrader()
        results = te.Evaluation.run(ds, env, grader, workers=4, agent_port=8000, mcp_port=9000)
    """

    @staticmethod
    async def run_async(
        datasets: Datasets,
        env: Any = None,
        grader: Grader | None = None,
        model: str = "mimo-v2.5-pro",
        api_key: str | None = None,
        base_url: str | None = None,
        workers: int = 4,
        agent_port: int | None = None,
        mcp_port: int | None = None,
        system_prompt: str | None = None,
        output: str | None = None,
    ) -> EvaluationResult:
        """Run evaluation.

        Args:
            datasets: Samples with queries, answers, rubrics.
            env: Env config (system_prompt, agent_url, mcp_url, Docker config).
            grader: Scorer (default: RubricGrader).
            model: Model name.
            api_key: API key.
            base_url: Base URL.
            workers: Concurrent workers.
            agent_port: Port for agent endpoint (overrides env).
            mcp_port: Port for MCP server (overrides env).
            system_prompt: System prompt (overrides env).
            output: Save results to this path.
        """
        # Wire env config
        if env is not None:
            if system_prompt is None:
                system_prompt = env.system_prompt
            if agent_port is None and env.agent_url:
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

        resolved_key = api_key or os.environ.get("TENSOREVAL_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        resolved_url = base_url or os.environ.get("TENSOREVAL_BASE_URL", "https://api.openai.com/v1")

        # Build inputs
        inputs = []
        for i, sample in enumerate(datasets):
            inputs.append({
                "query": sample.input,
                "answer": sample.target if isinstance(sample.target, str) else str(sample.target),
                "rubrics": sample.rubrics if hasattr(sample, "rubrics") else [],
                "info": sample.metadata if hasattr(sample, "metadata") and sample.metadata else {},
                "index": i,
            })

        # Run evaluations
        sem = asyncio.Semaphore(workers)

        async def eval_one(inp: dict) -> dict:
            async with sem:
                return await _evaluate_single(
                    inp, grader, model, resolved_key, resolved_url,
                    agent_port, mcp_port, system_prompt,
                )

        tasks = [eval_one(inp) for inp in inputs]
        runs = await asyncio.gather(*tasks)

        result = EvaluationResult(list(runs), datasets, model)

        if output:
            result.save(output)

        return result

    @staticmethod
    def run(
        datasets: Datasets,
        env: Any = None,
        grader: Grader | None = None,
        model: str = "mimo-v2.5-pro",
        api_key: str | None = None,
        base_url: str | None = None,
        workers: int = 4,
        agent_port: int | None = None,
        mcp_port: int | None = None,
        system_prompt: str | None = None,
        output: str | None = None,
    ) -> EvaluationResult:
        """Run evaluation synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = Evaluation.run_async(
            datasets=datasets, env=env, grader=grader, model=model,
            api_key=api_key, base_url=base_url, workers=workers,
            agent_port=agent_port, mcp_port=mcp_port,
            system_prompt=system_prompt, output=output,
        )

        if loop is not None:
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return asyncio.run(coro)


async def _evaluate_single(inp, grader, model, api_key, base_url, agent_port, mcp_port, system_prompt):
    query = inp["query"]
    answer = inp["answer"]

    if agent_port:
        response = await _call_agent(query, agent_port, system_prompt)
    else:
        response = await _call_model(query, model, api_key, base_url, system_prompt)

    state = {
        "prompt": [{"role": "user", "content": query}],
        "completion": [{"role": "assistant", "content": response}],
        "answer": answer,
        "info": {**inp.get("info", {}), "rubrics": inp.get("rubrics", [])},
    }

    reward = await grader.score(state)

    return {
        "query_id": f"q_{inp['index'] + 1}",
        "query": query,
        "answer": answer,
        "response": response,
        "reward": reward,
        "completion": state["completion"],
    }


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
