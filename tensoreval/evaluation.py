"""Evaluation runner for TensorEval.

Orchestrates: datasets + grader + environment → evaluation results.
Follows Verifiers generate() pattern with proper dataset flow.
"""

import asyncio
from typing import Any

from tensoreval.core.types import (
    GenerateOutputs,
    RolloutInput,
    UserMessage,
)
from tensoreval.datasets import Datasets
from tensoreval.envs.environment import Environment
from tensoreval.envs.singleturn_env import SingleTurnEnv
from tensoreval.rubrics.rubric import Rubric


class EvaluationResult:
    """Results from an evaluation run.

    Provides summary statistics and per-query breakdown.
    """

    def __init__(self, outputs: GenerateOutputs, datasets: Datasets, model: str):
        self.outputs = outputs
        self.datasets = datasets
        self.model = model
        self.runs = outputs.get("outputs", [])
        self.metadata = outputs.get("metadata", {})

    @property
    def pass_rate(self) -> float:
        """Fraction of runs that passed (reward >= 0.8)."""
        if not self.runs:
            return 0.0
        passed = sum(1 for r in self.runs if r.get("reward", 0) >= 0.8)
        return passed / len(self.runs)

    @property
    def avg_reward(self) -> float:
        """Average reward across all runs."""
        if not self.runs:
            return 0.0
        return sum(r.get("reward", 0) for r in self.runs) / len(self.runs)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict."""
        return {
            "model": self.model,
            "num_runs": len(self.runs),
            "avg_reward": round(self.avg_reward, 4),
            "pass_rate": round(self.pass_rate, 4),
            "pass_count": sum(1 for r in self.runs if r.get("reward", 0) >= 0.8),
            "fail_count": sum(1 for r in self.runs if r.get("reward", 0) < 0.8),
        }

    def per_query(self) -> list[dict[str, Any]]:
        """Return per-query results."""
        results = []
        for i, run in enumerate(self.runs):
            sample = self.datasets[i] if i < len(self.datasets) else None
            results.append({
                "query_id": sample.id if sample else f"q_{i+1}",
                "query": sample.input if sample else "",
                "reward": run.get("reward", 0),
                "passed": run.get("reward", 0) >= 0.8,
                "metrics": run.get("metrics", {}),
            })
        return results


class Evaluation:
    """Evaluation runner.

    Usage:
        results = Evaluation.run(datasets, grader, env, model="mimo-v2.5-pro")
        print(results.summary())
    """

    @staticmethod
    async def run_async(
        datasets: Datasets,
        grader: Rubric,
        env: Environment | None = None,
        model: str = "mimo-v2.5-pro",
        api_key: str | None = None,
        base_url: str | None = None,
        workers: int = 4,
        **kwargs,
    ) -> EvaluationResult:
        """Run evaluation asynchronously.

        Args:
            datasets: The evaluation datasets.
            grader: The rubric/grader for scoring.
            env: The environment (creates SingleTurnEnv if None).
            model: Model to evaluate.
            api_key: API key for the model.
            base_url: Base URL for the model API.
            workers: Number of concurrent workers.

        Returns:
            EvaluationResult with summary and per-query results.
        """
        if env is None:
            env = SingleTurnEnv(rubric=grader)

        # Inject grader and dataset into environment
        env.rubric = grader

        # Convert datasets to the format the environment expects
        dataset_dicts = datasets.to_dicts()
        env.dataset = dataset_dicts
        env.eval_dataset = dataset_dicts

        # Run evaluation
        outputs = await env.evaluate(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_concurrent=workers,
            **kwargs,
        )

        return EvaluationResult(outputs, datasets, model)

    @staticmethod
    def run(
        datasets: Datasets,
        grader: Rubric,
        env: Environment | None = None,
        model: str = "mimo-v2.5-pro",
        api_key: str | None = None,
        base_url: str | None = None,
        workers: int = 4,
        **kwargs,
    ) -> EvaluationResult:
        """Run evaluation synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = Evaluation.run_async(
            datasets=datasets,
            grader=grader,
            env=env,
            model=model,
            api_key=api_key,
            base_url=base_url,
            workers=workers,
            **kwargs,
        )

        if loop is not None:
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)

        return asyncio.run(coro)
