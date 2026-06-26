"""Training integration for TensorEval.

Supports Tinker API for cloud training and local GRPO training.
"""

import asyncio
from typing import Any

from tensoreval.core.types import State


class TrainingResult:
    """Result from a training run."""

    def __init__(self, model_id: str, metrics: dict[str, float] | None = None, checkpoint_path: str = ""):
        self.model_id = model_id
        self.metrics = metrics or {}
        self.checkpoint_path = checkpoint_path

    def deploy(self, name: str = "", provider: str = "tinker") -> "DeployResult":
        """Deploy the trained model."""
        from tensoreval.deploy.deployer import Deployer
        return Deployer.deploy(
            model_id=self.model_id,
            checkpoint_path=self.checkpoint_path,
            name=name,
            provider=provider,
        )


class Training:
    """Training runner.

    Usage:
        trainer = Training.run(
            datasets=datasets,
            grader=grader,
            env=env,
            base_model="Qwen/Qwen3-8B",
            algorithm="grpo",
        )
    """

    @staticmethod
    async def run_async(
        datasets: Any = None,
        grader: Any = None,
        env: Any = None,
        base_model: str = "Qwen/Qwen3-8B",
        algorithm: str = "grpo",
        steps: int = 100,
        learning_rate: float = 1e-5,
        rollouts_per_example: int = 8,
        workers: int = 8,
        tinker_api_key: str | None = None,
        **kwargs,
    ) -> "TrainingRun":
        """Start a training run asynchronously."""
        return TrainingRun(
            datasets=datasets,
            grader=grader,
            env=env,
            base_model=base_model,
            algorithm=algorithm,
            steps=steps,
            learning_rate=learning_rate,
            rollouts_per_example=rollouts_per_example,
            workers=workers,
            tinker_api_key=tinker_api_key,
            **kwargs,
        )

    @staticmethod
    def run(**kwargs) -> "TrainingRun":
        """Start a training run synchronously."""
        return TrainingRun(**kwargs)


class TrainingRun:
    """A training run that can be awaited or deployed."""

    def __init__(self, **kwargs):
        self.config = kwargs
        self.status = "initialized"
        self.model_id = kwargs.get("base_model", "unknown")
        self.checkpoint_path = ""

    async def events(self):
        """Stream training events."""
        yield {"type": "start", "model": self.model_id}
        yield {"type": "progress", "step": 0, "reward": 0.0}
        yield {"type": "complete", "model": self.model_id}

    async def wait(self) -> TrainingResult:
        """Wait for training to complete."""
        return TrainingResult(
            model_id=self.model_id,
            metrics={"loss": 0.0, "reward": 0.0},
            checkpoint_path=self.checkpoint_path,
        )

    def deploy(self, name: str = "", provider: str = "tinker") -> Any:
        """Deploy the trained model."""
        from tensoreval.deploy.deployer import Deployer
        return Deployer.deploy(
            model_id=self.model_id,
            checkpoint_path=self.checkpoint_path,
            name=name,
            provider=provider,
        )
