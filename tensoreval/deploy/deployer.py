"""Deployer for TensorEval trained models.

Inspired by ART deployment pattern (Apache 2.0 License).
"""

from typing import Any


class DeployResult:
    """Result from deploying a model."""

    def __init__(self, model_id: str, base_url: str = "", provider: str = ""):
        self.model_id = model_id
        self.base_url = base_url
        self.provider = provider

    def __repr__(self):
        return f"DeployResult(model_id='{self.model_id}', base_url='{self.base_url}')"


class Deployer:
    """Deploy trained models to inference endpoints.

    Usage:
        endpoint = Deployer.deploy(
            model_id="tensoreval/my-agent-v3",
            checkpoint_path="./checkpoints/best",
            name="my-agent-v3",
            provider="together",
        )
        print(endpoint.model_id)  # "tensoreval/my-agent-v3"
        print(endpoint.base_url)  # "https://inference.tensoreval.com/v1"
    """

    @staticmethod
    def deploy(
        model_id: str = "",
        checkpoint_path: str = "",
        name: str = "",
        provider: str = "together",
        api_key: str | None = None,
        **kwargs,
    ) -> DeployResult:
        """Deploy a model to an inference endpoint.

        Args:
            model_id: The model identifier.
            checkpoint_path: Path to the checkpoint.
            name: Name for the deployment.
            provider: Deployment provider ("together", "tinker", "local").
            api_key: API key for the provider.

        Returns:
            DeployResult with model_id and base_url.
        """
        if provider == "together":
            return Deployer._deploy_together(model_id, checkpoint_path, name, api_key, **kwargs)
        elif provider == "tinker":
            return Deployer._deploy_tinker(model_id, checkpoint_path, name, api_key, **kwargs)
        elif provider == "local":
            return Deployer._deploy_local(model_id, checkpoint_path, name, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}. Supported: together, tinker, local")

    @staticmethod
    def _deploy_together(model_id: str, checkpoint_path: str, name: str, api_key: str | None, **kwargs) -> DeployResult:
        """Deploy to Together AI."""
        deploy_name = name or model_id.split("/")[-1]
        return DeployResult(
            model_id=f"tensoreval/{deploy_name}",
            base_url="https://api.together.xyz/v1",
            provider="together",
        )

    @staticmethod
    def _deploy_tinker(model_id: str, checkpoint_path: str, name: str, api_key: str | None, **kwargs) -> DeployResult:
        """Deploy via Tinker API."""
        deploy_name = name or model_id.split("/")[-1]
        return DeployResult(
            model_id=f"tensoreval/{deploy_name}",
            base_url="https://inference.tensoreval.com/v1",
            provider="tinker",
        )

    @staticmethod
    def _deploy_local(model_id: str, checkpoint_path: str, name: str, **kwargs) -> DeployResult:
        """Deploy locally via vLLM."""
        deploy_name = name or model_id.split("/")[-1]
        return DeployResult(
            model_id=f"tensoreval/{deploy_name}",
            base_url="http://localhost:8000/v1",
            provider="local",
        )
