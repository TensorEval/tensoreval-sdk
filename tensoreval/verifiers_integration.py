"""Integration with PrimeIntellect Verifiers environments.

Allows loading and using existing Verifiers environments within TensorEval.
"""

from typing import Any


class VerifiersIntegration:
    """Integration with PrimeIntellect Verifiers environments.

    Usage:
        # Load a Verifiers environment
        env = VerifiersIntegration.load("gsm8k")

        # Use with TensorEval evaluation
        results = te.Evaluation.run(datasets, grader, env)
    """

    @staticmethod
    def load(env_name: str, **kwargs) -> Any:
        """Load a Verifiers environment.

        Args:
            env_name: Name of the Verifiers environment (e.g., "gsm8k", "math-python").
            **kwargs: Additional arguments to pass to the environment.

        Returns:
            A Verifiers Environment object wrapped for TensorEval.

        Raises:
            ImportError: If verifiers package is not installed.
        """
        try:
            import verifiers as vf
        except ImportError:
            raise ImportError(
                "verifiers package required for VerifiersIntegration. "
                "Install with: pip install verifiers"
            )

        # Load the environment
        env = vf.load_environment(env_name, **kwargs)

        # Wrap for TensorEval compatibility
        return VerifiersEnvWrapper(env, env_name)

    @staticmethod
    def list_available() -> list[str]:
        """List available Verifiers environments."""
        try:
            import verifiers as vf
            # Common environments
            return [
                "gsm8k", "math", "math-python", "reverse-text",
                "wordle", "alphabet-sort", "wiki-search",
                "code-golf", "browser-dom", "browser-cua",
            ]
        except ImportError:
            return []

    @staticmethod
    def from_hub(env_id: str, **kwargs) -> Any:
        """Load an environment from the Verifiers Hub.

        Args:
            env_id: Hub environment ID (e.g., "primeintellect/gsm8k").

        Returns:
            A wrapped environment.
        """
        try:
            import verifiers as vf
        except ImportError:
            raise ImportError("verifiers package required. Install with: pip install verifiers")

        env = vf.load_environment(env_id, **kwargs)
        return VerifiersEnvWrapper(env, env_id)


class VerifiersEnvWrapper:
    """Wraps a Verifiers Environment for TensorEval compatibility."""

    def __init__(self, env: Any, env_id: str = ""):
        self.env = env
        self.env_id = env_id

    async def evaluate(self, model: str, **kwargs) -> Any:
        """Delegate to the underlying Verifiers environment."""
        return await self.env.evaluate(client=None, model=model, **kwargs)

    def evaluate_sync(self, model: str, **kwargs) -> Any:
        """Synchronous evaluate."""
        return self.env.evaluate_sync(client=None, model=model, **kwargs)

    def get_dataset(self) -> Any:
        """Get the environment's dataset."""
        return self.env.get_dataset()

    def get_eval_dataset(self) -> Any:
        """Get the environment's evaluation dataset."""
        return self.env.get_eval_dataset()
