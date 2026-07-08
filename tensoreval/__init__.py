"""TensorEval — Evaluation SDK for AI Agents.

Usage:
    import tensoreval as te

    # Load dataset
    ds = te.Dataset.from_list([
        {"query": "What is 2+2?", "reference_answer": "4"},
    ])

    # Create grader
    grader = te.Grader(model="gpt-4o", api_key="sk-...")

    # Bring your own agent — just an async callable
    async def my_agent(query: str) -> str:
        return "4"

    # Run evaluation
    results = te.Evaluation.run(ds, grader, agent=my_agent)
    print(results.summary())
"""

__version__ = "1.0.0"

# Core types
from tensoreval.types import (
    AgentResult,
    EvalConfig,
    Run,
    Rubric,
    Sample,
    Summary,
)

# Dataset
from tensoreval.dataset import Dataset

# Evaluation
from tensoreval.evaluation import Evaluation, EvaluationResult

# Grader
from tensoreval.grader import Grader

# Environment (Docker + MCP config)
from tensoreval.env import Environment

# Docker
from tensoreval.docker import DockerCompose

# Utilities
from tensoreval.utils import extract_boxed_answer, extract_hash_answer

# Platform client (optional — connects SDK to TensorEval backend)
from tensoreval.client import TensorEvalClient, TensorEvalError


__all__ = [
    "__version__",
    # Types
    "AgentResult", "Rubric", "Sample", "Run", "EvalConfig", "Summary",
    # Core
    "Dataset", "Evaluation", "EvaluationResult", "Grader",
    # Environment
    "Environment", "DockerCompose",
    # Utilities
    "extract_boxed_answer", "extract_hash_answer",
    # Platform client
    "TensorEvalClient", "TensorEvalError",
]
