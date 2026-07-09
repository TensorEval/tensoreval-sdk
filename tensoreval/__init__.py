"""TensorEval — Evaluation SDK for AI agents.

Zero-dependency Python SDK for evaluating AI agents against rubrics
with LLM-as-judge grading, MCP tool verification, and live dashboard integration.

Usage::

    import tensoreval as te

    dataset = te.Dataset.from_jsonl("tests.jsonl")
    env = te.Env.from_endpoint(agent_url="http://localhost:8000/v1/chat/completions")
    grader = te.AgenticGrader(provider="openai", model="gpt-4.1", api_key="sk-...")
    results = te.Evaluation.run(dataset, env, grader, workers=4)
    results.report("report.html")
"""

from __future__ import annotations

__version__ = "0.7.0"

from tensoreval.agentic_grader import AgenticGrader
from tensoreval.client import DashboardClient
from tensoreval.dataset import Dataset
from tensoreval.env import Env
from tensoreval.errors import (
    APIError,
    AuthenticationError,
    EvaluationError,
    MCPError,
    RateLimitError,
    TensorEvalError,
    TimeoutError,
)
from tensoreval.evaluation import Evaluation, EvaluationResult, EvaluationRun
from tensoreval.tool_loop import ProviderConfig
from tensoreval.types import Rubric, Sample

__all__ = [
    "__version__",
    "AgenticGrader",
    "APIError",
    "AuthenticationError",
    "DashboardClient",
    "Dataset",
    "Env",
    "Evaluation",
    "EvaluationError",
    "EvaluationResult",
    "EvaluationRun",
    "MCPError",
    "ProviderConfig",
    "RateLimitError",
    "Rubric",
    "Sample",
    "TensorEvalError",
    "TimeoutError",
]
