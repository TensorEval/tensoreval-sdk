"""TensorEval — Evaluation SDK for AI Agents.

Usage:
    import tensoreval as te

    # Load dataset
    ds = te.Datasets.load_from_dict([
        {"query": "What is 2+2?", "reference_answer": "4"},
    ])

    # Create grader
    grader = te.VercelAgentGrader()

    # Run evaluation — agent can be a function, class, URL, or model name
    results = te.Evaluation.run(ds, grader, agent=my_agent_function)
    print(results.summary())
"""

__version__ = "0.6.0"

# Core types
from tensoreval.types import (
    Rubric,
    Sample,
    Score,
    Run,
    EvalConfig,
    Summary,
    TEvalResult,
    GraderType,
)

# Dataset loading
from tensoreval.datasets import Datasets

# Evaluation runner
from tensoreval.evaluation import Evaluation, EvaluationResult

# Graders
from tensoreval.graders.agent_grader import AgentGrader, VercelAgentGrader
from tensoreval.graders.verification_grader import VerificationGrader

# Agents
from tensoreval.agents import (
    Agent,
    Context,
    FunctionAgent,
    OpenAIAgent,
)

# Environment config
from tensoreval.env import Environment

# Tools
from tensoreval.tools.mcp import LocalTool, MCPTool, MCPServer, tool

# Utilities
from tensoreval.utils.data_utils import extract_boxed_answer, extract_hash_answer

# Platform client (connect SDK to the TensorEval backend)
from tensoreval.client import TensorEvalClient, TensorEvalError

# Observability (opt-in tracing; LangSmith-style spans/runs/gaps)
from tensoreval.observability import (
    ObservabilityTracer,
    RunContext,
    Span,
    current_run,
    current_span,
    get_tracer,
    set_tracer,
    observe,
    observe_run,
)


__all__ = [
    # Version
    "__version__",
    # Types
    "Rubric", "Sample", "Score", "Run", "EvalConfig", "Summary",
    "TEvalResult", "GraderType",
    # Core
    "Datasets", "Evaluation", "EvaluationResult",
    # Graders
    "AgentGrader", "VercelAgentGrader", "VerificationGrader",
    # Agents
    "Agent", "Context", "FunctionAgent", "OpenAIAgent",
    # Environment
    "Environment",
    # Tools
    "LocalTool", "MCPTool", "MCPServer", "tool",
    # Utilities
    "extract_boxed_answer", "extract_hash_answer",
    # Observability
    "ObservabilityTracer", "RunContext", "Span", "current_run", "current_span",
    "get_tracer", "set_tracer", "observe", "observe_run",
    # Platform client
    "TensorEvalClient", "TensorEvalError",
]
