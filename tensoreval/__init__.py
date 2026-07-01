"""TensorEval — Evaluation SDK for AI Agents.

Usage:
    import tensoreval as te

    # Load dataset
    ds = te.Datasets.load_from_dict([
        {"query": "What is 2+2?", "reference_answer": "4"},
    ])

    # Create grader
    grader = te.RubricGrader(model="gpt-4o", api_key="sk-...", base_url="https://api.openai.com/v1")

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
    GraderType,
    Difficulty,
)

# Dataset loading
from tensoreval.datasets import Datasets

# Evaluation runner
from tensoreval.evaluation import Evaluation, EvaluationResult

# Graders
from tensoreval.graders.base import Grader
from tensoreval.graders.rubric_grader import RubricGrader
from tensoreval.graders.agent_grader import AgentGrader
from tensoreval.graders.ruler_grader import RulerGrader

# Agents
from tensoreval.agents import (
    Agent,
    Context,
    FunctionAgent,
    OpenAIAgent,
    AnthropicAgent,
    EndpointAgent,
)

# Environment config
from tensoreval.env import Env

# Tools
from tensoreval.tools.docker import DockerCompose
from tensoreval.tools.mcp import MCPTool, MCPServer, MCPToolRegistry

# Metrics
from tensoreval.metrics.voice import VoiceMetrics, IndianLanguageMetrics, AudioMetrics

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
    "GraderType", "Difficulty",
    # Core
    "Datasets", "Evaluation", "EvaluationResult",
    # Graders
    "Grader", "RubricGrader", "AgentGrader", "RulerGrader",
    # Agents
    "Agent", "Context", "FunctionAgent", "OpenAIAgent", "AnthropicAgent", "EndpointAgent",
    # Environment
    "Env",
    # Tools
    "DockerCompose", "MCPTool", "MCPServer", "MCPToolRegistry",
    # Metrics
    "VoiceMetrics", "IndianLanguageMetrics", "AudioMetrics",
    # Utilities
    "extract_boxed_answer", "extract_hash_answer",
    # Observability
    "ObservabilityTracer", "RunContext", "Span", "current_run", "current_span",
    "get_tracer", "set_tracer", "observe", "observe_run",
    # Platform client
    "TensorEvalClient", "TensorEvalError",
]
