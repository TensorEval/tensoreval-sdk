"""TensorEval — Evaluation SDK for AI Agents.

Usage:
    import tensoreval as te

    env = te.Env.from_dict({"system_prompt": "..."})
    ds = te.Datasets.load_from_file("tasks.jsonl")
    grader = te.RubricGrader()
    results = te.Evaluation.run(ds, env, grader, workers=4)
"""

from tensoreval.env import Env
from tensoreval.graders import Grader, RubricGrader, AgentGrader, RulerGrader
from tensoreval.datasets import Datasets
from tensoreval.evaluation import Evaluation, EvaluationResult
from tensoreval.voice import VoiceMetrics
from tensoreval.docker_compose import DockerCompose
from tensoreval.mcp_tools import MCPTool, MCPServer, MCPToolRegistry
from tensoreval.utils import extract_boxed_answer, extract_hash_answer

__version__ = "0.5.0"

__all__ = [
    "Env",
    "Grader", "RubricGrader", "AgentGrader", "RulerGrader",
    "Datasets",
    "Evaluation", "EvaluationResult",
    "VoiceMetrics",
    "DockerCompose",
    "MCPTool", "MCPServer", "MCPToolRegistry",
    "extract_boxed_answer", "extract_hash_answer",
]
