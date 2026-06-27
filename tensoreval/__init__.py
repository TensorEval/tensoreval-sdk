"""TensorEval — Evaluation SDK for AI Agents.

Simple API:
    import tensoreval as te

    # From config (Docker auto-started)
    env = te.Env.load_from_file("config.yaml")
    ds = te.Datasets.load_from_dict([...])
    grader = te.AgentGrader(model="mimo-v2.5-pro", api_key="...", base_url="...")
    results = te.Evaluation.run(ds, grader, env=env, workers=4)
"""

from tensoreval.enums import EnvType, Modality, GraderType, Difficulty
from tensoreval.env import Env
from tensoreval.graders import Grader, RubricGrader, AgentGrader, RulerGrader
from tensoreval.datasets import Datasets
from tensoreval.evaluation import Evaluation, EvaluationResult
from tensoreval.voice import VoiceMetrics, IndianLanguageMetrics, AudioMetrics
from tensoreval.mcp_tools import MCPTool, MCPServer, MCPToolRegistry
from tensoreval.docker_compose import DockerCompose
from tensoreval.utils import extract_boxed_answer, extract_hash_answer

__version__ = "0.5.0"

__all__ = [
    "EnvType", "Modality", "GraderType", "Difficulty",
    "Env",
    "Grader", "RubricGrader", "AgentGrader", "RulerGrader",
    "Datasets",
    "Evaluation", "EvaluationResult",
    "VoiceMetrics", "IndianLanguageMetrics", "AudioMetrics",
    "MCPTool", "MCPServer", "MCPToolRegistry",
    "DockerCompose",
    "extract_boxed_answer", "extract_hash_answer",
]
