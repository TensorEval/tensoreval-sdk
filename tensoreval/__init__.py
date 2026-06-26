"""TensorEval — Evaluation, Training, and Deployment SDK for AI Agents.

Usage:
    import tensoreval as te

    # Load from built-in environments
    env = te.load_env("gsm8k")
    env = te.load_env("customer-support")

    # Or from file
    datasets = te.Datasets.load_from_file("tasks.jsonl")

    # Evaluate
    results = te.Evaluation.run(datasets, grader, env, model="mimo-v2.5-pro")
    print(results.summary())

    # Train
    trainer = te.Training.run(datasets=datasets, base_model="Qwen/Qwen3-8B")

    # Deploy
    endpoint = trainer.deploy(name="my-agent-v3")
"""

# Core types
from tensoreval.core import (
    Error,
    Sample,
    Score,
    RubricScore,
    State,
    Messages,
    RolloutInput,
    RolloutOutput,
)

# Parsers
from tensoreval.parsers import Parser, ThinkParser, MaybeThinkParser, XMLParser

# Rubrics
from tensoreval.rubrics import Rubric, RubricGroup, JudgeRubric, ruler

# Environments
from tensoreval.envs import (
    Environment,
    SingleTurnEnv,
    MultiTurnEnv,
    ToolEnv,
    MultiTurnToolEnv,
    DockerSandboxEnv,
    SandboxToolEnv,
)

# Datasets
from tensoreval.datasets import Datasets

# Evaluation
from tensoreval.evaluation import Evaluation, EvaluationResult

# Grader
from tensoreval.grader import Grader

# Training
from tensoreval.training import (
    Training,
    TrainingRun,
    TrainingResult,
    TokenSample,
    TrajectoryData,
    TrainingDataExporter,
)

# Deploy
from tensoreval.deploy import Deployer, DeployResult

# Built-in environments
from tensoreval.environments import load_env, list_envs

# Auto-generation
from tensoreval.auto_generate import AutoGenerator

# Verified evaluator
from tensoreval.verified_evaluator import VerifiedEvaluator

# Verifiers integration
from tensoreval.verifiers_integration import VerifiersIntegration

# MCP tools
from tensoreval.mcp_tools import MCPTool, MCPServer, MCPToolRegistry

# Utilities
from tensoreval.utils import (
    extract_boxed_answer,
    extract_hash_answer,
    load_example_dataset,
)

__version__ = "0.2.0"

__all__ = [
    # Core
    "Error", "Sample", "Score", "RubricScore", "State", "Messages",
    "RolloutInput", "RolloutOutput",
    # Parsers
    "Parser", "ThinkParser", "MaybeThinkParser", "XMLParser",
    # Rubrics
    "Rubric", "RubricGroup", "JudgeRubric", "ruler",
    # Environments
    "Environment", "SingleTurnEnv", "MultiTurnEnv", "ToolEnv",
    "MultiTurnToolEnv", "DockerSandboxEnv", "SandboxToolEnv",
    # Built-in environments
    "load_env", "list_envs",
    # Datasets
    "Datasets",
    # Evaluation
    "Evaluation", "EvaluationResult",
    # Grader
    "Grader",
    # Training
    "Training", "TrainingRun", "TrainingResult",
    "TokenSample", "TrajectoryData", "TrainingDataExporter",
    # Deploy
    "Deployer", "DeployResult",
    # Auto-generation
    "AutoGenerator",
    # Verified evaluator
    "VerifiedEvaluator",
    # Verifiers integration
    "VerifiersIntegration",
    # MCP tools
    "MCPTool", "MCPServer", "MCPToolRegistry",
    # Utilities
    "extract_boxed_answer", "extract_hash_answer", "load_example_dataset",
]
