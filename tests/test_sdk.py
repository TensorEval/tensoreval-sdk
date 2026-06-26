"""Comprehensive test suite for TensorEval SDK.

Tests all modules, imports, and core functionality.
"""

import sys
import json
from pathlib import Path

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_core_types():
    """Test core type definitions."""
    from tensoreval.core.types import (
        State, RolloutInput, RolloutOutput, Messages,
        SystemMessage, UserMessage, AssistantMessage, ToolMessage,
        Tool, Usage, Response, RolloutTiming, TrajectoryStep,
        TokenUsage, GenerateOutputs, GenerateMetadata, RolloutScore,
        flatten_task_input, state_to_output,
    )

    # Test State
    state = State({"prompt": "test", "answer": "42"})
    assert state["prompt"] == "test"
    assert state["answer"] == "42"

    # Test message types
    sys_msg = SystemMessage(content="You are helpful.")
    assert sys_msg.role == "system"
    user_msg = UserMessage(content="What is 2+2?")
    assert user_msg.role == "user"
    asst_msg = AssistantMessage(content="4")
    assert asst_msg.role == "assistant"
    tool_msg = ToolMessage(tool_call_id="tc1", content="result")
    assert tool_msg.role == "tool"

    # Test Tool
    tool = Tool(name="test", description="A test tool", parameters={"type": "object"})
    assert tool.name == "test"

    # Test Usage
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert usage.total_tokens == 15

    # Test RolloutTiming
    timing = RolloutTiming()
    assert timing.start_time > 0

    # Test TrajectoryStep
    step = TrajectoryStep(prompt=[], completion=[], trajectory_id="test")
    assert step.trajectory_id == "test"

    # Test state_to_output
    state["is_completed"] = True
    state["is_truncated"] = False
    state["metrics"] = {}
    state["reward"] = 0.5
    state["timing"] = timing
    state["trajectory"] = []
    state["info"] = {}
    output = state_to_output(state)
    assert output["reward"] == 0.5

    print("  core.types: PASS")


def test_core_errors():
    """Test error hierarchy."""
    from tensoreval.core.errors import (
        Error, ModelError, InvalidModelResponseError,
        EmptyModelResponseError, OverlongPromptError,
        ToolError, ToolParseError, ToolCallError,
        InfraError, SandboxError, EvaluationError, TrainingError,
    )

    assert issubclass(ModelError, Error)
    assert issubclass(ToolError, Error)
    assert issubclass(InfraError, Error)
    assert issubclass(EvaluationError, Error)

    print("  core.errors: PASS")


def test_core_decorators():
    """Test decorator functions."""
    from tensoreval.core.decorators import (
        reward, metric, stop, setup, cleanup, teardown, discover_decorated,
    )

    # Test reward decorator
    @reward(weight=0.5)
    def my_reward():
        return 1.0

    assert hasattr(my_reward, "reward")
    assert my_reward.reward_weight == 0.5

    # Test stop decorator
    @stop(priority=10)
    def my_stop():
        return True

    assert hasattr(my_stop, "stop")
    assert my_stop.stop_priority == 10

    print("  core.decorators: PASS")


def test_core_sample():
    """Test Sample data model."""
    from tensoreval.core.sample import Sample

    sample = Sample(
        input="What is 2+2?",
        target="4",
        rubrics=[{"name": "correctness", "rubric": "Must answer 4", "weight": 1.0}],
    )
    assert sample.input == "What is 2+2?"
    assert sample.target == "4"
    assert len(sample.rubrics) == 1

    print("  core.sample: PASS")


def test_core_score():
    """Test Score data model."""
    from tensoreval.core.score import Score, RubricScore, CORRECT, INCORRECT

    score = Score(value=0.85, answer="4", explanation="Correct")
    assert score.as_float() == 0.85
    assert score.as_str() == "0.85"

    rubric_score = RubricScore(rubric_name="correctness", score=0.9, weight=0.6, reasoning="Good")
    assert rubric_score.score == 0.9

    print("  core.score: PASS")


def test_parsers():
    """Test parser classes."""
    from tensoreval.parsers import Parser, ThinkParser, MaybeThinkParser, XMLParser

    # Test Parser
    parser = Parser()
    assert parser.parse("hello") == "hello"

    # Test ThinkParser
    tp = ThinkParser()
    assert tp.parse("<think>reasoning</think>answer") == "answer"
    assert tp.parse("<think>only") == ""  # No closing tag

    # Test MaybeThinkParser
    mtp = MaybeThinkParser()
    assert mtp.parse("<think>reasoning</think>answer") == "answer"
    assert mtp.parse("no think tags") == "no think tags"

    # Test XMLParser
    xp = XMLParser(["reasoning", "answer"])
    result = xp.parse("<reasoning>step 1</reasoning><answer>42</answer>")
    assert result.reasoning == "step 1"
    assert result.answer == "42"

    # Test XMLParser with alternatives
    xp2 = XMLParser(["reasoning", ("code", "answer")])
    result2 = xp2.parse("<reasoning>thinking</reasoning><answer>42</answer>")
    assert result2.answer == "42"

    print("  parsers: PASS")


def test_rubrics():
    """Test rubric classes."""
    from tensoreval.rubrics import Rubric, RubricGroup

    # Test Rubric
    def my_reward(state, **kwargs):
        return 1.0

    rubric = Rubric(funcs=[my_reward], weights=[1.0])
    assert len(rubric.funcs) == 1
    assert rubric.weights[0] == 1.0

    # Test add_reward_func
    def another_reward(state, **kwargs):
        return 0.5

    rubric.add_reward_func(another_reward, weight=0.5)
    assert len(rubric.funcs) == 2
    assert len(rubric.weights) == 2

    print("  rubrics: PASS")


def test_datasets():
    """Test Datasets class."""
    from tensoreval.datasets import Datasets

    # Test from_dicts
    samples = [
        {"query": "What is 2+2?", "reference_answer": "4"},
        {"query": "What is 3*3?", "reference_answer": "9"},
    ]
    ds = Datasets.from_dicts(samples, name="test")
    assert len(ds) == 2
    assert ds[0].input == "What is 2+2?"
    assert ds[0].target == "4"

    # Test load_from_file
    tasks_path = Path(__file__).parent / "tasks.jsonl"
    if tasks_path.exists():
        ds2 = Datasets.load_from_file(tasks_path)
        assert len(ds2) > 0

    # Test to_dicts
    dicts = ds.to_dicts()
    assert len(dicts) == 2
    assert dicts[0]["query"] == "What is 2+2?"

    print("  datasets: PASS")


def test_evaluation():
    """Test Evaluation class."""
    from tensoreval.evaluation import Evaluation, EvaluationResult

    # Test EvaluationResult
    from tensoreval.core.types import GenerateOutputs, GenerateMetadata
    outputs = GenerateOutputs(
        outputs=[{"reward": 0.8}, {"reward": 0.6}],
        metadata=GenerateMetadata(env_id="test", model="test", num_examples=2, rollouts_per_example=1, avg_reward=0.7, avg_metrics={}),
    )
    from tensoreval.datasets import Datasets
    ds = Datasets.from_dicts([{"query": "q1"}, {"query": "q2"}])
    result = EvaluationResult(outputs, ds, "test")

    assert result.pass_rate == 0.5  # Only 0.8 >= 0.8
    assert result.avg_reward == 0.7
    summary = result.summary()
    assert summary["num_runs"] == 2

    print("  evaluation: PASS")


def test_grader():
    """Test Grader class."""
    from tensoreval.grader import Grader

    # Test basic grader
    def my_reward(state, **kwargs):
        return 1.0

    grader = Grader(funcs=[my_reward], weights=[1.0], model="mimo-v2.5-pro")
    assert grader.model == "mimo-v2.5-pro"

    # Test judge grader
    grader_judge = Grader(model="mimo-v2.5-pro", judge=True)
    assert grader_judge.judge_enabled == True

    # Test ruler grader
    grader_ruler = Grader.ruler(model="mimo-v2.5-pro")
    assert hasattr(grader_ruler, "_ruler_rubric")

    print("  grader: PASS")


def test_training():
    """Test Training class."""
    from tensoreval.training import Training, TrainingRun, TrainingResult
    from tensoreval.training.grpo import compute_grpo_advantage, compute_kl_penalty, policy_gradient_loss

    # Test GRPO advantage
    rewards = [0.9, 0.7, 0.3, 0.1]
    advantages = compute_grpo_advantage(rewards, group_size=4)
    assert len(advantages) == 4
    assert abs(sum(advantages)) < 0.01  # Should sum to ~0

    # Test KL penalty
    penalties = compute_kl_penalty([0.1, 0.2], [0.0, 0.1], kl_coef=0.1)
    assert len(penalties) == 2

    # Test policy gradient loss
    loss = policy_gradient_loss([0.1, 0.2], [0.0, 0.1], [0.5, -0.5])
    assert isinstance(loss, float)

    # Test Training.run
    run = Training.run(base_model="Qwen/Qwen3-8B")
    assert run.status == "initialized"

    print("  training: PASS")


def test_deploy():
    """Test Deployer class."""
    from tensoreval.deploy import Deployer, DeployResult

    # Test together deploy
    result = Deployer.deploy(model_id="test", name="my-agent", provider="together")
    assert result.provider == "together"
    assert "my-agent" in result.model_id

    # Test tinker deploy
    result2 = Deployer.deploy(model_id="test", name="my-agent", provider="tinker")
    assert result2.provider == "tinker"

    # Test local deploy
    result3 = Deployer.deploy(model_id="test", name="my-agent", provider="local")
    assert result3.provider == "local"
    assert "localhost" in result3.base_url

    print("  deploy: PASS")


def test_utils():
    """Test utility functions."""
    from tensoreval.utils import (
        extract_boxed_answer, extract_hash_answer,
        normalize_messages, concat_messages,
    )

    # Test extract_boxed_answer
    assert extract_boxed_answer("The answer is \\boxed{42}") == "42"
    assert extract_boxed_answer("The answer is \\boxed{42}", strict=True) == "42"
    assert extract_boxed_answer("no boxed answer", strict=True) == ""

    # Test extract_hash_answer
    assert extract_hash_answer("#### 42") == "42"
    assert extract_hash_answer("no hash") == "no hash"

    # Test normalize_messages
    msgs = normalize_messages("hello")
    assert len(msgs) == 1
    assert msgs[0].role == "text"

    # Test concat_messages
    from tensoreval.core.types import UserMessage
    m1 = [UserMessage(content="a")]
    m2 = [UserMessage(content="b")]
    combined = concat_messages([m1, m2])
    assert len(combined) == 2

    print("  utils: PASS")


def test_auto_generate():
    """Test AutoGenerator module."""
    from tensoreval.auto_generate import AutoGenerator, _extract_json_array, _extract_json_object

    # Test JSON extraction
    assert _extract_json_array('[1, 2, 3]') == [1, 2, 3]
    assert _extract_json_array('```json\n[1, 2, 3]\n```') == [1, 2, 3]
    assert _extract_json_object('{"key": "value"}') == {"key": "value"}

    print("  auto_generate: PASS")


def test_verified_evaluator():
    """Test VerifiedEvaluator module."""
    from tensoreval.verified_evaluator import VerifiedEvaluator

    # Test that the class exists and has the expected methods
    assert hasattr(VerifiedEvaluator, "evaluate_single")
    assert hasattr(VerifiedEvaluator, "evaluate_batch")

    print("  verified_evaluator: PASS")


def test_mcp_tools():
    """Test MCP tools module."""
    from tensoreval.mcp_tools import MCPTool, MCPServer, MCPToolRegistry

    # Test MCPTool
    server = MCPServer(url="http://localhost:9000/mcp", name="test")
    tool = MCPTool(name="test_tool", description="A test tool", parameters={}, server=server)
    assert tool.name == "test_tool"

    # Test MCPToolRegistry
    registry = MCPToolRegistry()
    registry.add_server("test", server)
    assert "test" in registry.servers

    print("  mcp_tools: PASS")


def test_verifiers_integration():
    """Test VerifiersIntegration module."""
    from tensoreval.verifiers_integration import VerifiersIntegration

    # Test that the class exists
    assert hasattr(VerifiersIntegration, "load")
    assert hasattr(VerifiersIntegration, "list_available")
    assert hasattr(VerifiersIntegration, "from_hub")

    print("  verifiers_integration: PASS")


def test_top_level_import():
    """Test top-level import."""
    import tensoreval as te

    # Check version
    assert te.__version__ == "0.1.0"

    # Check key exports
    assert hasattr(te, "Datasets")
    assert hasattr(te, "Evaluation")
    assert hasattr(te, "Grader")
    assert hasattr(te, "Training")
    assert hasattr(te, "Deployer")
    assert hasattr(te, "AutoGenerator")
    assert hasattr(te, "VerifiedEvaluator")
    assert hasattr(te, "VerifiersIntegration")
    assert hasattr(te, "MCPTool")
    assert hasattr(te, "MCPServer")

    print("  top_level_import: PASS")


def run_all_tests():
    """Run all tests."""
    print("Running TensorEval SDK Test Suite")
    print("=" * 50)

    tests = [
        test_core_types,
        test_core_errors,
        test_core_decorators,
        test_core_sample,
        test_core_score,
        test_parsers,
        test_rubrics,
        test_datasets,
        test_evaluation,
        test_grader,
        test_training,
        test_deploy,
        test_utils,
        test_auto_generate,
        test_verified_evaluator,
        test_mcp_tools,
        test_verifiers_integration,
        test_top_level_import,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            print(f"  {test.__name__}: FAIL - {e}")

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")

    if errors:
        print()
        print("Failures:")
        for name, error in errors:
            print(f"  {name}: {error}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
