"""Test suite for TensorEval SDK v0.5.0."""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_env():
    """Test Env class."""
    import tensoreval as te

    env = te.Env.from_dict({"system_prompt": "test"})
    assert env.system_prompt == "test"

    env2 = te.Env.from_dict({
        "system_prompt": "test",
        "agent": {"image": "python:3.12", "port": 8000},
        "mcp": {"image": "node:18", "port": 9000},
    })
    assert env2.agent is not None
    assert env2.mcp is not None
    print("  env: PASS")


def test_datasets():
    """Test Datasets class."""
    import tensoreval as te

    ds = te.Datasets.load_from_dict([
        {"query": "What is 2+2?", "reference_answer": "4"},
        {"query": "What is 3*3?", "reference_answer": "9"},
    ])
    assert len(ds) == 2
    assert ds[0].input == "What is 2+2?"
    assert ds[0].target == "4"

    ds2 = te.Datasets.load_from_dict([{"query": "test", "reference_answer": "ok"}])
    assert len(ds2) == 1

    # Test load_from_file
    tasks_path = Path(__file__).parent.parent / "examples" / "tasks.jsonl"
    if tasks_path.exists():
        ds3 = te.Datasets.load_from_file(tasks_path)
        assert len(ds3) > 0

    print("  datasets: PASS")


def test_graders():
    """Test grader classes."""
    import tensoreval as te

    # RubricGrader
    g1 = te.RubricGrader()
    assert g1 is not None

    # AgentGrader
    g2 = te.AgentGrader(model="test", api_key="test", base_url="http://test")
    assert g2.model == "test"

    # RulerGrader
    g3 = te.RulerGrader(model="test")
    assert g3.model == "test"

    print("  graders: PASS")


def test_evaluation():
    """Test Evaluation class."""
    import tensoreval as te

    sig_params = list(te.Evaluation.run.__code__.co_varnames[:te.Evaluation.run.__code__.co_argcount])
    assert "datasets" in sig_params
    assert "env" in sig_params
    assert "grader" in sig_params
    assert "agent_port" in sig_params
    assert "mcp_port" in sig_params
    assert "output" in sig_params

    print("  evaluation: PASS")


def test_docker_compose():
    """Test DockerCompose class."""
    import tensoreval as te

    compose = te.DockerCompose(services={
        "agent": {"image": "python:3.12", "port": 8000, "env": {"KEY": "val"}},
        "mcp": {"image": "node:18", "port": 9000},
    })
    yaml = compose._generate_compose_yaml()
    assert "agent:" in yaml
    assert "mcp:" in yaml
    assert "8000:8000" in yaml

    print("  docker_compose: PASS")


def test_mcp_tools():
    """Test MCP tools classes."""
    import tensoreval as te

    server = te.MCPServer(url="http://localhost:9000/mcp", name="test")
    assert server.name == "test"

    registry = te.MCPToolRegistry()
    registry.add_server("test", server)
    assert "test" in registry.servers

    print("  mcp_tools: PASS")


def test_voice_metrics():
    """Test voice metrics."""
    from tensoreval.voice.metrics import VoiceMetrics, IndianLanguageMetrics

    vm = VoiceMetrics()
    assert vm.wer is True
    assert vm.ttft is True

    ilm = IndianLanguageMetrics()
    assert ilm.oi_wer is True

    print("  voice_metrics: PASS")


def test_utils():
    """Test utility functions."""
    import tensoreval as te

    assert te.extract_boxed_answer("The answer is \\boxed{42}") == "42"
    assert te.extract_hash_answer("#### 42") == "42"

    print("  utils: PASS")


def test_rubric_score():
    """Test RubricGrader scoring."""
    import tensoreval as te

    async def run():
        grader = te.RubricGrader()
        state = {
            "prompt": [{"role": "user", "content": "What is 2+2?"}],
            "completion": [{"role": "assistant", "content": "The answer is 4."}],
            "answer": "4",
            "info": {},
        }
        score = await grader.score(state)
        assert score == 1.0

        state2 = {
            "prompt": [{"role": "user", "content": "What is 2+2?"}],
            "completion": [{"role": "assistant", "content": "I don't know."}],
            "answer": "4",
            "info": {},
        }
        score2 = await grader.score(state2)
        assert score2 == 0.0

    asyncio.run(run())
    print("  rubric_score: PASS")


def test_save_load():
    """Test result save/load."""
    import tensoreval as te
    import tempfile
    import os

    ds = te.Datasets.load_from_dict([{"query": "test", "reference_answer": "ok"}])
    result = te.EvaluationResult(
        runs=[{"query_id": "q_1", "query": "test", "reward": 1.0, "response": "ok"}],
        datasets=ds,
        model="test-model",
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        result.save(path)
        loaded = te.EvaluationResult.load(path)
        assert loaded.model == "test-model"
        assert len(loaded.runs) == 1
        assert loaded.summary()["avg_reward"] == 1.0
    finally:
        os.unlink(path)

    print("  save_load: PASS")


def test_top_level_import():
    """Test top-level imports."""
    import tensoreval as te

    assert te.__version__ == "0.5.0"
    assert hasattr(te, "Env")
    assert hasattr(te, "Datasets")
    assert hasattr(te, "Evaluation")
    assert hasattr(te, "EvaluationResult")
    assert hasattr(te, "RubricGrader")
    assert hasattr(te, "AgentGrader")
    assert hasattr(te, "RulerGrader")
    assert hasattr(te, "DockerCompose")
    assert hasattr(te, "MCPServer")

    print("  top_level_import: PASS")


def run_all():
    """Run all tests."""
    print("TensorEval SDK Test Suite")
    print("=" * 50)

    tests = [
        test_env,
        test_datasets,
        test_graders,
        test_evaluation,
        test_docker_compose,
        test_mcp_tools,
        test_voice_metrics,
        test_utils,
        test_rubric_score,
        test_save_load,
        test_top_level_import,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  {t.__name__}: FAIL - {e}")
            failed += 1

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
