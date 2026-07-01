"""Comprehensive test suite for TensorEval SDK v0.6.0.

Tests are organized by module:
1. Types — data structures
2. Datasets — loading and conversion
3. Graders — scoring logic (no API calls)
4. Agents — agent abstraction
5. Evaluation — full pipeline (with mock agent)
6. Env — environment config
7. Tools — Docker compose YAML generation
8. Metrics — voice metrics computation
9. Utils — parsing helpers
"""

import asyncio
import json
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ===========================================================================
# 1. TYPES
# ===========================================================================

def test_types():
    """Test core data structures."""
    from tensoreval.types import Rubric, Sample, Score, Run, EvalConfig, Summary
    from tensoreval.enums import GraderType, Difficulty

    # Rubric
    r = Rubric(name="accuracy", criteria="Must be correct", weight=0.5)
    assert r.name == "accuracy"
    assert r.weight == 0.5

    r2 = Rubric.from_dict({"name": "empathy", "rubric": "Must show empathy", "weight": 0.3})
    assert r2.criteria == "Must show empathy"

    # Score
    s = Score(value=0.8, explanation="Good response")
    assert s.value == 0.8

    # Sample
    sample = Sample(input="What is 2+2?", target="4")
    assert sample.input == "What is 2+2?"
    assert sample.target == "4"
    assert sample.id != ""  # Auto-generated

    sample2 = Sample(input="test", id="custom_id")
    assert sample2.id == "custom_id"

    # Run
    run = Run(sample_id="q1", query="test", answer="4", response="4", reward=1.0)
    assert run.reward == 1.0
    assert run.error is None

    # EvalConfig
    config = EvalConfig(model="gpt-4o", workers=8)
    assert config.model == "gpt-4o"
    assert config.workers == 8
    assert config.pass_threshold == 0.8

    # Summary
    summary = Summary(model="gpt-4o", num_runs=10, avg_reward=0.8, pass_rate=0.8, pass_count=8, fail_count=2)
    d = summary.to_dict()
    assert d["model"] == "gpt-4o"
    assert d["num_runs"] == 10

    # Enums
    assert GraderType.RUBRIC.value == "rubric"
    assert Difficulty.EASY.value == "easy"

    print("  types: PASS")


# ===========================================================================
# 2. DATASETS
# ===========================================================================

def test_datasets():
    """Test dataset loading."""
    import tensoreval as te

    # From dict
    ds = te.Datasets.load_from_dict([
        {"query": "What is 2+2?", "reference_answer": "4"},
        {"query": "What is 3*3?", "reference_answer": "9"},
    ])
    assert len(ds) == 2
    assert ds[0].input == "What is 2+2?"
    assert ds[0].target == "4"
    assert ds[1].target == "9"

    # Flexible field names
    ds2 = te.Datasets.load_from_dict([
        {"input": "Question 1", "target": "Answer 1"},
        {"question": "Question 2", "answer": "Answer 2"},
    ])
    assert ds2[0].input == "Question 1"
    assert ds2[1].input == "Question 2"

    # With rubrics
    ds3 = te.Datasets.load_from_dict([{
        "query": "Handle refund",
        "reference_answer": "Process refund",
        "rubrics": [
            {"name": "policy", "criteria": "Must follow 30-day policy", "weight": 0.5},
            {"name": "empathy", "criteria": "Must show empathy", "weight": 0.5},
        ],
    }])
    assert len(ds3[0].rubrics) == 2
    assert ds3[0].rubrics[0].name == "policy"

    # Iteration
    for sample in ds:
        assert sample.input != ""

    # to_dicts
    d = ds.to_dicts()
    assert len(d) == 2
    assert d[0]["query"] == "What is 2+2?"

    # From JSONL file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"query": "Q1", "reference_answer": "A1"}\n')
        f.write('{"query": "Q2", "reference_answer": "A2"}\n')
        f.write('\n')  # Empty line should be skipped
        path = f.name

    try:
        ds4 = te.Datasets.load_from_file(path)
        assert len(ds4) == 2
        assert ds4[0].input == "Q1"
        assert ds4[1].target == "A2"
    finally:
        os.unlink(path)

    # Empty dataset raises
    try:
        te.Datasets.load_from_dict([])
        assert False, "Should have raised"
    except ValueError:
        pass

    print("  datasets: PASS")


# ===========================================================================
# 3. GRADERS (no API calls)
# ===========================================================================

def test_rubric_grader_simple():
    """Test RubricGrader in simple mode (no API calls)."""
    import tensoreval as te

    async def run():
        grader = te.RubricGrader(simple=True)

        # Exact match
        score = await grader.score({
            "query": "What is 2+2?",
            "completion": [{"role": "assistant", "content": "The answer is 4."}],
            "answer": "4",
            "info": {},
        })
        assert score == 1.0

        # No match
        score2 = await grader.score({
            "query": "What is 2+2?",
            "completion": [{"role": "assistant", "content": "I don't know."}],
            "answer": "4",
            "info": {},
        })
        assert score2 == 0.0

        # Numeric match (number in response)
        score3 = await grader.score({
            "query": "What is 12*15?",
            "completion": [{"role": "assistant", "content": "12 * 15 = **180**"}],
            "answer": "180",
            "info": {},
        })
        assert score3 == 1.0

        # Empty completion
        score4 = await grader.score({
            "query": "test",
            "completion": [],
            "answer": "test",
            "info": {},
        })
        assert score4 == 0.0

    asyncio.run(run())
    print("  rubric_grader_simple: PASS")


def test_rubric_grader_llm_fallback():
    """Test RubricGrader falls back to simple when no rubrics."""
    import tensoreval as te

    async def run():
        # With model configured but no rubrics → falls back to simple
        grader = te.RubricGrader(
            model="test-model",
            api_key="test-key",
            base_url="http://test",
        )
        score = await grader.score({
            "query": "What is 2+2?",
            "completion": [{"role": "assistant", "content": "4"}],
            "answer": "4",
            "info": {"rubrics": []},
        })
        assert score == 1.0

    asyncio.run(run())
    print("  rubric_grader_llm_fallback: PASS")


def test_ruler_grader_single():
    """Test RulerGrader single-sample scoring."""
    import tensoreval as te

    async def run():
        grader = te.RulerGrader(model="test", api_key="test", base_url="http://test")

        # With answer match
        score = await grader.score({
            "query": "What is 2+2?",
            "completion": [{"role": "assistant", "content": "The answer is 4."}],
            "answer": "4",
            "info": {},
        })
        assert score >= 0.5  # Should get reasonable heuristic score

        # Empty response
        score2 = await grader.score({
            "query": "test",
            "completion": [],
            "answer": "test",
            "info": {},
        })
        assert score2 == 0.0

    asyncio.run(run())
    print("  ruler_grader_single: PASS")


def test_ruler_grader_group():
    """Test RulerGrader group scoring (fallback)."""
    import tensoreval as te

    async def run():
        grader = te.RulerGrader(model="test", api_key="test", base_url="http://test")

        # Single item in group → returns 0.5
        scores = await grader.score_group([{
            "completion": [{"role": "assistant", "content": "test"}],
            "answer": "test",
        }])
        assert scores == [0.5]

        # Empty group
        scores2 = await grader.score_group([])
        assert scores2 == []

    asyncio.run(run())
    print("  ruler_grader_group: PASS")


def test_grader_base():
    """Test Grader base class."""
    from tensoreval.graders.base import Grader

    # Base class score() raises NotImplementedError
    g = Grader()
    try:
        asyncio.run(g.score({}))
        assert False
    except NotImplementedError:
        pass

    # score_group() calls score() for each
    class TestGrader(Grader):
        async def score(self, state, **kwargs):
            return 0.5

    g2 = TestGrader()
    result = asyncio.run(g2.score_group([{}, {}, {}]))
    assert result == [0.5, 0.5, 0.5]

    print("  grader_base: PASS")


# ===========================================================================
# 4. AGENTS
# ===========================================================================

def test_agents():
    """Test agent abstraction."""
    import tensoreval as te
    from tensoreval.agents import Agent, Context, FunctionAgent, resolve_agent

    # FunctionAgent wraps a callable
    async def my_fn(query: str) -> str:
        return f"Response to: {query}"

    agent = FunctionAgent(my_fn)
    ctx = Context(query="test")
    result = asyncio.run(agent.run("test", ctx))
    assert result == "Response to: test"

    # Custom Agent subclass
    class MyAgent(Agent):
        async def run(self, query: str, context: Context) -> str:
            return f"Custom: {query}"

    agent2 = MyAgent()
    result2 = asyncio.run(agent2.run("hello", ctx))
    assert result2 == "Custom: hello"

    # resolve_agent with None → defaults to OpenAIAgent
    agent3 = resolve_agent(None, model="gpt-4o", api_key="sk-test")
    assert isinstance(agent3, te.OpenAIAgent)

    # resolve_agent with callable → FunctionAgent
    agent4 = resolve_agent(my_fn)
    assert isinstance(agent4, FunctionAgent)

    # resolve_agent with Agent instance → returned as-is
    agent5 = resolve_agent(agent2)
    assert agent5 is agent2

    # resolve_agent with http URL → EndpointAgent
    agent6 = resolve_agent("http://localhost:8000")
    assert isinstance(agent6, te.EndpointAgent)

    # resolve_agent with anthropic: prefix → AnthropicAgent
    agent7 = resolve_agent("anthropic:mimo-v2.5-pro", api_key="key", base_url="url")
    assert isinstance(agent7, te.AnthropicAgent)

    print("  agents: PASS")


# ===========================================================================
# 5. EVALUATION (with mock agent)
# ===========================================================================

def test_evaluation_with_function_agent():
    """Test full evaluation pipeline with a function agent."""
    import tensoreval as te

    async def mock_agent(query: str) -> str:
        # Simulate different responses
        if "2+2" in query or "2 + 2" in query:
            return "4"
        if "capital of France" in query:
            return "Paris"
        return "I don't know."

    ds = te.Datasets.load_from_dict([
        {"query": "What is 2+2?", "reference_answer": "4"},
        {"query": "What is the capital of France?", "reference_answer": "Paris"},
        {"query": "What is the speed of light?", "reference_answer": "299792458"},
    ])

    grader = te.RubricGrader(simple=True)
    results = te.Evaluation.run(ds, grader, agent=mock_agent)

    assert len(results.runs) == 3
    assert results.runs[0].reward == 1.0  # 2+2 = 4 → match
    assert results.runs[1].reward == 1.0  # Paris → match
    assert results.runs[2].reward == 0.0  # speed of light → no match

    summary = results.summary()
    assert summary.num_runs == 3
    assert summary.pass_count == 2
    assert summary.fail_count == 1

    print("  evaluation_function_agent: PASS")


def test_evaluation_with_agent_class():
    """Test evaluation with a custom Agent class."""
    import tensoreval as te
    from tensoreval.agents import Agent, Context

    class MathAgent(Agent):
        async def run(self, query: str, context: Context) -> str:
            if "2+2" in query:
                return "4"
            return "unknown"

    ds = te.Datasets.load_from_dict([
        {"query": "What is 2+2?", "reference_answer": "4"},
    ])

    results = te.Evaluation.run(ds, te.RubricGrader(simple=True), agent=MathAgent())
    assert results.runs[0].reward == 1.0

    print("  evaluation_agent_class: PASS")


def test_evaluation_results_save_load():
    """Test saving and loading results."""
    import tensoreval as te

    ds = te.Datasets.load_from_dict([{"query": "test", "reference_answer": "ok"}])
    results = te.EvaluationResult(
        runs=[te.Run(sample_id="q1", query="test", answer="ok", response="ok", reward=1.0)],
        datasets=ds,
        config=te.EvalConfig(model="test-model"),
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        results.save(path)
        loaded = te.EvaluationResult.load(path)
        assert loaded.config.model == "test-model"
        assert len(loaded.runs) == 1
        assert loaded.runs[0].reward == 1.0
        assert loaded.summary().avg_reward == 1.0
    finally:
        os.unlink(path)

    print("  evaluation_save_load: PASS")


def test_evaluation_summary():
    """Test summary computation."""
    import tensoreval as te

    ds = te.Datasets.load_from_dict([
        {"query": "Q1", "reference_answer": "A1"},
        {"query": "Q2", "reference_answer": "A2"},
        {"query": "Q3", "reference_answer": "A3"},
    ])

    runs = [
        te.Run(sample_id="q1", query="Q1", answer="A1", response="A1", reward=1.0),
        te.Run(sample_id="q2", query="Q2", answer="A2", response="wrong", reward=0.0),
        te.Run(sample_id="q3", query="Q3", answer="A3", response="A3", reward=0.9),
    ]

    results = te.EvaluationResult(runs=runs, datasets=ds, config=te.EvalConfig(model="test"))

    assert results.avg_reward > 0.6
    assert results.summary().pass_count == 2  # 1.0 and 0.9 pass
    assert results.summary().fail_count == 1  # 0.0 fails
    assert results.pass_rate > 0.6

    print("  evaluation_summary: PASS")


def test_evaluation_observability_traces():
    """Test evaluation emits self-contained observability events."""
    import tensoreval as te

    events = []
    previous_tracer = te.get_tracer()
    te.set_tracer(te.ObservabilityTracer(sinks=[events.append], auto_env=False))

    async def mock_agent(query: str) -> str:
        return "4"

    try:
        ds = te.Datasets.load_from_dict([{"query": "What is 2+2?", "reference_answer": "4"}])
        results = te.Evaluation.run(ds, te.RubricGrader(simple=True), agent=mock_agent, model="gpt-4o")
    finally:
        te.set_tracer(previous_tracer)

    assert results.runs[0].reward == 1.0
    assert [e["event"] for e in events if e["event"].startswith("run_")] == ["run_start", "run_end"]

    spans = [e for e in events if e["event"] == "span_end"]
    assert {s["kind"] for s in spans} == {"agent", "grader"}
    assert all(s["run_id"] == events[0]["run_id"] for s in spans)

    run_end = events[-1]
    assert run_end["summary"]["avg_reward"] == 1.0
    assert run_end["summary"]["pass_rate"] == 1.0

    print("  evaluation_observability_traces: PASS")


# ===========================================================================
# 6. ENV
# ===========================================================================

def test_env():
    """Test environment configuration."""
    import tensoreval as te

    # Simple env
    env = te.Env.from_dict({"system_prompt": "You are helpful."})
    assert env.system_prompt == "You are helpful."

    # With Docker config
    env2 = te.Env.from_dict({
        "system_prompt": "test",
        "agent": {"image": "python:3.12", "port": 8000},
        "mcp": {"image": "node:18", "port": 9000},
    })
    assert env2.agent is not None
    assert env2.mcp is not None

    # With direct URLs
    env3 = te.Env.from_dict({
        "system_prompt": "test",
        "agent_url": "http://localhost:8000",
        "mcp_url": "http://localhost:9000/mcp",
    })
    assert env3.agent_url == "http://localhost:8000"

    print("  env: PASS")


# ===========================================================================
# 7. TOOLS — Docker Compose
# ===========================================================================

def test_docker_compose_yaml():
    """Test Docker Compose YAML generation."""
    import tensoreval as te

    compose = te.DockerCompose(services={
        "agent": {
            "image": "python:3.12-slim",
            "port": 8000,
            "command": "python /app/agent.py",
            "env": {"API_KEY": "secret"},
        },
        "mcp": {
            "image": "node:18-slim",
            "port": 9000,
        },
    })

    yaml = compose._generate_compose_yaml()

    assert "services:" in yaml
    assert "agent:" in yaml
    assert "mcp:" in yaml
    assert "python:3.12-slim" in yaml
    assert "127.0.0.1:8000:8000" in yaml
    assert "127.0.0.1:9000:9000" in yaml
    assert "API_KEY=secret" in yaml
    assert "python /app/agent.py" in yaml

    print("  docker_compose_yaml: PASS")


def test_docker_compose_urls():
    """Test Docker Compose URL helpers."""
    import tensoreval as te

    compose = te.DockerCompose(services={
        "agent": {"image": "python:3.12", "port": 8000},
        "mcp-server": {"image": "node:18", "port": 9000},
    })

    assert compose.get_agent_url() == "http://localhost:8000"
    assert compose.get_mcp_url() == "http://localhost:9000/mcp"

    print("  docker_compose_urls: PASS")


# ===========================================================================
# 8. METRICS — Voice
# ===========================================================================

def test_voice_metrics():
    """Test voice metrics computation."""
    from tensoreval.metrics.voice import VoiceMetrics

    vm = VoiceMetrics()

    # WER — exact match
    transcript = [{"role": "assistant", "content": "the cat sat on the mat"}]
    wer = vm._compute_wer(transcript, reference="the cat sat on the mat")
    assert wer == 0.0

    # WER — one word wrong
    wer2 = vm._compute_wer(transcript, reference="the dog sat on the mat")
    assert wer2 > 0.0
    assert wer2 < 1.0

    # WER — no reference
    wer3 = vm._compute_wer(transcript, reference="")
    assert wer3 == 0.0

    # TTFT
    transcript2 = [
        {"role": "user", "content": "hello", "start_time": 0, "end_time": 1.0},
        {"role": "assistant", "content": "hi", "start_time": 1.5, "end_time": 2.0},
    ]
    ttft = vm._compute_ttft(transcript2)
    assert ttft == 0.5  # 1.5 - 1.0

    # Talk ratio
    ratio = vm._compute_talk_ratio(transcript2)
    assert 0 < ratio < 1

    # Interruptions
    transcript3 = [
        {"role": "user", "content": "hello", "start_time": 0, "end_time": 2.0},
        {"role": "assistant", "content": "hi", "start_time": 1.5, "end_time": 3.0},  # Interrupts
    ]
    interruptions = vm._compute_interruptions(transcript3)
    assert interruptions == 1

    # WPM
    wpm = vm._compute_wpm(transcript2)
    assert wpm > 0

    print("  voice_metrics: PASS")


def test_indian_language_metrics():
    """Test Indian language metrics."""
    from tensoreval.metrics.voice import IndianLanguageMetrics

    ilm = IndianLanguageMetrics()

    # Code-switching detection
    transcript = [{"role": "assistant", "content": "Main aapko नमस्ते कहता हूँ"}]
    cs = ilm._detect_code_switching(transcript)
    assert cs == 1.0  # Detected (Latin + Devanagari)

    transcript2 = [{"role": "assistant", "content": "Hello world"}]
    cs2 = ilm._detect_code_switching(transcript2)
    assert cs2 == 0.0  # Not detected (Latin only)

    print("  indian_language_metrics: PASS")


# ===========================================================================
# 9. UTILS — Parsing
# ===========================================================================

def test_utils_parsing():
    """Test answer extraction utilities."""
    import tensoreval as te

    # Boxed answer
    assert te.extract_boxed_answer("The answer is \\boxed{42}") == "42"
    assert te.extract_boxed_answer("Result: \\boxed{180}") == "180"
    assert te.extract_boxed_answer("No boxed answer") == "No boxed answer"

    # Hash answer (GSM8K)
    assert te.extract_hash_answer("Solution here\n#### 42") == "42"
    assert te.extract_hash_answer("Just text") == "Just text"

    print("  utils_parsing: PASS")


# ===========================================================================
# 10. MCP TOOLS
# ===========================================================================

def test_mcp_tools():
    """Test MCP tool classes."""
    import tensoreval as te

    server = te.MCPServer(url="http://localhost:9000/mcp", name="test-server")
    assert server.name == "test-server"
    assert server.url == "http://localhost:9000/mcp"

    registry = te.MCPToolRegistry()
    registry.add_server("my_server", server)
    assert "my_server" in registry.servers

    # Unknown server raises
    try:
        asyncio.run(registry.call_tool("nonexistent", "tool", {}))
        assert False
    except ValueError:
        pass

    print("  mcp_tools: PASS")


# ===========================================================================
# RUNNER
# ===========================================================================

def run_all():
    """Run all tests."""
    print()
    print("TensorEval SDK Test Suite v0.6.0")
    print("=" * 50)

    tests = [
        test_types,
        test_datasets,
        test_rubric_grader_simple,
        test_rubric_grader_llm_fallback,
        test_ruler_grader_single,
        test_ruler_grader_group,
        test_grader_base,
        test_agents,
        test_evaluation_with_function_agent,
        test_evaluation_with_agent_class,
        test_evaluation_results_save_load,
        test_evaluation_summary,
        test_evaluation_observability_traces,
        test_env,
        test_docker_compose_yaml,
        test_docker_compose_urls,
        test_voice_metrics,
        test_indian_language_metrics,
        test_utils_parsing,
        test_mcp_tools,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  {t.__name__}: FAIL - {e}")
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
