"""Comprehensive test suite for TensorEval SDK v0.6.0.

Tests are organized by module:
1. Types — data structures
2. Datasets — loading and conversion
3. Graders — scoring logic (no API calls)
4. Agents — agent abstraction
5. Evaluation — full pipeline (with mock agent)
6. Env — environment config
7. Tools — Docker compose YAML generation
8. Utils — parsing helpers
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

    # Top-level scenario metadata is preserved for dashboard grouping.
    ds_category = te.Datasets.load_from_dict([{
        "query": "Handle duplicate charge",
        "reference_answer": "Refund duplicate charge",
        "category": "billing",
        "difficulty": "easy",
    }])
    assert ds_category[0].metadata["category"] == "billing"
    assert ds_category[0].metadata["difficulty"] == "easy"

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

def test_agent_grader_fallback():
    """Test AgentGrader deterministic fallback path (no Gateway call required)."""
    import tensoreval as te

    async def run():
        grader = te.AgentGrader(fallback_on_error=True)

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
    print("  agent_grader_fallback: PASS")


def test_vercel_agent_grader_export():
    """Test the VercelAgentGrader public export and alias."""
    import tensoreval as te

    grader = te.VercelAgentGrader(model="openai/gpt-5.5", fallback_on_error=True)
    assert isinstance(grader, te.AgentGrader)
    assert grader.model == "openai/gpt-5.5"

    print("  vercel_agent_grader_export: PASS")


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

    # resolve_agent with http URL → raises (need wrapper function)
    try:
        resolve_agent("http://localhost:8000")
        assert False, "Should have raised"
    except ValueError:
        pass

    # resolve_agent with anthropic: prefix → raises (need wrapper function)
    try:
        resolve_agent("anthropic:mimo-v2.5-pro", api_key="key", base_url="url")
        assert False, "Should have raised"
    except ValueError:
        pass

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

    grader = te.AgentGrader(fallback_on_error=True)
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

    results = te.Evaluation.run(ds, te.AgentGrader(fallback_on_error=True), agent=MathAgent())
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
        results = te.Evaluation.run(ds, te.AgentGrader(fallback_on_error=True), agent=mock_agent, model="gpt-4o")
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
# 6. ENVIRONMENT
# ===========================================================================

def test_env():
    """Test environment configuration."""
    import tensoreval as te

    # Simple env
    env = te.Environment(system_prompt="You are helpful.")
    assert env.system_prompt == "You are helpful."
    assert te.Environment is te.Environment

    # With Docker config
    env2 = te.Environment(
        system_prompt="test",
        agent={"image": "python:3.12", "port": 8000},
        mcp={"image": "node:18", "port": 9000},
    )
    assert env2.agent is not None
    assert env2.mcp is not None

    # With direct URLs
    env3 = te.Environment(
        system_prompt="test",
        agent_url="http://localhost:8000",
        mcp_url="http://localhost:9000/mcp",
    )
    assert env3.agent_url == "http://localhost:8000"

    # Multiple MCP servers
    env4 = te.Environment(
        mcp_servers=[
            "http://localhost:9000/mcp",
            {"name": "crm", "url": "http://localhost:9001/mcp"},
        ],
        mcp_port=8001,
    )
    assert len(env4.mcp_servers) == 2
    assert env4.mcp_port == 8001

    print("  env: PASS")


# ===========================================================================
# 7. ENVIRONMENT — Docker Compose
# ===========================================================================

def test_docker_compose_yaml():
    """Test Environment Docker Compose YAML generation."""
    import tensoreval as te

    env = te.Environment(
        agent={
            "image": "python:3.12-slim",
            "port": 8000,
            "command": "python /app/agent.py",
            "env": {"API_KEY": "secret"},
        },
        mcp={
            "image": "node:18-slim",
            "port": 9000,
        },
    )

    yaml = env.to_compose_yaml()

    assert "services:" in yaml
    assert "agent:" in yaml
    assert "mcp-server:" in yaml
    assert "python:3.12-slim" in yaml
    assert "127.0.0.1:8000:8000" in yaml
    assert "127.0.0.1:9000:9000" in yaml
    assert "API_KEY=secret" in yaml
    assert "python /app/agent.py" in yaml

    print("  docker_compose_yaml: PASS")


def test_docker_compose_urls():
    """Test Environment URL helpers."""
    import tensoreval as te

    env = te.Environment(
        agent={"image": "python:3.12", "port": 8000},
        mcp={"image": "node:18", "port": 9000},
    )

    assert env.get_agent_url() == "http://localhost:8000"
    assert env.get_mcp_url() == "http://localhost:9000/mcp"

    print("  docker_compose_urls: PASS")


# ===========================================================================
# 8. UTILS — Parsing
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
    from tensoreval.tools.mcp import MCPServer, MCPToolRegistry

    server = MCPServer(url="http://localhost:9000/mcp", name="test-server")
    assert server.name == "test-server"
    assert server.url == "http://localhost:9000/mcp"

    registry = MCPToolRegistry()
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
# 11. MCP TOOL CALLING + ENV WIRING
# ===========================================================================

def test_mcp_call_by_name():
    """Test MCPToolRegistry.call_tool_by_name for unknown tools."""
    from tensoreval.tools.mcp import MCPServer, MCPToolRegistry

    registry = MCPToolRegistry()
    result = asyncio.run(registry.call_tool_by_name("nonexistent", {}))
    assert "error" in result

    server = MCPServer(url="http://localhost:9000/mcp")
    registry.add_server("srv", server)
    result = asyncio.run(registry.call_tool_by_name("missing_tool", {}))
    assert "error" in result

    print("  mcp_call_by_name: PASS")


def test_env_docker_import_fixed():
    """Verify internal Docker import points to tools.docker (not deleted docker_compose)."""
    import tensoreval.tools.docker as docker_mod
    import tensoreval as te

    assert hasattr(docker_mod, "DockerCompose")
    assert not hasattr(te, "DockerCompose")

    try:
        import tensoreval.docker_compose
        assert False, "Old module should not exist"
    except ImportError:
        pass

    print("  env_docker_import: PASS")


def test_env_agent_url_port_extraction():
    """Test that Environment URL port extraction works."""
    import tensoreval as te

    env = te.Environment(agent_url="http://localhost:8000")
    port = int(env.agent_url.rsplit(":", 1)[-1].split("/")[0])
    assert port == 8000

    env2 = te.Environment(mcp_url="http://localhost:9000/mcp")
    mcp_port = int(env2.mcp_url.rsplit(":", 1)[-1].split("/")[0])
    assert mcp_port == 9000

    print("  env_agent_url_port_extraction: PASS")


def test_eval_starts_mcp_only_env():
    """Test evaluation starts Env when only an MCP Docker service is configured."""
    from tensoreval.evaluation import Evaluation
    import tensoreval as te

    class FakeEnv:
        system_prompt = None
        agent_url = None
        mcp_url = "http://localhost:9000/mcp"
        mcp_servers = []
        tools = []
        agent = None
        mcp = {"image": "node:18", "port": 9000}
        started = False

        async def start(self):
            self.started = True

        async def stop(self):
            pass

    async def agent(query: str) -> str:
        return "4"

    env = FakeEnv()
    ds = te.Datasets.load_from_dict([{"query": "2+2", "reference_answer": "4"}])
    results = te.Evaluation.run(ds, te.AgentGrader(fallback_on_error=True), agent=agent, env=env)

    assert env.started is True
    assert results.runs[0].reward == 1.0

    print("  eval_starts_mcp_only_env: PASS")


def test_openai_agent_tool_loop_config():
    """Test OpenAIAgent accepts max_tool_rounds parameter."""
    import tensoreval as te

    agent = te.OpenAIAgent(model="gpt-4o", api_key="sk-test", max_tool_rounds=5)
    assert agent.max_tool_rounds == 5

    agent2 = te.OpenAIAgent(model="gpt-4o", api_key="sk-test")
    assert agent2.max_tool_rounds == 10

    print("  openai_agent_tool_loop_config: PASS")


def test_evaluation_mcp_tools_in_context():
    """Test that MCP tools are wired into agent Context during evaluation."""
    import tensoreval as te
    from tensoreval.agents import Agent, Context
    from tensoreval.evaluation import _evaluate_single

    captured: dict = {}

    class CaptureAgent(Agent):
        async def run(self, query: str, context: Context) -> str:
            captured["tools"] = context.tools
            captured["has_registry"] = context.mcp_registry is not None
            return "4"

    ds = te.Datasets.load_from_dict([{"query": "2+2", "reference_answer": "4"}])
    fake_tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]

    asyncio.run(_evaluate_single(
        0, ds, te.AgentGrader(fallback_on_error=True), CaptureAgent(),
        te.EvalConfig(model="test"),
        mcp_tools=fake_tools,
        mcp_registry="fake_registry",
    ))

    assert len(captured["tools"]) == 1
    assert captured["has_registry"] is True

    print("  evaluation_mcp_tools_in_context: PASS")


def test_tool_registry_local_tools_and_mcp_servers():
    """Test local Python tools and multiple MCP servers normalize into one registry."""
    import tensoreval as te
    from tensoreval.evaluation import _build_tool_registry

    def lookup_order(order_id: str) -> dict:
        """Look up an order by ID."""
        return {"order_id": order_id, "status": "shipped"}

    def count_items(items: list[str], metadata: dict | None = None) -> int:
        """Count item IDs."""
        return len(items) + (1 if metadata else 0)

    env = te.Environment(
        mcp_servers=[
            "http://localhost:9000/mcp",
            {"name": "crm", "url": "http://localhost:9001/mcp"},
        ]
    )
    config = te.EvalConfig(tools=[lookup_order, count_items], mcp_port=8001)
    registry = _build_tool_registry(config, env)

    assert registry is not None
    assert "lookup_order" in registry.local_tools
    assert len(registry.servers) == 3

    tools = registry.to_openai_tools()
    assert tools[0]["function"]["name"] == "lookup_order"
    assert tools[0]["function"]["parameters"]["required"] == ["order_id"]
    assert tools[1]["function"]["parameters"]["properties"]["items"]["type"] == "array"
    assert tools[1]["function"]["parameters"]["properties"]["metadata"]["type"] == "object"

    result = asyncio.run(registry.call_tool_by_name("lookup_order", {"order_id": "O-1"}))
    assert result == {"order_id": "O-1", "status": "shipped"}

    print("  tool_registry_local_tools_and_mcp_servers: PASS")


def test_vercel_grader_tool_helpers():
    """Test grader-side Responses tool conversion and execution helpers."""
    import tensoreval as te
    from tensoreval.tools.mcp import MCPToolRegistry
    from tensoreval.graders.agent_grader import (
        _execute_grader_tool_call,
        _json_schema_unsupported,
        _response_function_calls,
        _responses_text_format,
        _to_responses_tools,
    )

    def check_policy(topic: str) -> str:
        """Check policy text."""
        return f"policy for {topic}"

    registry = MCPToolRegistry()
    registry.add_local_tool(check_policy)
    openai_tools = registry.to_openai_tools()
    response_tools = _to_responses_tools(openai_tools, True, "low")

    assert response_tools[0] == {"type": "web_search", "search_context_size": "low"}
    assert response_tools[1]["type"] == "function"
    assert response_tools[1]["name"] == "check_policy"

    class FakeResponse:
        output = [{
            "type": "function_call",
            "name": "check_policy",
            "call_id": "call_1",
            "arguments": '{"topic":"refunds"}',
        }]

    calls = _response_function_calls(FakeResponse())
    assert calls == [{"name": "check_policy", "call_id": "call_1", "arguments": {"topic": "refunds"}}]

    result = asyncio.run(_execute_grader_tool_call(calls[0], registry))
    assert result == "policy for refunds"

    err = Exception("text.format type 'json_schema' is not supported, only 'text' and 'json_object' are allowed")
    assert _json_schema_unsupported(err) is True
    assert _responses_text_format("json_object") == {"format": {"type": "json_object"}}

    @te.tool(name="decorated_policy", description="Decorated policy checker")
    def decorated(topic: str) -> str:
        return topic

    registry2 = MCPToolRegistry()
    registry2.add_local_tool(decorated)
    decorated_tool = registry2.to_openai_tools()[0]["function"]
    assert decorated_tool["name"] == "decorated_policy"
    assert decorated_tool["description"] == "Decorated policy checker"

    @te.tool
    def bare_decorated(topic: str) -> str:
        return topic

    registry3 = MCPToolRegistry()
    registry3.add_local_tool(bare_decorated)
    assert registry3.to_openai_tools()[0]["function"]["name"] == "bare_decorated"

    print("  vercel_grader_tool_helpers: PASS")


def test_langchain_integration_wrap_agent():
    """Test wrapping an existing LangChain-style runnable without LangChain installed."""
    import tensoreval as te
    from tensoreval.agents import Context

    class FakeRunnable:
        async def ainvoke(self, payload, config=None):
            for cb in (config or {}).get("callbacks", []):
                cb.on_tool_start({"name": "lookup"}, "order=O-1", run_id="tool-1")
                cb.on_tool_end("shipped", run_id="tool-1")
            return {"output": f"handled {payload['input']}"}

    async def run():
        agent = te.integrations.langchain.wrap_agent(FakeRunnable())
        context = Context(query="refund")
        result = await agent.run("refund", context)
        assert result == "handled refund"
        assert context.metadata["tool_trace"] == [{"tool": "lookup", "args": "order=O-1", "result": "shipped"}]

    asyncio.run(run())

    print("  langchain_integration_wrap_agent: PASS")


def test_verification_grader():
    """Test VerificationGrader with a mocked OpenAI chat completion."""
    import tensoreval as te
    from tensoreval.graders import verification_grader as vg

    grader = te.VerificationGrader(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:9999/v1",
        fallback_on_error=False,
    )

    class FakeMessage:
        content = '{"rubric_scores": [{"rubric_name": "correctness", "score": 1.0, "weight": 1.0, "reasoning": "verified"}], "grader_reasoning": "ok"}'

    class FakeChoice:
        message = FakeMessage()

    class FakeResult:
        choices = [FakeChoice()]

    async def fake_create(**kwargs):
        return FakeResult()

    original_create = vg.AsyncOpenAI
    vg.AsyncOpenAI = lambda **kw: type("FakeClient", (), {"chat": type("FakeChat", (), {"completions": type("FakeCompletions", (), {"create": staticmethod(fake_create)})})})()
    try:
        async def run():
            state = {
                "query": "What is 2+2?",
                "answer": "4",
                "completion": [{"role": "assistant", "content": "4"}],
                "info": {
                    "rubrics": [{"name": "correctness", "criteria": "Must be 4", "weight": 1.0}],
                    "tool_trace": [],
                    "mcp_tools": [],
                },
                "tools": [],
                "tool_registry": None,
            }
            score = await grader.score(state)
            assert score == 1.0
            assert state["grader_result"]["passed"] is True

        asyncio.run(run())
    finally:
        vg.AsyncOpenAI = original_create

    print("  verification_grader: PASS")


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
        test_agent_grader_fallback,
        test_vercel_agent_grader_export,
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
        test_utils_parsing,
        test_mcp_tools,
        test_mcp_call_by_name,
        test_env_docker_import_fixed,
        test_env_agent_url_port_extraction,
        test_eval_starts_mcp_only_env,
        test_openai_agent_tool_loop_config,
        test_evaluation_mcp_tools_in_context,
        test_tool_registry_local_tools_and_mcp_servers,
        test_vercel_grader_tool_helpers,
        test_langchain_integration_wrap_agent,
        test_verification_grader,
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
