"""Contract tests for the clean TensorEval SDK surface."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_dataset_loads_jsonl_with_optional_rubrics_and_reference():
    import tensoreval as te

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({
            "id": "case_1",
            "query": "Refund order O123",
            "reference_answer": "Check order and policy",
            "rubrics": [
                {"id": "policy", "description": "Must check policy", "weight": 0.7},
                {"name": "tone", "criteria": "Must be clear", "weight": 0.3},
            ],
        }) + "\n")
        f.write(json.dumps({"query": "No reference required"}) + "\n")
        path = f.name

    try:
        dataset = te.Dataset.from_jsonl(path)
    finally:
        os.unlink(path)

    assert len(dataset) == 2
    assert dataset[0].id == "case_1"
    assert dataset[0].query == "Refund order O123"
    assert dataset[0].reference_answer == "Check order and policy"
    assert dataset[0].rubrics[0].id == "policy"
    assert dataset[0].rubrics[0].description == "Must check policy"
    assert dataset[1].reference_answer == ""
    assert dataset[1].rubrics == []


def test_endpoint_env_normalizes_optional_mcp_urls():
    import tensoreval as te

    env = te.Env.from_endpoint(
        agent_url="http://localhost:8000/v1/chat/completions",
        mcp_urls="http://localhost:8001/mcp",
    )

    assert env.kind == "endpoint"
    assert env.agent_url == "http://localhost:8000/v1/chat/completions"
    assert env.mcp_urls == ["http://localhost:8001/mcp"]
    assert env.supports_parallel_workers is False


def test_evaluation_calls_openai_shaped_endpoint_and_captures_trace():
    import tensoreval as te
    import tensoreval.evaluation as evaluation_mod

    response = {
        "choices": [{"message": {"role": "assistant", "content": "Refund approved."}}],
        "trace": {
            "schema_version": "tensoreval.trace.v1",
            "steps": [
                {"id": "s1", "type": "message", "role": "assistant", "content": "Checking order."},
                {"id": "s2", "type": "tool_call", "name": "lookup_order", "arguments": {"order_id": "O123"}},
                {"id": "s3", "type": "tool_result", "tool_call_id": "s2", "name": "lookup_order", "result": {"status": "delivered"}},
            ],
        },
    }
    captured: dict = {}

    def fake_post_json(url: str, body: dict, timeout: float):
        captured["url"] = url
        captured["body"] = body
        captured["timeout"] = timeout
        return response

    dataset = te.Dataset.from_dicts([{
        "id": "case_1",
        "query": "Refund order O123",
        "reference_answer": "Refund approved",
        "rubrics": [{"id": "correctness", "description": "Must approve refund", "weight": 1.0}],
    }])
    env = te.Env.from_endpoint(
        agent_url="http://agent.local/v1/chat/completions",
        mcp_urls=["http://localhost:8001/mcp"],
    )

    previous = evaluation_mod._post_json
    evaluation_mod._post_json = fake_post_json
    try:
        result = te.Evaluation.run(
            dataset=dataset,
            env=env,
            grader=te.AgenticGrader(pass_threshold=0.8),
            workers=1,
            timeout=5,
            agent_config={"max_steps": 20, "mode": "eval"},
        )
    finally:
        evaluation_mod._post_json = previous

    request = captured["body"]
    assert captured["url"] == "http://agent.local/v1/chat/completions"
    assert captured["timeout"] == 5
    assert request["messages"] == [{"role": "user", "content": "Refund order O123"}]
    assert request["agent_config"] == {"max_steps": 20, "mode": "eval"}
    assert request["metadata"]["sample_id"] == "case_1"
    assert request["mcp_servers"] == [{"name": "mcp_0", "url": "http://localhost:8001/mcp"}]

    assert result.summary()["avg_reward"] == 1.0
    assert result.runs[0].final_response == "Refund approved."
    assert result.runs[0].trace["steps"][1]["type"] == "tool_call"
    assert result.runs[0].grader_trace["steps"]


def test_endpoint_env_supports_multiple_workers():
    import tensoreval as te
    import tensoreval.evaluation as evaluation_mod

    response = {"choices": [{"message": {"content": "ok"}}]}

    def fake_post_json(url: str, body: dict, headers=None, timeout: float = 60):
        return response

    dataset = te.Dataset.from_dicts([
        {"query": "Q1", "reference_answer": "ok"},
        {"query": "Q2", "reference_answer": "ok"},
        {"query": "Q3", "reference_answer": "ok"},
    ])
    env = te.Env.from_endpoint(agent_url="http://localhost:8000/v1/chat/completions")

    previous = evaluation_mod._post_json
    evaluation_mod._post_json = fake_post_json
    try:
        result = te.Evaluation.run(dataset=dataset, env=env, grader=te.AgenticGrader(), workers=3)
        assert len(result.runs) == 3
        assert all(run.error is None for run in result.runs)
    finally:
        evaluation_mod._post_json = previous


def test_result_report_includes_trace_steps():
    import tensoreval as te

    run = te.EvaluationRun(
        sample_id="case_1",
        query="Refund order O123",
        final_response="Refund approved.",
        reward=1.0,
        passed=True,
        latency_ms=12.0,
        trace={"steps": [{"type": "tool_call", "name": "lookup_order", "arguments": {"order_id": "O123"}}]},
        grader_trace={"steps": [{"type": "tool_call", "name": "verify_policy", "arguments": {"order_id": "O123"}}]},
    )
    result = te.EvaluationResult(runs=[run], metadata={"env": {"agent_url": "http://agent"}})

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name

    try:
        result.report(path)
        html = Path(path).read_text()
    finally:
        os.unlink(path)

    assert "TensorEval Report" in html
    assert "lookup_order" in html
    assert "verify_policy" in html
    assert "Refund approved." in html


def test_provider_config_defaults_cover_major_providers():
    from tensoreval.tool_loop import ProviderConfig

    assert ProviderConfig.from_name("openai").base_url == "https://api.openai.com/v1"
    assert ProviderConfig.from_name("openrouter").base_url == "https://openrouter.ai/api/v1"
    assert ProviderConfig.from_name("ollama").base_url == "http://localhost:11434/v1"
    assert ProviderConfig.from_name("anthropic").base_url == "https://api.anthropic.com"


def test_agentic_grader_runs_openai_compatible_tool_loop_with_mcp():
    import tensoreval as te
    import tensoreval.tool_loop as tool_loop_mod

    calls: list[tuple[str, dict]] = []

    def fake_post_json(url: str, body: dict, headers=None, timeout: float = 60):
        calls.append((url, body))
        if body.get("method") == "tools/list":
            return {
                "result": {
                    "tools": [
                        {
                            "name": "lookup_order",
                            "description": "Look up an order",
                            "inputSchema": {"type": "object", "properties": {"order_id": {"type": "string"}}},
                        }
                    ]
                }
            }
        if body.get("method") == "tools/call":
            return {"result": {"structuredContent": {"status": "delivered"}}}

        messages = body["messages"]
        if not any(message.get("role") == "tool" for message in messages):
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup_order", "arguments": "{\"order_id\":\"O123\"}"},
                        }],
                    }
                }]
            }
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "{\"reward\":0.9,\"passed\":true,\"rubric_scores\":{\"correctness\":{\"score\":0.9,\"reason\":\"Verified with tool\"}},\"reasoning_summary\":\"Policy and order state match.\"}",
                }
            }]
        }

    sample = te.Dataset.from_dicts([{
        "id": "case_1",
        "query": "Refund order O123",
        "reference_answer": "Refund approved",
        "rubrics": [{"id": "correctness", "description": "Must verify refund eligibility", "weight": 1.0}],
    }])[0]

    previous = tool_loop_mod._post_json
    tool_loop_mod._post_json = fake_post_json
    try:
        grade = te.AgenticGrader(
            provider="openai",
            model="gpt-4.1",
            api_key="sk-test",
            pass_threshold=0.8,
        ).grade(
            sample=sample,
            final_response="Refund approved.",
            agent_trace={"steps": []},
            mcp_urls=["http://mcp.local/mcp"],
        )
    finally:
        tool_loop_mod._post_json = previous

    assert grade["reward"] == 0.9
    assert grade["passed"] is True
    assert grade["rubric_scores"]["correctness"]["reason"] == "Verified with tool"
    assert grade["grader_trace"]["steps"][0]["type"] == "tool_call"
    assert any(call[0] == "https://api.openai.com/v1/chat/completions" for call in calls)
    assert any(call[1].get("method") == "tools/call" for call in calls)


def test_tool_loop_routes_major_providers_to_expected_apis():
    from tensoreval.tool_loop import ProviderConfig, ToolLoopAgent
    import tensoreval.tool_loop as tool_loop_mod

    calls: list[str] = []

    def fake_post_json(url: str, body: dict, headers=None, timeout: float = 60):
        calls.append(url)
        if url.endswith("/v1/messages"):
            return {"content": [{"type": "text", "text": "{\"reward\":1,\"passed\":true,\"rubric_scores\":{},\"reasoning_summary\":\"ok\"}"}]}
        return {"choices": [{"message": {"content": "{\"reward\":1,\"passed\":true,\"rubric_scores\":{},\"reasoning_summary\":\"ok\"}"}}]}

    previous = tool_loop_mod._post_json
    tool_loop_mod._post_json = fake_post_json
    try:
        for provider in ("openai", "openrouter", "ollama", "anthropic"):
            config = ProviderConfig.from_name(provider, api_key="key")
            result = ToolLoopAgent(config, instructions="grade").run("prompt", mcp_urls=[])
            assert result["reward"] == 1
    finally:
        tool_loop_mod._post_json = previous

    assert "https://api.openai.com/v1/chat/completions" in calls
    assert "https://openrouter.ai/api/v1/chat/completions" in calls
    assert "http://localhost:11434/v1/chat/completions" in calls
    assert "https://api.anthropic.com/v1/messages" in calls


def test_result_save_load_roundtrip():
    import tensoreval as te

    run = te.EvaluationRun(
        sample_id="case_1",
        query="Refund order O123",
        final_response="Refund approved.",
        reward=0.9,
        passed=True,
        latency_ms=42.0,
        trace={"steps": [{"type": "tool_call", "name": "lookup_order"}]},
        rubric_scores={"correctness": {"score": 0.9, "reason": "verified"}},
        reasoning="Policy and order state match.",
    )
    result = te.EvaluationResult(runs=[run], metadata={"env": {"agent_url": "http://agent"}})

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        result.save(path)
        loaded = te.EvaluationResult.load(path)
        assert len(loaded.runs) == 1
        assert loaded.runs[0].sample_id == "case_1"
        assert loaded.runs[0].query == "Refund order O123"
        assert loaded.runs[0].final_response == "Refund approved."
        assert loaded.runs[0].reward == 0.9
        assert loaded.runs[0].passed is True
        assert loaded.runs[0].rubric_scores["correctness"]["score"] == 0.9
    finally:
        os.unlink(path)


def test_sample_ids_are_stable_across_runs():
    from tensoreval.types import Sample

    s1 = Sample(query="What is 2+2?")
    s2 = Sample(query="What is 2+2?")
    assert s1.id == s2.id, "Sample ID should be deterministic"
    assert s1.id.startswith("case_")
    assert len(s1.id) == 13  # "case_" + 8 hex chars

    # Different queries should get different IDs
    s3 = Sample(query="Hello world")
    assert s1.id != s3.id


def test_dashboard_client_disabled_without_api_key():
    import tensoreval as te

    # Ensure no API key is set
    old_key = os.environ.pop("TENSOREVAL_API_KEY", None)
    try:
        client = te.DashboardClient()
        assert client.enabled is False
    finally:
        if old_key:
            os.environ["TENSOREVAL_API_KEY"] = old_key


def test_agent_result_coerce():
    from tensoreval.types import AgentResult

    # str → AgentResult with empty trace
    r = AgentResult.coerce("hello")
    assert r.response == "hello"
    assert r.tool_trace == []

    # AgentResult → as-is
    r = AgentResult.coerce(AgentResult(response="hi", tool_trace=[{"name": "tool1"}]))
    assert r.response == "hi"
    assert r.tool_trace == [{"name": "tool1"}]

    # dict → extract response + tool_trace
    r = AgentResult.coerce({"response": "ok", "tool_trace": [{"name": "t"}]})
    assert r.response == "ok"
    assert r.tool_trace == [{"name": "t"}]

    # dict with alternate keys
    r = AgentResult.coerce({"content": "yes", "trace": [{"name": "t2"}]})
    assert r.response == "yes"
    assert r.tool_trace == [{"name": "t2"}]


def test_evaluation_direct_callable_mode():
    """Test Evaluation.run with a direct agent callable instead of an HTTP endpoint."""
    import tensoreval as te

    dataset = te.Dataset.from_dicts([
        {"query": "What is 2+2?", "reference_answer": "4"},
        {"query": "Capital of France?", "reference_answer": "Paris"},
    ])

    async def my_agent(query: str) -> str:
        if "2+2" in query:
            return "4"
        return "London"  # wrong answer

    grader = te.AgenticGrader(pass_threshold=0.8)
    results = te.Evaluation.run(dataset=dataset, agent=my_agent, grader=grader, workers=2)

    assert len(results.runs) == 2
    assert results.runs[0].final_response == "4"
    assert results.runs[1].final_response == "London"
    assert results.runs[0].error is None
    assert results.runs[1].error is None


def test_evaluation_direct_callable_with_agent_result():
    """Test direct callable returning AgentResult with tool_trace."""
    import tensoreval as te

    dataset = te.Dataset.from_dicts([
        {"query": "Check order O123", "reference_answer": "Order found"},
    ])

    async def agent_with_trace(query: str) -> te.AgentResult:
        return te.AgentResult(
            response="Order O123 found, status: delivered",
            tool_trace=[{"name": "lookup_order", "arguments": {"order_id": "O123"}, "result": {"status": "delivered"}}],
        )

    grader = te.AgenticGrader(pass_threshold=0.8)
    results = te.Evaluation.run(dataset=dataset, agent=agent_with_trace, grader=grader)

    assert len(results.runs) == 1
    assert results.runs[0].final_response == "Order O123 found, status: delivered"
    assert results.runs[0].trace is not None
    assert results.runs[0].trace["steps"][0]["name"] == "lookup_order"


def test_evaluation_rejects_both_env_and_agent():
    import tensoreval as te

    dataset = te.Dataset.from_dicts([{"query": "hi"}])
    env = te.Env.from_endpoint(agent_url="http://localhost:8000")

    async def my_agent(query: str) -> str:
        return "hi"

    try:
        te.Evaluation.run(dataset=dataset, env=env, agent=my_agent)
        assert False, "Should have raised"
    except ValueError as exc:
        assert "both" in str(exc).lower()


def run_all():
    tests = [
        test_dataset_loads_jsonl_with_optional_rubrics_and_reference,
        test_endpoint_env_normalizes_optional_mcp_urls,
        test_evaluation_calls_openai_shaped_endpoint_and_captures_trace,
        test_endpoint_env_supports_multiple_workers,
        test_result_report_includes_trace_steps,
        test_provider_config_defaults_cover_major_providers,
        test_agentic_grader_runs_openai_compatible_tool_loop_with_mcp,
        test_tool_loop_routes_major_providers_to_expected_apis,
        test_result_save_load_roundtrip,
        test_sample_ids_are_stable_across_runs,
        test_dashboard_client_disabled_without_api_key,
        test_agent_result_coerce,
        test_evaluation_direct_callable_mode,
        test_evaluation_direct_callable_with_agent_result,
        test_evaluation_rejects_both_env_and_agent,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  {test.__name__}: PASS")
        except Exception as exc:
            failed += 1
            import traceback
            print(f"  {test.__name__}: FAIL - {exc}")
            traceback.print_exc()
    print(f"\nResults: {len(tests) - failed} passed, {failed} failed, {len(tests)} total")
    return failed == 0


if __name__ == "__main__":
    raise SystemExit(0 if run_all() else 1)
