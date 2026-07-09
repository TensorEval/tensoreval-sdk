"""Evaluation runner and result reporting."""

from __future__ import annotations

import asyncio
import html
import json
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from tensoreval.agentic_grader import AgenticGrader
from tensoreval.client import DashboardClient
from tensoreval.dataset import Dataset
from tensoreval.env import Env


@dataclass
class EvaluationRun:
    """Result of evaluating a single sample — agent response, grade, and traces."""

    sample_id: str
    query: str
    final_response: str
    reward: float
    passed: bool
    latency_ms: float
    reference_answer: str = ""
    trace: dict[str, Any] | None = None
    grader_trace: dict[str, Any] | None = None
    rubric_scores: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    error: str | None = None


@dataclass
class EvaluationResult:
    """Aggregate results from an evaluation run.

    Provides ``summary()``, ``recommendations()``, ``report()`` (HTML),
    and ``save()``/``load()`` for JSON persistence.
    """

    runs: list[EvaluationRun]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def avg_reward(self) -> float:
        return sum(run.reward for run in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def pass_rate(self) -> float:
        return sum(1 for run in self.runs if run.passed) / len(self.runs) if self.runs else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "num_runs": len(self.runs),
            "avg_reward": round(self.avg_reward, 4),
            "pass_rate": round(self.pass_rate, 4),
            "pass_count": sum(1 for run in self.runs if run.passed),
            "fail_count": sum(1 for run in self.runs if not run.passed),
        }

    def recommendations(self) -> list[str]:
        if not self.runs:
            return ["No runs to analyze"]
        errors = [run for run in self.runs if run.error]
        failed = [run for run in self.runs if not run.passed]
        recs: list[str] = []
        if errors:
            recs.append(f"Fix agent endpoint errors first: {len(errors)}/{len(self.runs)} runs errored.")
        if failed:
            recs.append(f"Review failed cases: {len(failed)}/{len(self.runs)} did not meet the pass threshold.")
        if not recs:
            recs.append("Agent passed all evaluated cases.")
        return recs

    def report(self, path: str | Path = "report.html") -> str:
        path = Path(path)
        summary = self.summary()
        cards = "\n".join(_run_card(run) for run in self.runs)
        recs = "".join(f"<li>{html.escape(rec)}</li>" for rec in self.recommendations())
        body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>TensorEval Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; color: #111827; background: #f9fafb; }}
    .summary {{ display: flex; gap: 12px; margin: 16px 0 24px; }}
    .metric, .run {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; }}
    .metric b {{ display: block; font-size: 24px; }}
    .run {{ margin: 14px 0; }}
    .ok {{ color: #15803d; }} .fail {{ color: #b91c1c; }}
    pre {{ overflow: auto; background: #111827; color: #f9fafb; padding: 12px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>TensorEval Report</h1>
  <div class="summary">
    <div class="metric"><b>{summary["avg_reward"]:.2f}</b>Avg Reward</div>
    <div class="metric"><b>{summary["pass_rate"]:.0%}</b>Pass Rate</div>
    <div class="metric"><b>{summary["pass_count"]}/{summary["num_runs"]}</b>Passed</div>
  </div>
  <h2>Recommendations</h2>
  <ul>{recs}</ul>
  <h2>Runs</h2>
  {cards}
</body>
</html>"""
        path.write_text(body, encoding="utf-8")
        return str(path)

    def save(self, path: str | Path) -> None:
        """Serialize the evaluation result to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "runs": [asdict(run) for run in self.runs],
            "metadata": self.metadata,
        }, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationResult":
        """Load an evaluation result from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        runs = [EvaluationRun(**run) for run in data.get("runs", [])]
        return cls(runs=runs, metadata=data.get("metadata", {}))


class Evaluation:
    """Entry point for running an evaluation.

    Call ``Evaluation.run(dataset, env, grader)`` to evaluate an agent
    across a dataset with concurrent workers and optional live dashboard updates.
    """

    @staticmethod
    def run(
        dataset: Dataset,
        env: Env,
        grader: AgenticGrader | None = None,
        workers: int = 1,
        timeout: float = 120.0,
        agent_config: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        if env.kind != "endpoint":
            raise NotImplementedError(f"Environment kind '{env.kind}' is not supported — use Env.from_endpoint()")

        grader = grader or AgenticGrader()
        dashboard = DashboardClient()
        dashboard_run_id = _start_dashboard(dashboard, env, len(dataset))

        try:
            runs = asyncio.run(_grade_samples(
                dataset, env, grader, timeout, agent_config or {}, workers, dashboard, dashboard_run_id,
            ))
            result = EvaluationResult(
                runs=runs,
                metadata={"env": {"kind": env.kind, "agent_url": env.agent_url, "mcp_urls": env.mcp_urls}},
            )
            _finish_dashboard(dashboard, dashboard_run_id, result.summary())
            return result
        except Exception:
            _finish_dashboard(dashboard, dashboard_run_id, failed=True)
            raise


async def _grade_samples(
    dataset: Dataset,
    env: Env,
    grader: AgenticGrader,
    timeout: float,
    agent_config: dict[str, Any],
    workers: int,
    dashboard: DashboardClient,
    dashboard_run_id: str | None,
) -> list[EvaluationRun]:
    """Run agent calls and grading concurrently across all samples."""
    semaphore = asyncio.Semaphore(max(1, workers))

    async def grade_one(sample: Any) -> EvaluationRun:
        async with semaphore:
            return await asyncio.to_thread(_evaluate_sample, sample, env, grader, timeout, agent_config)

    results: list[EvaluationRun] = []
    for coro in asyncio.as_completed([grade_one(s) for s in dataset]):
        run = await coro
        if dashboard.enabled and dashboard_run_id:
            dashboard.append_result(dashboard_run_id, _run_to_payload(run), total_count=len(dataset))
        results.append(run)
    return results


def _start_dashboard(dashboard: DashboardClient, env: Env, total_count: int) -> str | None:
    """Start a live evaluation run on the dashboard if enabled."""
    if not dashboard.enabled:
        return None
    return dashboard.start_evaluation(
        model=env.config.get("model", "default"),
        total_count=total_count,
        name=f"SDK: {env.config.get('model', 'default')}",
    )


def _finish_dashboard(
    dashboard: DashboardClient,
    run_id: str | None,
    summary: dict[str, Any] | None = None,
    failed: bool = False,
) -> None:
    """Mark the dashboard evaluation as complete or failed."""
    if dashboard.enabled and run_id:
        dashboard.complete_evaluation(run_id, summary=summary, failed=failed)


def _run_to_payload(run: EvaluationRun) -> dict[str, Any]:
    """Convert an EvaluationRun to the dashboard ingest API format."""
    rubric_scores = [
        {
            "rubric_name": rid,
            "score": data.get("score", 0) if isinstance(data, dict) else 0,
            "reasoning": data.get("reason", "") if isinstance(data, dict) else "",
            "weight": data.get("weight", 1.0) if isinstance(data, dict) else 1.0,
        }
        for rid, data in (run.rubric_scores or {}).items()
    ]
    return {
        "sample_id": run.sample_id,
        "query": run.query,
        "answer": run.reference_answer,
        "response": run.final_response,
        "reward": run.reward,
        "latency_ms": run.latency_ms,
        "error": run.error,
        "metadata": {
            "grader_result": {
                "rubric_scores": rubric_scores,
                "grader_reasoning": run.reasoning,
            },
            "tool_trace": run.trace,
            "grader_trace": run.grader_trace,
        },
    }


def _evaluate_sample(
    sample: Any,
    env: Env,
    grader: AgenticGrader,
    timeout: float,
    agent_config: dict[str, Any],
) -> EvaluationRun:
    """Call the agent and grade its response for a single sample."""
    started = time.monotonic()
    try:
        response = _post_json(
            env.agent_url or "",
            _build_agent_payload(sample, env, agent_config),
            timeout,
        )
        final_response = _extract_final_response(response)
        trace = response.get("trace") if isinstance(response.get("trace"), dict) else None
        grade = grader.grade(sample, final_response, agent_trace=trace, mcp_urls=env.mcp_urls)
        return EvaluationRun(
            sample_id=sample.id,
            query=sample.query,
            final_response=final_response,
            reward=grade["reward"],
            passed=grade["passed"],
            latency_ms=(time.monotonic() - started) * 1000,
            reference_answer=sample.reference_answer,
            trace=trace,
            grader_trace=grade.get("grader_trace"),
            rubric_scores=grade["rubric_scores"],
            reasoning=grade["reasoning"],
        )
    except Exception as exc:
        # Preserve whatever agent response we already captured so the trace
        # is not lost when grading fails.
        return EvaluationRun(
            sample_id=sample.id,
            query=sample.query,
            final_response=locals().get("final_response", ""),
            reward=0.0,
            passed=False,
            latency_ms=(time.monotonic() - started) * 1000,
            reference_answer=sample.reference_answer,
            trace=locals().get("trace"),
            error=str(exc),
        )


def _build_agent_payload(sample: Any, env: Env, agent_config: dict[str, Any]) -> dict[str, Any]:
    """Build the request body for calling the agent endpoint."""
    payload: dict[str, Any] = {
        "model": env.config.get("model", "default"),
        "messages": [{"role": "user", "content": sample.query}],
        "agent_config": agent_config,
        "metadata": {"sample_id": sample.id},
    }
    if env.mcp_urls:
        payload["mcp_servers"] = env.mcp_servers()
    return payload


def _post_json(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _extract_final_response(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        if content is not None:
            return str(content)

    for key in ("final_response", "output_text", "answer", "response", "output", "result"):
        value = response.get(key)
        if isinstance(value, str):
            return value
    return ""


def _run_card(run: EvaluationRun) -> str:
    status = "PASS" if run.passed else "FAIL"
    status_class = "ok" if run.passed else "fail"
    trace = json.dumps(run.trace or {}, indent=2, default=str)
    grader_trace = json.dumps(run.grader_trace or {}, indent=2, default=str)
    return f"""<div class="run">
  <h3><span class="{status_class}">{status}</span> {html.escape(run.sample_id)} · reward={run.reward:.2f}</h3>
  <p><b>Query:</b> {html.escape(run.query)}</p>
  <p><b>Final response:</b> {html.escape(run.final_response)}</p>
  {f'<p class="fail"><b>Error:</b> {html.escape(run.error)}</p>' if run.error else ''}
  <h4>Agent Trace</h4>
  <pre>{html.escape(trace)}</pre>
  <h4>Grader Trace</h4>
  <pre>{html.escape(grader_trace)}</pre>
</div>"""
