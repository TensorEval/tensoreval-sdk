"""Evaluation runner and result reporting."""

from __future__ import annotations

import html
import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tensoreval.agentic_grader import AgenticGrader
from tensoreval.dataset import Dataset
from tensoreval.env import Env


@dataclass
class EvaluationRun:
    sample_id: str
    query: str
    final_response: str
    reward: float
    passed: bool
    latency_ms: float
    trace: dict[str, Any] | None = None
    grader_trace: dict[str, Any] | None = None
    rubric_scores: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    error: str | None = None


@dataclass
class EvaluationResult:
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


class Evaluation:
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
            raise NotImplementedError("Only endpoint environments are implemented in this cleanup pass")
        if workers != 1 and not env.supports_parallel_workers:
            raise ValueError("Endpoint environments must run with workers=1")

        grader = grader or AgenticGrader()
        runs = [_run_one(sample, env, grader, timeout, agent_config or {}) for sample in dataset]
        return EvaluationResult(
            runs=runs,
            metadata={"env": {"kind": env.kind, "agent_url": env.agent_url, "mcp_urls": env.mcp_urls}},
        )


def _run_one(sample: Any, env: Env, grader: AgenticGrader, timeout: float, agent_config: dict[str, Any]) -> EvaluationRun:
    started = time.monotonic()
    try:
        payload = {
            "model": env.config.get("model", "default"),
            "messages": [{"role": "user", "content": sample.query}],
            "agent_config": agent_config,
            "metadata": {"sample_id": sample.id},
        }
        if env.mcp_urls:
            payload["mcp_servers"] = env.mcp_servers()

        response = _post_json(env.agent_url or "", payload, timeout)
        final_response = _extract_final_response(response)
        trace = response.get("trace") if isinstance(response.get("trace"), dict) else None
        grade = grader.grade(sample, final_response, agent_trace=trace, mcp_urls=env.mcp_urls)
        latency_ms = (time.monotonic() - started) * 1000
        return EvaluationRun(
            sample_id=sample.id,
            query=sample.query,
            final_response=final_response,
            reward=grade["reward"],
            passed=grade["passed"],
            latency_ms=latency_ms,
            trace=trace,
            grader_trace=grade.get("grader_trace"),
            rubric_scores=grade["rubric_scores"],
            reasoning=grade["reasoning"],
        )
    except Exception as exc:
        latency_ms = (time.monotonic() - started) * 1000
        return EvaluationRun(
            sample_id=sample.id,
            query=sample.query,
            final_response="",
            reward=0.0,
            passed=False,
            latency_ms=latency_ms,
            error=str(exc),
        )


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
