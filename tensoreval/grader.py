"""Grader — the orchestrator for end-to-end evaluation.

Mirrors the engine's two-stage flow:
  Stage 1: call agent (captures full trajectory incl. tool_calls)
  Stage 2: judge each response per-rubric (LLM-as-judge with reasoning)

Then adds:
  - suggest_actionable_recommendations() — pattern analysis on scores
  - report("output.html") — visual HTML report with traces + scores
  - push() — push results to TensorEval dashboard

Usage:
    env = te.Env(image="local", agent_port=8000, mcp_port=8001)
    ds = te.Dataset.load_from_dict([...])

    grader = te.Grader(env=env, dataset=ds, workers=4)
    result = grader.run()

    print(result.summary())
    print(grader.suggest_actionable_recommendations())
    grader.report("report.html")
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import html
from dataclasses import dataclass, field
from typing import Any

from tensoreval.datasets import Datasets
from tensoreval.types import EvalConfig, Run, Score, Summary


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryStep:
    """One step in an agent's conversation trajectory."""
    role: str
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tool_call_id: str = ""


@dataclass
class GradedRun:
    """A single sample's full evaluation result with per-rubric breakdown."""
    sample_id: str
    query: str
    reference_answer: str
    agent_response: str
    trajectory: list[TrajectoryStep]
    reward: float
    passed: bool
    latency_ms: float
    rubric_scores: dict[str, dict] = field(default_factory=dict)
    grader_reasoning: str = ""
    error: str | None = None


@dataclass
class GradingResult:
    """Full result from Grader.run()."""
    runs: list[GradedRun]
    config: EvalConfig

    @property
    def avg_reward(self) -> float:
        if not self.runs:
            return 0.0
        return sum(r.reward for r in self.runs) / len(self.runs)

    @property
    def pass_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.passed) / len(self.runs)

    def summary(self) -> dict:
        """Aggregate summary with per-rubric averages."""
        # Per-rubric averages
        rubric_sums: dict[str, list[float]] = {}
        for run in self.runs:
            for name, data in run.rubric_scores.items():
                rubric_sums.setdefault(name, []).append(data.get("score", 0.0))

        per_rubric = {
            name: round(sum(scores) / len(scores), 3)
            for name, scores in rubric_sums.items()
        }

        return {
            "model": self.config.model,
            "num_runs": len(self.runs),
            "avg_reward": round(self.avg_reward, 3),
            "pass_rate": round(self.pass_rate, 3),
            "pass_count": sum(1 for r in self.runs if r.passed),
            "fail_count": sum(1 for r in self.runs if not r.passed),
            "per_rubric_averages": per_rubric,
        }


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class Grader:
    """Evaluation orchestrator — owns env, dataset, and grading flow.

    Args:
        env: te.Env with agent_port/mcp_port/image config.
            If image="local" or None, assumes agent is already running locally.
        dataset: Datasets to evaluate.
        workers: Concurrent workers (default 1 — rate-limited APIs need this).
        rubrics: Optional global rubrics (override per-sample).
        model/api_key/base_url: Judge model config. Falls back to env vars.
        pass_threshold: Score >= this is a pass (default 0.8).
    """

    def __init__(
        self,
        env: Any = None,
        dataset: Datasets | None = None,
        workers: int = 1,
        rubrics: list[dict] | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        pass_threshold: float = 0.8,
    ):
        self.env = env
        self.dataset = dataset
        self.workers = workers
        self.rubrics = rubrics
        self.pass_threshold = pass_threshold

        # Judge model config from args or env
        self.model = model or os.environ.get("TENSOREVAL_MODEL_NAME", "mimo-v2.5-pro")
        self.api_key = api_key or os.environ.get("TENSOREVAL_MODEL_API_KEY", "")
        self.base_url = base_url or os.environ.get("TENSOREVAL_MODEL_BASE_URL", "https://api.openai.com/v1")

        self._result: GradingResult | None = None

    # ------------------------------------------------------------------
    # run() — the main entry point
    # ------------------------------------------------------------------

    def run(self) -> GradingResult:
        """Run the full evaluation pipeline. Returns GradingResult."""
        if self.dataset is None:
            raise ValueError("No dataset provided")

        config = EvalConfig(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            workers=self.workers,
            pass_threshold=self.pass_threshold,
        )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self._run_async(config)
        if loop is not None:
            import nest_asyncio
            nest_asyncio.apply()
            result = loop.run_until_complete(coro)
        else:
            result = asyncio.run(coro)

        self._result = result
        return result

    async def _run_async(self, config: EvalConfig) -> GradingResult:
        """Internal: async evaluation pipeline."""
        sem = asyncio.Semaphore(self.workers)

        # Resolve agent endpoint
        agent_url, mcp_url = self._resolve_endpoints()

        # Build MCP registry if mcp_url is set
        mcp_registry = None
        mcp_tools: list[dict] = []
        if mcp_url:
            from tensoreval.tools.mcp import MCPServer, MCPToolRegistry
            mcp_registry = MCPToolRegistry()
            mcp_registry.add_server("default", MCPServer(url=mcp_url))
            try:
                await mcp_registry.list_all_tools()
                mcp_tools = mcp_registry.to_openai_tools()
            except Exception:
                pass

        # Evaluate each sample
        async def grade_one(idx: int) -> GradedRun:
            async with sem:
                return await self._grade_single(idx, agent_url, mcp_tools, mcp_registry, config)

        tasks = [grade_one(i) for i in range(len(self.dataset))]
        graded_runs = await asyncio.gather(*tasks)

        return GradingResult(runs=list(graded_runs), config=config)

    # ------------------------------------------------------------------
    # Stage 1 + 2: call agent + grade response
    # ------------------------------------------------------------------

    async def _grade_single(
        self,
        idx: int,
        agent_url: str,
        mcp_tools: list[dict],
        mcp_registry: Any,
        config: EvalConfig,
    ) -> GradedRun:
        """Call agent, capture trajectory, then judge."""
        sample = self.dataset[idx]
        start_time = time.monotonic()

        # Determine rubrics for this sample
        rubrics = self.rubrics or sample.rubrics
        if not rubrics:
            await asyncio.sleep(2.0)  # rate-limit buffer
            rubrics = await self._auto_generate_rubrics(sample.input, config)

        # Stage 1: call the agent endpoint
        trajectory: list[TrajectoryStep] = []
        agent_response = ""
        error = None

        try:
            trajectory, agent_response = await self._call_agent(
                agent_url, sample.input, config, mcp_tools
            )
        except Exception as e:
            error = str(e)

        latency_ms = (time.monotonic() - start_time) * 1000

        # Stage 2: judge the response (delay to avoid rate-limit)
        await asyncio.sleep(2.0)
        rubric_scores: dict[str, dict] = {}
        grader_reasoning = ""
        reward = 0.0

        if not error and agent_response:
            try:
                rubric_scores, grader_reasoning, reward = await self._judge(
                    sample.input, agent_response, sample.target, rubrics, trajectory, config
                )
            except Exception as e:
                error = f"Judge error: {e}"
                reward = 0.0
        elif error:
            reward = 0.0

        return GradedRun(
            sample_id=sample.id,
            query=sample.input,
            reference_answer=sample.target,
            agent_response=agent_response,
            trajectory=trajectory,
            reward=reward,
            passed=reward >= self.pass_threshold,
            latency_ms=latency_ms,
            rubric_scores=rubric_scores,
            grader_reasoning=grader_reasoning,
            error=error,
        )

    # ------------------------------------------------------------------
    # Agent calling — OpenAI /chat/completions with trajectory capture
    # ------------------------------------------------------------------

    async def _call_agent(
        self,
        agent_url: str,
        query: str,
        config: EvalConfig,
        mcp_tools: list[dict],
    ) -> tuple[list[TrajectoryStep], str]:
        """Call agent endpoint, return (trajectory, final_response).

        Handles single-turn and multi-turn tool-calling.
        """
        import httpx

        messages: list[dict] = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        messages.append({"role": "user", "content": query})

        trajectory: list[TrajectoryStep] = [
            TrajectoryStep(role="user", content=query)
        ]

        body: dict = {"model": config.model, "messages": messages}
        if mcp_tools:
            body["tools"] = mcp_tools

        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        final_response = ""

        for turn in range(10):  # max 10 tool-calling rounds
            # Rate-limit retry
            resp_data = None
            for attempt in range(4):
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        agent_url,
                        json=body,
                        headers=headers,
                        timeout=120.0,
                    )
                if resp.status_code == 429 and attempt < 3:
                    await asyncio.sleep(10.0 * (attempt + 1))
                    continue
                resp_data = resp.json()
                break

            if resp_data is None:
                break
            data = resp_data

            # Extract from OpenAI /chat/completions format
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls", [])
            finish_reason = choice.get("finish_reason", "stop")

            # Record this step in trajectory
            step = TrajectoryStep(role="assistant", content=content)
            if tool_calls:
                step.tool_calls = tool_calls
            trajectory.append(step)

            if content:
                final_response = content

            # No tool calls → done
            if not tool_calls or finish_reason == "stop":
                break

            # Execute tool calls via MCP
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    args = {}

                tool_result = {"error": "no MCP registry"}
                if mcp_registry:
                    try:
                        tool_result = await mcp_registry.call_tool_by_name(tool_name, args)
                    except Exception as e:
                        tool_result = {"error": str(e)}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": json.dumps(tool_result, default=str),
                })
                trajectory.append(TrajectoryStep(
                    role="tool", content=json.dumps(tool_result, default=str)[:500],
                    tool_call_id=tc.get("id", ""),
                ))

            body["messages"] = messages

        return trajectory, final_response

    # ------------------------------------------------------------------
    # Judging — LLM-as-judge with per-rubric scores + reasoning
    # ------------------------------------------------------------------

    async def _judge(
        self,
        query: str,
        response: str,
        reference: str,
        rubrics: list,
        trajectory: list[TrajectoryStep],
        config: EvalConfig,
    ) -> tuple[dict[str, dict], str, float]:
        """Judge response against rubrics. Returns (scores, reasoning, reward)."""
        from tensoreval.graders.agent_grader import AgentGrader

        grader = AgentGrader(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            fallback_on_error=True,
        )

        # Build rubric list for the judge
        rubric_dicts = []
        for r in rubrics:
            if hasattr(r, "name"):
                rubric_dicts.append({
                    "name": r.name,
                    "criteria": r.criteria,
                    "weight": r.weight,
                })
            elif isinstance(r, dict):
                rubric_dicts.append({
                    "name": r.get("name", r.get("id", "rubric")),
                    "criteria": r.get("criteria", r.get("description", "")),
                    "weight": r.get("weight", 1.0 / len(rubrics)),
                })

        # Build trajectory display for the judge
        traj_lines = []
        for step in trajectory:
            if step.role == "user":
                traj_lines.append(f"User: {step.content[:200]}")
            elif step.role == "assistant":
                if step.content:
                    traj_lines.append(f"Assistant: {step.content[:200]}")
                if step.tool_calls:
                    for tc in step.tool_calls:
                        fn = tc.get("function", {})
                        traj_lines.append(f"  → Tool call: {fn.get('name', '?')}({fn.get('arguments', '')[:100]})")
            elif step.role == "tool":
                traj_lines.append(f"Tool result: {step.content[:150]}")

        trajectory_text = "\n".join(traj_lines) if traj_lines else response

        # Build judge prompt
        rubrics_text = "\n".join(
            f"{i+1}. {r['name']}: {r['criteria']} (weight: {r['weight']})"
            for i, r in enumerate(rubric_dicts)
        )

        judge_prompt = f"""Evaluate this agent response against the rubrics.

Query: {query[:500]}
Reference Answer: {reference or "No reference answer"}

Agent Trajectory (messages + tool calls):
{trajectory_text[:2000]}

Rubrics:
{rubrics_text}

For EACH rubric, assign 0.0-1.0 and write reasoning.
Output ONLY JSON:
{{"scores": [{{"name": "<name>", "score": <0.0-1.0>, "reason": "<why>"}}], "overall_reasoning": "<summary>"}}"""

        # Call the judge
        try:
            content = await grader._call_openai(judge_prompt)
        except Exception:
            content = ""

        scores_list = grader._parse_scores(content)

        # Build per-rubric scores dict
        rubric_scores: dict[str, dict] = {}
        total = 0.0
        for rubric, score_data in zip(rubric_dicts, scores_list):
            name = rubric["name"]
            score = score_data.get("score", 0.0)
            reason = score_data.get("reason", "")
            rubric_scores[name] = {"score": score, "reason": reason}
            total += score * rubric["weight"]

        # Extract overall reasoning
        reasoning = ""
        try:
            match_json = json.loads(content.strip())
            reasoning = match_json.get("overall_reasoning", "")
        except Exception:
            pass

        return rubric_scores, reasoning, min(max(total, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Auto-rubric generation
    # ------------------------------------------------------------------

    async def _auto_generate_rubrics(self, query: str, config: EvalConfig) -> list[dict]:
        """Generate rubrics from query using LLM."""
        from tensoreval.graders.agent_grader import AgentGrader

        grader = AgentGrader(
            model=self.model, api_key=self.api_key, base_url=self.base_url,
            fallback_on_error=True,
        )
        try:
            return await grader._auto_generate_rubrics(query)
        except Exception:
            return [
                {"name": "correctness", "criteria": "Response is factually correct", "weight": 0.4},
                {"name": "completeness", "criteria": "Response addresses the full question", "weight": 0.3},
                {"name": "clarity", "criteria": "Response is clear and well-formed", "weight": 0.3},
            ]

    # ------------------------------------------------------------------
    # Endpoint resolution
    # ------------------------------------------------------------------

    def _resolve_endpoints(self) -> tuple[str, str | None]:
        """Resolve agent + MCP URLs from env config.

        If image='local' or None, assumes agent is running locally on agent_port.
        """
        agent_url = None
        mcp_url = None

        if self.env is not None:
            # Direct URLs take priority
            agent_url = getattr(self.env, "agent_url", None)
            mcp_url = getattr(self.env, "mcp_url", None)

            # Extract from ports if no direct URL
            if not agent_url:
                port = getattr(self.env, "agent_port", None) or getattr(self.env, "_agent_port", None)
                if port:
                    agent_url = f"http://localhost:{port}/chat/completions"

            if not mcp_url:
                mcp_port = getattr(self.env, "mcp_port", None) or getattr(self.env, "_mcp_port", None)
                if mcp_port:
                    mcp_url = f"http://localhost:{mcp_port}/mcp"

        # Fallback to default
        if not agent_url:
            agent_url = "http://localhost:8000/chat/completions"

        return agent_url, mcp_url

    # ------------------------------------------------------------------
    # Actionable recommendations
    # ------------------------------------------------------------------

    def suggest_actionable_recommendations(self) -> list[str]:
        """Analyze score patterns and suggest what to fix."""
        if self._result is None:
            return ["Run grader.run() first"]

        runs = self._result.runs
        if not runs:
            return ["No runs to analyze"]

        recommendations: list[str] = []

        # 1. Per-rubric weakness detection
        rubric_scores: dict[str, list[float]] = {}
        for run in runs:
            for name, data in run.rubric_scores.items():
                rubric_scores.setdefault(name, []).append(data.get("score", 0.0))

        for name, scores in sorted(rubric_scores.items(), key=lambda x: sum(x[1]) / len(x[1])):
            avg = sum(scores) / len(scores)
            if avg < 0.5:
                count_low = sum(1 for s in scores if s < 0.4)
                recommendations.append(
                    f"FIX: Rubric '{name}' averages {avg:.2f} ({count_low}/{len(scores)} samples scored <0.4). "
                    f"This is your weakest dimension — prioritize prompt/agent changes here."
                )
            elif avg < 0.8:
                recommendations.append(
                    f"IMPROVE: Rubric '{name}' averages {avg:.2f} — decent but has room for improvement."
                )

        # 2. Error pattern detection
        error_runs = [r for r in runs if r.error]
        if error_runs:
            pct = len(error_runs) / len(runs) * 100
            recommendations.append(
                f"CRITICAL: {len(error_runs)}/{len(runs)} ({pct:.0f}%) runs had errors. "
                f"First error: {error_runs[0].error[:100] if error_runs[0].error else 'unknown'}"
            )

        # 3. Empty response detection
        empty_runs = [r for r in runs if not r.agent_response and not r.error]
        if empty_runs:
            recommendations.append(
                f"WARNING: {len(empty_runs)} runs returned empty responses — "
                f"check agent endpoint connectivity."
            )

        # 4. Tool-call contradiction detection
        contradiction_count = 0
        for run in runs:
            has_tool = any(s.tool_calls for s in run.trajectory)
            if has_tool and run.reward < 0.5:
                contradiction_count += 1
        if contradiction_count > 0:
            recommendations.append(
                f"PATTERN: {contradiction_count} runs made tool calls but scored low — "
                f"agent may be calling tools correctly but ignoring/contradicting the results. "
                f"Check answer-generation logic."
            )

        # 5. Overall assessment
        if self._result.avg_reward >= 0.9:
            recommendations.append("OVERALL: Agent is performing well across all rubrics.")
        elif self._result.avg_reward >= 0.7:
            recommendations.append("OVERALL: Agent is mostly working — focus on the weakest rubrics above.")
        else:
            recommendations.append("OVERALL: Agent needs significant improvement — start with the lowest-scoring rubric.")

        return recommendations

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def report(self, path: str = "report.html") -> str:
        """Generate a visual HTML report with traces, scores, and recommendations."""
        if self._result is None:
            raise ValueError("Run grader.run() first")

        summary = self._result.summary()
        recommendations = self.suggest_actionable_recommendations()

        # Build HTML
        rows_html = []
        for run in self._result.runs:
            status_color = "#22c55e" if run.passed else "#ef4444"
            status_text = "PASS" if run.passed else "FAIL"

            # Trajectory
            traj_html = ""
            for step in run.trajectory:
                role_class = f"traj-{step.role}"
                content = html.escape(step.content[:200])
                traj_html += f'<div class="{role_class}"><b>{step.role}</b>: {content}</div>'
                if step.tool_calls:
                    for tc in step.tool_calls:
                        fn = tc.get("function", {})
                        traj_html += f'<div class="traj-tool">→ {html.escape(fn.get("name", "?"))}({html.escape(fn.get("arguments", "")[:80])})</div>'

            # Rubric scores
            rubric_html = ""
            for name, data in run.rubric_scores.items():
                score = data.get("score", 0.0)
                reason = data.get("reason", "")
                bar_color = "#22c55e" if score >= 0.8 else "#f59e0b" if score >= 0.5 else "#ef4444"
                rubric_html += f"""
                <div class="rubric-row">
                    <span class="rubric-name">{html.escape(name)}</span>
                    <div class="rubric-bar"><div class="rubric-fill" style="width:{score*100:.0f}%;background:{bar_color}"></div></div>
                    <span class="rubric-score">{score:.2f}</span>
                    <span class="rubric-reason">{html.escape(reason[:100])}</span>
                </div>"""

            rows_html.append(f"""
            <div class="run-card {'passed' if run.passed else 'failed'}">
                <div class="run-header">
                    <span class="run-status" style="color:{status_color}">{status_text}</span>
                    <span class="run-reward">reward={run.reward:.2f}</span>
                    <span class="run-latency">{run.latency_ms:.0f}ms</span>
                </div>
                <div class="run-query">{html.escape(run.query[:150])}</div>
                <div class="run-response"><b>Agent:</b> {html.escape(run.agent_response[:300])}</div>
                {f'<div class="run-error">{html.escape(run.error)}</div>' if run.error else ''}
                <div class="run-traj">{traj_html}</div>
                <div class="run-rubrics">{rubric_html}</div>
                {f'<div class="run-grader"><b>Judge:</b> {html.escape(run.grader_reasoning[:300])}</div>' if run.grader_reasoning else ''}
            </div>""")

        recs_html = "".join(f"<li>{html.escape(r)}</li>" for r in recommendations)

        full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>TensorEval Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; background: #f8fafc; }}
h1 {{ color: #1e293b; }}
.summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }}
.summary-card {{ background: white; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0; text-align: center; }}
.summary-card .val {{ font-size: 24px; font-weight: bold; }}
.summary-card .label {{ font-size: 12px; color: #64748b; text-transform: uppercase; }}
.run-card {{ background: white; border-radius: 8px; padding: 16px; margin: 12px 0; border-left: 4px solid #cbd5e1; }}
.run-card.passed {{ border-left-color: #22c55e; }}
.run-card.failed {{ border-left-color: #ef4444; }}
.run-header {{ display: flex; gap: 16px; align-items: center; margin-bottom: 8px; }}
.run-status {{ font-weight: bold; font-size: 14px; }}
.run-reward {{ font-family: monospace; font-size: 13px; color: #475569; }}
.run-query {{ font-size: 14px; color: #334155; margin-bottom: 6px; font-style: italic; }}
.run-response {{ font-size: 13px; color: #1e293b; margin-bottom: 6px; }}
.run-error {{ font-size: 13px; color: #dc2626; margin-bottom: 6px; }}
.run-traj {{ background: #f1f5f9; border-radius: 6px; padding: 8px; margin: 8px 0; font-size: 12px; }}
.traj-user {{ color: #2563eb; }} .traj-assistant {{ color: #16a34a; }} .traj-tool {{ color: #ea580c; padding-left: 20px; font-family: monospace; }}
.rubric-row {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 12px; }}
.rubric-name {{ width: 120px; font-weight: 500; }}
.rubric-bar {{ flex: 1; height: 8px; background: #e2e8f0; border-radius: 4px; }}
.rubric-fill {{ height: 100%; border-radius: 4px; }}
.rubric-score {{ width: 40px; font-family: monospace; }}
.rubric-reason {{ color: #64748b; flex: 2; }}
.run-grader {{ font-size: 12px; color: #475569; margin-top: 8px; border-top: 1px solid #e2e8f0; padding-top: 8px; }}
.recommendations {{ background: #fef3c7; border-radius: 8px; padding: 16px; margin: 20px 0; }}
.recommendations li {{ margin: 6px 0; }}
</style></head><body>
<h1>TensorEval Report</h1>
<div class="summary">
    <div class="summary-card"><div class="val">{summary['avg_reward']:.2f}</div><div class="label">Avg Reward</div></div>
    <div class="summary-card"><div class="val">{summary['pass_rate']:.0%}</div><div class="label">Pass Rate</div></div>
    <div class="summary-card"><div class="val">{summary['pass_count']}/{summary['num_runs']}</div><div class="label">Pass/Total</div></div>
    <div class="summary-card"><div class="val">{summary.get('per_rubric_averages', {}) and min(summary['per_rubric_averages'].values()) or 0:.2f}</div><div class="label">Weakest Rubric</div></div>
</div>
<div class="recommendations"><h3>Actionable Recommendations</h3><ul>{recs_html}</ul></div>
{''.join(rows_html)}
</body></html>"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(full_html)

        return path
