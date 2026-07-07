"""Tool-trace-aware LLM grader for TensorEval.

Single-turn LLM-as-judge that reads the agent's tool trace and scores
against rubrics. No external dependencies beyond ``openai``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from tensoreval.enums import GraderType
from tensoreval.graders.agent_grader import (
    PASS_THRESHOLD,
    _extract_query,
    _extract_response,
    _normalize_rubrics,
    _parse_eval_result,
)
from tensoreval.graders.base import Grader

try:
    from openai import AsyncOpenAI

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


DEFAULT_MODEL = "mimo-v2.5-pro"
DEFAULT_BASE_URL = "https://token-plan-sgp.xiaomimimo.com/v1"


class VerificationGrader(Grader):
    """LLM-as-judge grader that reads tool traces.

    A single-turn chat/completions call that feeds the query, reference
    answer, agent response, rubrics, and tool trace to the model. The
    model returns structured JSON with per-rubric scores.

    Args:
        model: OpenAI-compatible model ID.
        api_key: Model API key.
        base_url: OpenAI-compatible base URL.
        timeout: Request timeout in seconds.
        fallback_on_error: If true, fall back to VercelAgentGrader on error.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        fallback_on_error: bool = True,
    ) -> None:
        super().__init__(GraderType.AGENT)
        self.model = model or os.environ.get("TENSOREVAL_MODEL_NAME") or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("TENSOREVAL_MODEL_API_KEY") or ""
        self.base_url = base_url or os.environ.get("TENSOREVAL_MODEL_BASE_URL") or DEFAULT_BASE_URL
        self.timeout = timeout
        self.fallback_on_error = fallback_on_error
        self.last_result: dict[str, Any] | None = None

    async def score(self, state: dict[str, Any], **kwargs: Any) -> float:
        """Score one response and return its weighted score."""
        if not _OPENAI_AVAILABLE:
            return await self._fallback(state, "openai not installed")

        completion = state.get("completion", [])
        answer = str(state.get("answer", ""))
        query_text = _extract_query(state)
        response = _extract_response(completion)

        if not completion:
            return 0.0

        raw_rubrics = state.get("info", {}).get("rubrics", [])
        rubrics = _normalize_rubrics(raw_rubrics, answer)

        info = state.get("info", {})
        tool_trace = info.get("tool_trace", [])
        mcp_tools = info.get("mcp_tools", []) or state.get("tools", [])

        try:
            result = await self._grade(
                query_text=query_text,
                answer=answer,
                response=response,
                rubrics=rubrics,
                tool_trace=tool_trace,
                mcp_tools=mcp_tools,
            )
            normalized = _normalize_eval_result(result, rubrics)
            self.last_result = normalized
            state["grader_result"] = normalized
            return float(normalized["weighted_score"])
        except Exception as exc:
            return await self._fallback(state, str(exc))

    async def _grade(
        self,
        query_text: str,
        answer: str,
        response: str,
        rubrics: list[dict[str, Any]],
        tool_trace: list[dict[str, Any]],
        mcp_tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Single-turn LLM call to grade the response."""
        client = AsyncOpenAI(
            api_key=self.api_key or "dummy",
            base_url=self.base_url,
            timeout=self.timeout,
        )

        system_prompt = _build_system_prompt(rubrics, mcp_tools)
        user_prompt = _build_user_prompt(query_text, answer, response, rubrics, tool_trace, mcp_tools)

        result = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        text = result.choices[0].message.content or ""
        return _parse_eval_result(text)

    async def _fallback(self, state: dict[str, Any], reason: str) -> float:
        """Fall back to the text-only VercelAgentGrader."""
        if not self.fallback_on_error:
            raise RuntimeError(f"VerificationGrader failed: {reason}")
        from tensoreval.graders.agent_grader import VercelAgentGrader

        fallback = VercelAgentGrader(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            fallback_on_error=False,
            timeout=self.timeout,
            use_web_search=False,
        )
        return await fallback.score(state)


# ---------------------------------------------------------------------------
# Prompts and result normalization
# ---------------------------------------------------------------------------

def _build_system_prompt(
    rubrics: list[dict[str, Any]],
    mcp_tools: list[dict[str, Any]],
) -> str:
    rubric_instructions = "\n\n".join(
        f"### {idx}. {rubric['name']} (weight: {rubric['weight']})\n"
        f"{rubric['criteria']}\n\n"
        "Score bands:\n"
        "- 1.0: Fully satisfies the rubric\n"
        "- 0.8-0.9: Mostly satisfies, minor issues only\n"
        "- 0.5-0.7: Partially satisfies, significant gaps\n"
        "- 0.2-0.4: Barely satisfies, major issues\n"
        "- 0.0-0.1: Does not satisfy at all"
        for idx, rubric in enumerate(rubrics, start=1)
    )

    mcp_section = ""
    if mcp_tools:
        mcp_section = "\n## Agent's Available Tools\n" + "\n".join(
            f"- {t.get('function', {}).get('name', 'tool')}: {t.get('function', {}).get('description', '')}"
            for t in mcp_tools
        ) + "\n"

    return f"""You are a rigorous AI agent evaluator. Your job is to VERIFY and score an agent's response against rubrics.
{mcp_section}
## Verification Strategy
1. Read the query, reference answer (if any), and agent response carefully
2. Review the tool trace — did the agent call the right tools with the right arguments?
3. For EACH rubric, compare your findings against the agent's response
4. Assign a score (0.0 to 1.0) per rubric
5. Do NOT compute weighted_score or pass/fail — the system computes these from your rubric scores

## Rubrics to Evaluate

{rubric_instructions}

## Special Rules
- **Empty/error responses**: All rubrics score 0
- **Numerical tolerance**: Accept +/-1% unless rubric specifies otherwise
- **Tool-call-only responses**: If the agent responded with tool calls but no text, evaluate the correctness of the tool calls
- **Truncated responses**: Evaluate what IS present, but penalize on completeness rubrics
- **Do NOT penalize for missing citations or source names** if the underlying claim is verifiable and correct
- **Do NOT penalize for response format** — bullet points, paragraphs, tables are all fine if the content is correct

## Output
After verification, output exactly ONE JSON object with no markdown fences:

{{
  "rubric_scores": [
    {{ "rubric_name": "...", "score": 0.85, "weight": 1.0, "reasoning": "What you verified and why this score" }}
  ],
  "grader_reasoning": "One paragraph summary of the overall evaluation including what you verified"
}}"""


def _build_user_prompt(
    query_text: str,
    answer: str,
    response: str,
    rubrics: list[dict[str, Any]],
    tool_trace: list[dict[str, Any]],
    mcp_tools: list[dict[str, Any]],
) -> str:
    reference = answer if answer else "No reference answer provided. Evaluate against rubrics."
    response_text = response if response else "[EMPTY - agent returned no response]"
    trace_text = json.dumps(tool_trace or [], indent=2, default=str)

    rubric_list = "\n".join(
        f"{idx}. {rubric['name']} (weight: {rubric['weight']}): {rubric['criteria']}"
        for idx, rubric in enumerate(rubrics, start=1)
    )

    return f"""## Query
{query_text}

## Reference Answer
{reference}

## Rubrics
{rubric_list}

## Agent Response
{response_text}

## Tool Trace
{trace_text}

Evaluate this response. Use the tool trace as evidence of actions the agent actually took. Then return the structured JSON result."""


def _normalize_eval_result(obj: dict[str, Any], rubrics: list[dict[str, Any]]) -> dict[str, Any]:
    raw_scores = obj.get("rubric_scores", [])
    rubric_scores: list[dict[str, Any]] = []
    if isinstance(raw_scores, list):
        for score in raw_scores:
            if not isinstance(score, dict):
                continue
            name = str(score.get("rubric_name") or score.get("name") or "")
            if not name:
                continue
            rubric_scores.append({
                "rubric_name": name,
                "score": _to_score(score.get("score")),
                "weight": _find_weight(rubrics, name),
                "reasoning": str(score.get("reasoning") or score.get("reason") or ""),
            })

    scored_names = {score["rubric_name"] for score in rubric_scores}
    for rubric in rubrics:
        if rubric["name"] not in scored_names:
            rubric_scores.append({
                "rubric_name": rubric["name"],
                "score": 0.0,
                "weight": rubric["weight"],
                "reasoning": "Not scored by grader",
            })

    weighted_score = sum(score["score"] * score["weight"] for score in rubric_scores)
    return {
        "rubric_scores": rubric_scores,
        "weighted_score": round(max(0.0, min(1.0, weighted_score)), 3),
        "passed": weighted_score >= PASS_THRESHOLD,
        "grader_reasoning": str(obj.get("grader_reasoning", "")),
    }


def _find_weight(rubrics: list[dict[str, Any]], rubric_name: str) -> float:
    for rubric in rubrics:
        if rubric["name"] == rubric_name:
            return float(rubric["weight"])
    return 1.0 / len(rubrics) if rubrics else 1.0


def _to_score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
