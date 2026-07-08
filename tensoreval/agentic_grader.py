"""Agentic grading primitives."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from tensoreval.tool_loop import ProviderConfig, ToolLoopAgent
from tensoreval.types import Sample


@dataclass
class AgenticGrader:
    """Grades agent responses against rubrics.

    When a provider (OpenAI, Anthropic, etc.) is configured, the grader uses
    a multi-turn tool loop with MCP access to verify claims.  Without a
    provider, it falls back to heuristic reference-answer matching.
    """
    model: str | None = None
    provider: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    instructions: str | None = None
    pass_threshold: float = 0.8
    timeout: float = 120.0
    max_steps: int = 10
    temperature: float = 0.0

    def grade(
        self,
        sample: Sample,
        final_response: str,
        agent_trace: dict[str, Any] | None = None,
        mcp_urls: list[str] | None = None,
    ) -> dict[str, Any]:
        if self._uses_provider():
            return self._grade_with_tool_loop(sample, final_response, agent_trace, mcp_urls or [])
        return self._grade_locally(sample, final_response)

    def _uses_provider(self) -> bool:
        return bool(
            self.provider
            or self.model
            or self.api_key
            or self.base_url
            or os.environ.get("TENSOREVAL_GRADER_API_KEY")
        )

    def _grade_with_tool_loop(
        self,
        sample: Sample,
        final_response: str,
        agent_trace: dict[str, Any] | None,
        mcp_urls: list[str],
    ) -> dict[str, Any]:
        provider = ProviderConfig.from_name(
            self.provider or "openai",
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
        )
        agent = ToolLoopAgent(
            provider=provider,
            instructions=self.instructions or _default_instructions(),
            max_steps=self.max_steps,
            timeout=self.timeout,
            temperature=self.temperature,
        )
        result = agent.run(_grader_prompt(sample, final_response, agent_trace), mcp_urls=mcp_urls)
        reward = round(max(0.0, min(1.0, float(result.get("reward", 0.0)))), 4)
        return {
            "reward": reward,
            "passed": bool(result.get("passed", reward >= self.pass_threshold)),
            "rubric_scores": result.get("rubric_scores", {}),
            "reasoning": result.get("reasoning_summary", ""),
            "grader_trace": result.get("grader_trace", {"steps": []}),
        }

    def _grade_locally(self, sample: Sample, final_response: str) -> dict[str, Any]:
        if sample.rubrics:
            score = _reference_score(sample.reference_answer, final_response)
            rubric_scores = {
                rubric.id: {
                    "score": score,
                    "reason": "Reference answer matched" if score == 1.0 else "Reference answer did not match",
                }
                for rubric in sample.rubrics
            }
            reward = sum(data["score"] * rubric.weight for rubric, data in zip(sample.rubrics, rubric_scores.values()))
            total_weight = sum(r.weight for r in sample.rubrics) or 1.0
            reward = reward / total_weight
        elif sample.reference_answer:
            reward = _reference_score(sample.reference_answer, final_response)
            rubric_scores = {"reference": {"score": reward, "reason": "Reference answer match"}}
        else:
            reward = 1.0 if final_response.strip() else 0.0
            rubric_scores = {"response": {"score": reward, "reason": "Non-empty final response"}}

        reward = round(max(0.0, min(1.0, reward)), 4)
        return {
            "reward": reward,
            "passed": reward >= self.pass_threshold,
            "rubric_scores": rubric_scores,
            "reasoning": "Heuristic local grading",
            "grader_trace": {"steps": [{"type": "heuristic_grade", "reward": reward}]},
        }


def _reference_score(reference: str, response: str) -> float:
    if not reference:
        return 1.0 if response.strip() else 0.0
    ref = _normalize(reference)
    res = _normalize(response)
    return 1.0 if ref and ref in res else 0.0


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _default_instructions() -> str:
    return (
        "You are an evaluation grader. Grade the agent response using the query, "
        "rubrics, reference answer, agent trace, and any MCP tools available. "
        "Return only JSON with keys: reward, passed, rubric_scores, reasoning_summary."
    )


def _grader_prompt(sample: Sample, final_response: str, agent_trace: dict[str, Any] | None) -> str:
    rubrics = [
        {"id": rubric.id, "description": rubric.description, "weight": rubric.weight}
        for rubric in sample.rubrics
    ]
    return (
        "Grade this agent run.\n\n"
        f"Query: {sample.query}\n"
        f"Reference answer: {sample.reference_answer or 'None'}\n"
        f"Rubrics: {rubrics}\n"
        f"Agent final response: {final_response}\n"
        f"Agent trace: {agent_trace or {}}\n\n"
        "Return JSON only. Example: "
        "{\"reward\":0.8,\"passed\":true,\"rubric_scores\":{\"rubric_id\":{\"score\":0.8,\"reason\":\"...\"}},"
        "\"reasoning_summary\":\"...\"}"
    )
