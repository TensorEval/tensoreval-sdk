"""RubricGrader — LLM-judged rubric scoring.

Uses an LLM to evaluate the agent's response against rubrics.
For simple substring matching, use `simple=True`.

Usage:
    # LLM-judged rubrics (default)
    grader = te.RubricGrader(model="gpt-4o", api_key="...", base_url="...")

    # Simple substring matching (no API call)
    grader = te.RubricGrader(simple=True)
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from tensoreval.graders.base import Grader
from tensoreval.types import GraderType


RUBRIC_JUDGE_PROMPT = """You are an evaluation judge. Score the agent's response against the criteria.

Query: {query}
Reference Answer: {reference}
Agent Response: {response}

Criteria: {criteria}

Score from 0.0 to 1.0:
- 1.0 = fully satisfies the criteria
- 0.5 = partially satisfies
- 0.0 = does not satisfy

Output ONLY a JSON object:
{{"score": <0.0-1.0>, "reason": "<brief explanation>"}}"""


class RubricGrader(Grader):
    """Rubric-based grader with LLM judging.

    When rubrics exist in the dataset, the LLM judges each one.
    When no rubrics exist, falls back to numeric/substring matching.

    Args:
        model: LLM model for judging (default: "mimo-v2.5-pro").
        api_key: API key for the LLM.
        base_url: Base URL for the LLM API.
        simple: If True, always use substring matching. Default: False.
        timeout: Per-judge-call timeout in seconds. Default: 30.0.
    """

    def __init__(
        self,
        model: str = "mimo-v2.5-pro",
        api_key: str | None = None,
        base_url: str | None = None,
        simple: bool = False,
        timeout: float = 30.0,
    ):
        super().__init__(GraderType.RUBRIC)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.simple = simple
        self.timeout = timeout

    async def score(self, state: dict[str, Any], **kwargs) -> float:
        """Score response using LLM judge against rubrics."""
        completion = state.get("completion", [])
        answer = state.get("answer", "")
        rubrics = state.get("info", {}).get("rubrics", [])
        query = state.get("query", "")

        if not query:
            prompt = state.get("prompt", [])
            if isinstance(prompt, list) and prompt:
                last = prompt[-1]
                query = last.get("content", "") if isinstance(last, dict) else str(last)

        if not completion:
            return 0.0

        last = completion[-1]
        response = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))

        # Simple mode or no rubrics: use substring/numeric matching
        if self.simple or not rubrics:
            return _simple_score(answer, response)

        # LLM judge mode: score each rubric
        try:
            total = 0.0
            for rubric in rubrics:
                criteria = rubric.get("criteria", rubric.get("rubric", ""))
                weight = rubric.get("weight", 1.0 / len(rubrics))
                if not criteria:
                    continue
                score = await self._judge_rubric(query, answer, response, criteria)
                total += score * weight
            return min(max(total, 0.0), 1.0)
        except Exception:
            return _simple_score(answer, response)

    async def _judge_rubric(self, query: str, reference: str, response: str, criteria: str) -> float:
        """Use LLM to judge a single rubric."""
        prompt = RUBRIC_JUDGE_PROMPT.format(
            query=query[:500],
            reference=reference[:500] if reference else "No reference",
            response=response[:1500],
            criteria=criteria,
        )

        is_anthropic = self.base_url and "anthropic" in self.base_url.lower()

        if is_anthropic:
            content = await self._call_anthropic(prompt)
        else:
            content = await self._call_openai(prompt)

        return _parse_score(content)

    async def _call_anthropic(self, prompt: str) -> str:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=self.api_key, base_url=self.base_url)
        response = await asyncio.wait_for(
            client.messages.create(
                model=self.model,
                max_tokens=200,
                system="You are an evaluation judge. Output only valid JSON.",
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=self.timeout,
        )
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    async def _call_openai(self, prompt: str) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key or "dummy", base_url=self.base_url or "https://api.openai.com/v1")
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an evaluation judge. Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
            ),
            timeout=self.timeout,
        )
        return response.choices[0].message.content or ""


def _simple_score(answer: str, response: str) -> float:
    """Simple substring + numeric matching."""
    if not answer or not response:
        return 0.0

    answer_nums = set(re.findall(r'-?\d+\.?\d*', answer))
    response_nums = set(re.findall(r'-?\d+\.?\d*', response))
    if answer_nums and answer_nums.issubset(response_nums):
        return 1.0

    return 1.0 if answer.lower().strip() in response.lower().strip() else 0.0


def _parse_score(content: str) -> float:
    """Parse a score from LLM output."""
    try:
        data = json.loads(content.strip())
        return float(data.get("score", 0.0))
    except (json.JSONDecodeError, AttributeError, ValueError):
        pass

    match = re.search(r'"?score"?\s*[:=]\s*([\d.]+)', content)
    if match:
        return min(float(match.group(1)), 1.0)
    return 0.0
