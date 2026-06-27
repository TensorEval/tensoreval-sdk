"""AgentGrader — LLM-as-judge that reads your rubrics.

The LLM reads each rubric, looks at the response, and scores it 0.0-1.0.
No keyword matching. No hacks. The LLM understands what "empathy" means.

Usage:
    grader = te.AgentGrader(
        model="mimo-v2.5-pro",
        api_key="tp-...",
        base_url="https://token-plan-sgp.xiaomimimo.com/anthropic",
    )

    # Then just run evaluation — rubrics come from each sample
    results = te.Evaluation.run(ds, grader, env=env)
"""

import json
import re
from typing import Any
from tensoreval.graders.base import Grader
from tensoreval.enums import GraderType


JUDGE_PROMPT = """You are an evaluation judge. Score the agent's response against each rubric.

Query: {query}
Reference Answer: {reference}
Agent Response: {response}

Rubrics to evaluate:
{rubrics_text}

For EACH rubric, assign a score from 0.0 to 1.0:
- 1.0 = fully satisfies the rubric
- 0.5 = partially satisfies
- 0.0 = does not satisfy

Output ONLY a JSON object:
{{"scores": [{{"name": "<rubric_name>", "score": <0.0-1.0>, "reason": "<brief>}}]}}"""


class AgentGrader(Grader):
    """LLM-as-judge grader.

    Reads rubrics from each sample and uses an LLM to score them.
    Works with OpenAI-compatible and Anthropic-compatible APIs.

    Args:
        model: Model name (e.g., "mimo-v2.5-pro").
        api_key: API key.
        base_url: API base URL.
    """

    def __init__(
        self,
        model: str = "mimo-v2.5-pro",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(GraderType.AGENT)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    async def score(self, state: dict, **kwargs) -> float:
        """Score response using LLM judge against rubrics."""
        completion = state.get("completion", [])
        answer = state.get("answer", "")
        rubrics = state.get("info", {}).get("rubrics", [])
        query = ""

        # Extract query
        prompt = state.get("prompt", [])
        if isinstance(prompt, list) and prompt:
            last = prompt[-1]
            query = last.get("content", "") if isinstance(last, dict) else str(last)

        if not completion:
            return 0.0

        last = completion[-1]
        response = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))

        # If no rubrics, do simple answer match
        if not rubrics:
            return 1.0 if answer and answer.lower() in response.lower() else 0.0

        # Build rubrics text for judge
        rubrics_text = "\n".join(
            f"{i+1}. {r.get('name', 'rubric')}: {r.get('criteria', r.get('rubric', ''))}"
            for i, r in enumerate(rubrics)
        )

        # Call LLM judge
        prompt_text = JUDGE_PROMPT.format(
            query=query[:500],
            reference=answer or "No reference answer",
            response=response[:2000],
            rubrics_text=rubrics_text,
        )

        try:
            scores = await self._call_judge(prompt_text)
            # Compute weighted score
            total = 0.0
            for rubric, score_data in zip(rubrics, scores):
                weight = rubric.get("weight", 1.0 / len(rubrics))
                total += score_data.get("score", 0.0) * weight
            return min(max(total, 0.0), 1.0)
        except Exception as e:
            # Fallback: simple answer match
            return 1.0 if answer and answer.lower() in response.lower() else 0.0

    async def _call_judge(self, prompt: str) -> list[dict]:
        """Call the LLM judge and parse scores."""
        is_anthropic = self.base_url and "anthropic" in self.base_url.lower()

        if is_anthropic:
            content = await self._call_anthropic(prompt)
        else:
            content = await self._call_openai(prompt)

        # Parse JSON from response
        return self._parse_scores(content)

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic-compatible API (Mimo uses this)."""
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=self.api_key, base_url=self.base_url)
        response = await client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        # Mimo returns ThinkingBlock + TextBlock — find the text block
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI-compatible API."""
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key or "dummy", base_url=self.base_url or "https://api.openai.com/v1")
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        return response.choices[0].message.content or ""

    def _parse_scores(self, content: str) -> list[dict]:
        """Parse LLM judge output into scores list."""
        # Extract JSON from response
        try:
            # Try to find JSON object
            match = re.search(r'\{[^{}]*"scores"[^{}]*\[.*?\].*?\}', content, re.DOTALL)
            if not match:
                # Try broader search
                match = re.search(r'\{.*"scores".*\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return data.get("scores", [])
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: extract individual scores
        scores = re.findall(r'"?score"?\s*[:=]\s*([\d.]+)', content)
        return [{"score": float(s)} for s in scores]
