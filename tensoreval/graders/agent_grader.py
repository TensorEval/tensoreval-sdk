"""AgentGrader — LLM-as-judge that reads your rubrics.

The LLM reads each rubric, looks at the response, and scores it 0.0-1.0.
No keyword matching. No hacks. The LLM understands what "empathy" means.

If a sample has NO rubrics defined, the grader auto-generates appropriate
rubrics from the query using the LLM (auto-rubric generation).

Usage:
    # Option 1: explicit rubrics in dataset
    grader = te.AgentGrader(model="mimo-v2.5-pro", api_key="...", base_url="...")

    # Option 2: no rubrics → auto-generated from query
    ds = te.Datasets.load_from_dict([{"query": "What is 2+2?", "reference_answer": "4"}])
    results = te.Evaluation.run(ds, grader, agent=my_agent)

    # Option 3: env-based config (no manual grader setup)
    #   TENSOREVAL_MODEL_API_KEY=...
    #   TENSOREVAL_MODEL_BASE_URL=...
    #   TENSOREVAL_MODEL_NAME=mimo-v2.5-pro
    results = te.Evaluation.run(ds, agent=my_agent)  # grader auto-created from env
"""

import asyncio
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
{{"scores": [{{"name": "<rubric_name>", "score": <0.0-1.0>, "reason": "<brief>"}}]}}"""


AUTO_RUBRIC_PROMPT = """Generate evaluation rubrics for this query. The agent will respond to this query and needs to be graded.

Query: {query}

Generate 3-4 rubrics that a judge would use to evaluate ANY response to this query. Think about what makes a good response here — accuracy, tone, completeness, format, etc.

Output ONLY a JSON array:
[{{"name": "<short_name>", "criteria": "<what makes a good response for this rubric>", "weight": <0.0-1.0>}}]

Weights must sum to 1.0."""


class AgentGrader(Grader):
    """LLM-as-judge grader.

    Reads rubrics from each sample and uses an LLM to score them.
    Works with OpenAI-compatible and Anthropic-compatible APIs.

    Args:
        model: Model name (e.g., "mimo-v2.5-pro").
        api_key: API key.
        base_url: API base URL.
        fallback_on_error: If True, falls back to substring matching on errors.
            If False, raises the error. Default: False (fail loudly).
    """

    def __init__(
        self,
        model: str = "mimo-v2.5-pro",
        api_key: str | None = None,
        base_url: str | None = None,
        fallback_on_error: bool = False,
    ):
        super().__init__(GraderType.AGENT)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.fallback_on_error = fallback_on_error
        self._rubric_cache: dict[str, list[dict]] = {}

    async def score(self, state: dict, **kwargs) -> float:
        """Score response using LLM judge against rubrics.

        If no rubrics are defined for this sample, auto-generates them
        from the query using the LLM.
        """
        completion = state.get("completion", [])
        answer = state.get("answer", "")
        rubrics = state.get("info", {}).get("rubrics", [])
        query = state.get("query", "")

        # Extract query from prompt if not in state
        if not query:
            prompt = state.get("prompt", [])
            if isinstance(prompt, list) and prompt:
                last = prompt[-1]
                query = last.get("content", "") if isinstance(last, dict) else str(last)

        if not completion:
            return 0.0

        last = completion[-1]
        response = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))

        # ── Auto-generate rubrics if none defined ───────────────────
        if not rubrics:
            try:
                rubrics = await self._auto_generate_rubrics(query)
            except Exception:
                pass

        # If still no rubrics, do simple answer match
        if not rubrics:
            return 1.0 if answer and answer.lower() in response.lower() else 0.0

        # Build rubrics text for judge
        rubrics_text = "\n".join(
            f"{i+1}. {r.get('name', 'rubric')}: {r.get('criteria', r.get('rubric', ''))} (weight: {r.get('weight', 1.0/len(rubrics))})"
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
            if self.fallback_on_error:
                return 1.0 if answer and answer.lower() in response.lower() else 0.0
            raise RuntimeError(f"AgentGrader judge call failed: {e}") from e

    async def _auto_generate_rubrics(self, query: str) -> list[dict]:
        """Generate rubrics from the query using the LLM. Cached per query."""
        cache_key = query[:200]
        if cache_key in self._rubric_cache:
            return self._rubric_cache[cache_key]

        prompt = AUTO_RUBRIC_PROMPT.format(query=query[:500])

        is_anthropic = self.base_url and "anthropic" in self.base_url.lower()
        if is_anthropic:
            content = await self._call_anthropic(prompt)
        else:
            content = await self._call_openai(prompt)

        rubrics = self._parse_rubrics(content)
        if rubrics:
            self._rubric_cache[cache_key] = rubrics
        return rubrics

    def _parse_rubrics(self, content: str) -> list[dict]:
        """Parse LLM output into a rubrics list."""
        try:
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                rubrics = json.loads(match.group())
                # Validate structure
                valid = []
                for r in rubrics:
                    if isinstance(r, dict) and "name" in r and "criteria" in r:
                        valid.append({
                            "name": r["name"],
                            "criteria": r["criteria"],
                            "weight": float(r.get("weight", 1.0 / len(rubrics))),
                        })
                return valid if valid else []
        except (json.JSONDecodeError, TypeError):
            pass
        return []

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
        """Call Anthropic-compatible API with correct system message handling."""
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=self.api_key, base_url=self.base_url)
        response = await asyncio.wait_for(
            client.messages.create(
                model=self.model,
                max_tokens=2000,
                system="You are an evaluation judge. Output only valid JSON.",
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=60.0,
        )
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI-compatible API with rate-limit retry."""
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key or "dummy", base_url=self.base_url or "https://api.openai.com/v1")

        async def _call():
            return await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an evaluation judge. Output only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,
                ),
                timeout=60.0,
            )

        for attempt in range(4):
            try:
                response = await _call()
                return response.choices[0].message.content or ""
            except Exception as e:
                is_rate_limit = getattr(e, "status_code", 0) == 429 or "429" in str(e) or "rate" in str(e).lower()
                if not is_rate_limit or attempt == 3:
                    raise
                await asyncio.sleep(3.0 * (attempt + 1))
        return ""

    def _parse_scores(self, content: str) -> list[dict]:
        """Parse LLM judge output into scores list."""
        # Try direct JSON parse first
        try:
            data = json.loads(content.strip())
            if "scores" in data:
                return data["scores"]
        except (json.JSONDecodeError, AttributeError):
            pass

        # Extract JSON from response
        try:
            match = re.search(r'\{[^{}]*"scores"[^{}]*\[.*?\].*?\}', content, re.DOTALL)
            if not match:
                match = re.search(r'\{.*"scores".*\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return data.get("scores", [])
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: extract individual scores
        scores = re.findall(r'"?score"?\s*[:=]\s*([\d.]+)', content)
        return [{"score": float(s)} for s in scores]
