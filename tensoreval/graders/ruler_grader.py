"""RulerGrader — Zero-config relative ranking via LLM.

Based on ART's RULER (Relative Universal LLM-Elicited Rewards).
No rubrics needed — the LLM judge ranks trajectories relatively.

Usage:
    grader = RulerGrader(model="gpt-4o-mini")
"""

import asyncio
import json
import re
from typing import Any
from tensoreval.graders.base import Grader
from tensoreval.enums import GraderType


DEFAULT_RUBRIC = """- A trajectory that achieves its goal should always get a significantly higher score than one that does not.
- More efficient achievement should score higher.
- Small differences should produce small score gaps; large differences should produce large gaps.
- Partial credit for progress toward the goal is acceptable."""


class RulerGrader(Grader):
    """Zero-config grader using relative ranking.

    RULER sends all trajectories in a group to an LLM judge,
    which ranks them relative to each other. No rubrics needed.

    Based on ART's RULER implementation (Apache 2.0 License).
    """

    def __init__(
        self,
        model: str = "mimo-v2.5-pro",
        api_key: str | None = None,
        base_url: str | None = None,
        rubric: str = DEFAULT_RUBRIC,
    ):
        super().__init__(GraderType.RULER)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.rubric = rubric

    async def score(self, state: dict, **kwargs) -> float:
        """Score a single response using heuristic fallback.

        When multiple samples are evaluated, _run_eval detects score_group
        and calls it for proper relative ranking instead of this method.
        """
        completion = state.get("completion", [])
        answer = state.get("answer", "")

        if not completion:
            return 0.0

        last = completion[-1]
        response = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))

        # Simple heuristic for single-sample scoring
        if not response:
            return 0.0

        score = 0.5
        # If there's a reference answer, check for match
        if answer and answer.lower().strip() in response.lower().strip():
            score = 0.8
        # Longer, more detailed responses get slightly higher scores
        word_count = len(response.split())
        if word_count > 50:
            score += 0.1
        return min(score, 1.0)

    async def score_group(self, states: list[dict], **kwargs) -> list[float]:
        """Score a group using relative ranking."""
        if len(states) <= 1:
            return [0.5 for _ in states]

        # Extract responses
        responses = []
        for state in states:
            completion = state.get("completion", [])
            if completion:
                last = completion[-1]
                content = last.get("content", "") if isinstance(last, dict) else str(getattr(last, "content", ""))
                responses.append(content)
            else:
                responses.append("")

        # Call RULER
        try:
            scores = await self._call_ruler(responses)
        except Exception:
            # Fallback: equal scores
            scores = [0.5 for _ in states]

        # Compute advantages
        mean_score = sum(scores) / len(scores) if scores else 0.0
        advantages = [s - mean_score for s in scores]
        for state, advantage in zip(states, advantages):
            state["advantage"] = advantage

        return scores

    async def _call_ruler(self, responses: list[str]) -> list[float]:
        """Call the RULER LLM judge."""
        # Build trajectory XML
        trajectories = []
        for i, resp in enumerate(responses):
            trajectories.append(f'<trajectory id="{i+1}">\n{resp[:1000]}\n</trajectory>')

        user_text = "Trajectories:\n\n" + "\n\n".join(trajectories)

        judge_prompt = f"""All of the trajectories below have been given the same goal. Your job is to consider each of them and give them a score between 0 and 1.

Grading standards:
{self.rubric}

Output ONLY a JSON array of scores, one per trajectory, in order:
[0.7, 0.3, 0.9]"""

        # Call LLM
        is_anthropic = self.base_url and "anthropic" in self.base_url.lower()

        if is_anthropic:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=self.api_key, base_url=self.base_url)
            response = await asyncio.wait_for(
                client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    system="You are a trajectory judge. Output only a JSON array of scores.",
                    messages=[{"role": "user", "content": user_text}],
                ),
                timeout=60.0,
            )
            content = response.content[0].text if response.content else "[]"
        else:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key or "dummy", base_url=self.base_url or "https://api.openai.com/v1")
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a trajectory judge. Output only a JSON array of scores."},
                        {"role": "user", "content": user_text},
                    ],
                    max_tokens=1000,
                ),
                timeout=60.0,
            )
            content = response.choices[0].message.content or "[]"

        # Parse scores — try JSON array first, then regex
        scores = []
        try:
            # Try to find JSON array
            match = re.search(r'\[[\d\s,\.]+\]', content)
            if match:
                scores = json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        if not scores:
            # Fallback: regex for individual scores
            score_pattern = re.findall(r'"?score"?\s*[:=]\s*(\d+\.?\d*)', content)
            scores = [float(s) for s in score_pattern[:len(responses)]]

        # Clamp and pad
        scores = [max(0.0, min(1.0, float(s))) for s in scores]
        while len(scores) < len(responses):
            scores.append(0.5)

        return scores[:len(responses)]
