"""LLM-as-judge rubric for automated evaluation.

Ported from PrimeIntellect Verifiers (MIT License).
"""

from typing import Any

from tensoreval.rubrics.rubric import Rubric, _maybe_await
from tensoreval.core.types import Messages, State


DEFAULT_JUDGE_PROMPT = """Given a ground truth answer \
and a response, determine if the response is correct.

Question:
```
{question}
```

Ground truth answer:
```
{answer}
```

Response:
```
{response}
```

Respond either "yes" or "no" only."""


class JudgeRubric(Rubric):
    """Rubric that uses an LLM to judge correctness."""

    def __init__(
        self,
        parser: Any | None = None,
        judge_client: Any | None = None,
        judge_model: str = "mimo-v2.5-pro",
        judge_sampling_args: dict[str, Any] | None = None,
        judge_prompt: str = DEFAULT_JUDGE_PROMPT,
    ):
        super().__init__(parser=parser)
        self.judge_client = judge_client
        self.judge_model = judge_model
        self.judge_prompt = judge_prompt
        self.judge_sampling_args = judge_sampling_args or {}
        self.class_objects = {
            "parser": self.parser,
            "judge": self.judge,
            "judge_client": self.judge_client,
            "judge_model": self.judge_model,
            "judge_prompt": self.judge_prompt,
        }

    async def judge(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State | None = None,
    ) -> str:
        """Call the judge model to evaluate a response."""
        if isinstance(prompt, list):
            last_msg = prompt[-1]
            if isinstance(last_msg, dict) and "content" in last_msg:
                question = str(last_msg["content"])
            else:
                question = ""
        else:
            question = str(prompt)

        if self.parser:
            response = self.parser.parse_answer(completion)
        else:
            if isinstance(completion, list) and completion:
                last = completion[-1]
                response = last.get("content", "") if isinstance(last, dict) else str(last)
            else:
                response = str(completion)

        judge_prompt = self.judge_prompt.format(question=question, answer=answer, response=response)

        # Check cache
        cached = state.get("judge_response") if state else None
        if isinstance(cached, dict) and judge_prompt in cached:
            return cached[judge_prompt]

        # Call judge
        if self.judge_client is None:
            # Lazy import openai
            try:
                from openai import AsyncOpenAI
                self.judge_client = AsyncOpenAI()
            except ImportError:
                raise RuntimeError("openai package required for JudgeRubric. Install with: pip install openai")

        judge_args = dict(self.judge_sampling_args or {})
        if "max_tokens" in judge_args:
            judge_args["max_completion_tokens"] = judge_args.pop("max_tokens")

        try:
            judge_response = await _maybe_await(
                self.judge_client.chat.completions.create,
                model=self.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                **judge_args,
            )
            judge_response = str(judge_response.choices[0].message.content)
        except Exception as e:
            raise RuntimeError(f"Judge model error ({self.judge_model}): {e}") from e

        if state:
            if not isinstance(cached, dict):
                cached = {}
            cached[judge_prompt] = judge_response
            state["judge_response"] = cached
        return judge_response
