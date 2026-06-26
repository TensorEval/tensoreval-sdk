"""Parser that extracts content after </think> tags.

Ported from PrimeIntellect Verifiers (MIT License).
"""

from typing import Callable

from tensoreval.parsers.parser import Parser
from tensoreval.core.types import Messages


class ThinkParser(Parser):
    """Parser that strips everything before </think>."""

    def __init__(self, extract_fn: Callable[[str], str] = lambda x: x):
        super().__init__(extract_fn=extract_fn)

    def parse(self, text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        else:
            text = ""
        return self.extract_fn(text.strip())

    def get_format_reward_func(self) -> Callable:
        """Return a reward function that checks think-tag format."""

        def follows_format(text: str) -> float:
            if (
                text.strip().startswith("<think>")
                and text.count("<think>") == 1
                and text.count("</think>") == 1
                and len(text.split("</think>")[-1]) > 0
            ):
                return 1.0
            return 0.0

        def format_reward_func(completion: Messages, **kwargs) -> float:
            messages = self.get_assistant_messages(completion)
            if not messages:
                return 0.0
            return sum(
                follows_format(
                    self._content_to_text(
                        m.get("content", "")
                        if isinstance(m, dict)
                        else (m.content or "")
                    )
                )
                for m in messages
            ) / len(messages)

        return format_reward_func
