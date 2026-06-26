"""Parser that extracts content after </think> if present, passes through otherwise.

Ported from PrimeIntellect Verifiers (MIT License).
"""

from typing import Callable

from tensoreval.parsers.parser import Parser


class MaybeThinkParser(Parser):
    """Lenient think parser — strips </think> if present, passes through otherwise."""

    def __init__(self, extract_fn: Callable[[str], str] = lambda x: x):
        super().__init__(extract_fn=extract_fn)

    def parse(self, text: str) -> str:
        text = text.split("</think>")[-1].strip()
        return self.extract_fn(text)
