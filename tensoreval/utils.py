"""Utility functions for TensorEval.

Answer extraction helpers (ported from PrimeIntellect Verifiers, MIT License)
and HuggingFace dataset loading.
"""

from __future__ import annotations

from typing import Any, Callable


def extract_boxed_answer(text: str, strict: bool = False) -> str:
    """Extract the last ``\\boxed{...}`` answer from text.

    Args:
        text: The text to extract from.
        strict: If True, return ``""`` when no ``\\boxed{}`` is found.
    """
    boxed_start = text.rfind("\\boxed{")
    if boxed_start == -1:
        return "" if strict else text
    content_start = boxed_start + 7
    brace_count = 1
    closing_brace = content_start
    while closing_brace < len(text) and brace_count > 0:
        if text[closing_brace] == "{":
            brace_count += 1
        elif text[closing_brace] == "}":
            brace_count -= 1
        closing_brace += 1
    closing_brace = closing_brace - 1 if brace_count == 0 else -1
    if closing_brace == -1:
        return "" if strict else text
    return text[content_start:closing_brace]


def extract_hash_answer(text: str) -> str:
    """Extract answer after ``####`` separator (GSM8K format)."""
    if "####" not in text:
        return text
    return text.split("####")[1].strip()


def load_example_dataset(
    name: str = "gsm8k",
    split: str | None = None,
    n: int | None = None,
    seed: int = 0,
) -> Any:
    """Load a standard benchmark dataset from HuggingFace.

    Requires: ``pip install tensoreval[datasets]``
    """
    from datasets import load_dataset

    if name == "gsm8k":
        if split is None:
            split = "test"
        dataset = load_dataset("openai/gsm8k", "main")[split]
        preprocess = lambda x: {"question": x["question"], "answer": extract_hash_answer(x["answer"])}
    elif name == "math":
        if split is None:
            split = "train"
        dataset = load_dataset("chiayewken/competition_math")[split]
        preprocess = lambda x: {"question": x["problem"], "answer": extract_boxed_answer(x["solution"])}
    else:
        raise ValueError(f"Dataset {name} not supported. Supported: gsm8k, math")

    if n is not None and n > 0:
        dataset = dataset.shuffle(seed=seed).select(range(n))
    dataset = dataset.map(preprocess, num_proc=4, remove_columns=dataset.column_names)
    return dataset
