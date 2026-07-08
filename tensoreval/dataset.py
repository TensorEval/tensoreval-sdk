"""Dataset loading for TensorEval evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from tensoreval.types import Rubric, Sample


class Dataset:
    def __init__(self, samples: list[Sample], name: str = ""):
        if not samples:
            raise ValueError("Dataset must contain at least one sample")
        self.samples = samples
        self.name = name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    @classmethod
    def from_jsonl(cls, path: str | Path, name: str = "") -> "Dataset":
        path = Path(path)
        samples: list[Sample] = []
        with path.open(encoding="utf-8") as f:
            for index, line in enumerate(f):
                line = line.strip()
                if line:
                    samples.append(_row_to_sample(json.loads(line), index))
        return cls(samples, name=name or path.stem)

    @classmethod
    def from_dicts(cls, rows: list[dict[str, Any]], name: str = "") -> "Dataset":
        return cls([_row_to_sample(row, index) for index, row in enumerate(rows)], name=name)


def _row_to_sample(row: dict[str, Any], index: int) -> Sample:
    query = row.get("query") or row.get("input") or row.get("question")
    if not query:
        raise ValueError(f"Row {index} missing query")

    return Sample(
        id=str(row.get("id") or f"case_{index + 1}"),
        query=str(query),
        reference_answer=str(row.get("reference_answer") or row.get("target") or row.get("answer") or ""),
        rubrics=[Rubric.from_dict(r) for r in row.get("rubrics", [])],
        setup_script=str(row.get("setup_script") or ""),
        verify_script=str(row.get("verify_script") or ""),
        metadata=dict(row.get("metadata") or row.get("info") or {}),
    )
