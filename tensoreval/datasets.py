"""Datasets loader for TensorEval.

Supports loading from JSONL files, HuggingFace datasets, or Verifiers environments.
"""

import json
from pathlib import Path
from typing import Any

from tensoreval.core.sample import Sample


class Datasets:
    """Collection of evaluation samples.

    Usage:
        # From JSONL file
        datasets = Datasets.load_from_file("tasks.jsonl")

        # From HuggingFace dataset
        datasets = Datasets.from_huggingface("openai/gsm8k", split="test", n=100)

        # From list of dicts
        datasets = Datasets.from_dicts([{"query": "...", "rubrics": [...], "reference_answer": "..."}])
    """

    def __init__(self, samples: list[Sample], name: str = ""):
        self.samples = samples
        self.name = name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    @classmethod
    def load_from_file(cls, path: str | Path, name: str = "") -> "Datasets":
        """Load samples from a JSONL file.

        Each line should have:
        - query: str (the question/prompt)
        - rubrics: list[dict] (each with name, rubric, weight)
        - reference_answer: str (optional)
        - category: str (optional)
        - difficulty: str (optional)
        """
        path = Path(path)
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sample = Sample(
                    input=row.get("query", row.get("input", row.get("question", ""))),
                    target=row.get("reference_answer", row.get("target", row.get("answer", ""))),
                    id=row.get("id", row.get("query_id", f"q_{i+1}")),
                    metadata=row.get("metadata", row.get("context")),
                    rubrics=row.get("rubrics", []),
                    reference_answer=row.get("reference_answer"),
                    category=row.get("category", ""),
                    difficulty=row.get("difficulty", "medium"),
                    context=row.get("context"),
                )
                samples.append(sample)
        return cls(samples, name=name or path.stem)

    @classmethod
    def from_dicts(cls, rows: list[dict], name: str = "") -> "Datasets":
        """Create from a list of dicts."""
        samples = []
        for i, row in enumerate(rows):
            sample = Sample(
                input=row.get("query", row.get("input", row.get("question", ""))),
                target=row.get("reference_answer", row.get("target", row.get("answer", ""))),
                id=row.get("id", row.get("query_id", f"q_{i+1}")),
                metadata=row.get("metadata", row.get("context")),
                rubrics=row.get("rubrics", []),
                reference_answer=row.get("reference_answer"),
                category=row.get("category", ""),
                difficulty=row.get("difficulty", "medium"),
                context=row.get("context"),
            )
            samples.append(sample)
        return cls(samples, name=name)

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        split: str | None = None,
        n: int | None = None,
        seed: int = 0,
        name: str = "",
    ) -> "Datasets":
        """Load from a HuggingFace dataset (requires `datasets` package)."""
        from tensoreval.utils.data_utils import load_example_dataset
        dataset = load_example_dataset(dataset_name, split=split, n=n, seed=seed)
        samples = []
        for i, row in enumerate(dataset):
            sample = Sample(
                input=row.get("question", row.get("input", "")),
                target=row.get("answer", ""),
                id=f"q_{i+1}",
            )
            samples.append(sample)
        return cls(samples, name=name or dataset_name)

    def to_dicts(self) -> list[dict]:
        """Convert samples to list of dicts."""
        return [
            {
                "query": s.input,
                "rubrics": s.rubrics,
                "reference_answer": s.reference_answer or s.target,
                "category": s.category,
                "difficulty": s.difficulty,
                "id": s.id,
                "context": s.context,
            }
            for s in self.samples
        ]
