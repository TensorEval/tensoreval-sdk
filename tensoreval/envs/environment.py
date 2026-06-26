"""Base Environment class for TensorEval.

Architecture based on PrimeIntellect Verifiers (MIT License) and HUD (MIT License).
Provides the core evaluation pipeline: dataset → rollout → score → results.

Key patterns adopted:
- Verifiers: lazy dataset builders, State with input forwarding, Rubric scoring
- HUD: @env.template() async generator, capability system
- Inspect: Sample/Score data models
- ART: RULER zero-config reward, messages_and_choices trajectory format
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from tensoreval.core.decorators import discover_decorated
from tensoreval.core.types import (
    AssistantMessage,
    GenerateMetadata,
    GenerateOutputs,
    LogCallback,
    Message,
    Messages,
    ProgressCallback,
    Response,
    RolloutInput,
    RolloutOutput,
    RolloutScore,
    RolloutTiming,
    SamplingArgs,
    StartCallback,
    State,
    SystemMessage,
    TimeSpan,
    Tool,
    TrajectoryStep,
    UserMessage,
    flatten_task_input,
    state_to_output,
)
from tensoreval.parsers.parser import Parser
from tensoreval.rubrics.rubric import Rubric


# ---------------------------------------------------------------------------
# Dataset builder type (lazy callable)
# ---------------------------------------------------------------------------

DatasetBuilder = Callable[[], Any]


# ---------------------------------------------------------------------------
# Environment base class
# ---------------------------------------------------------------------------

class Environment(ABC):
    """Base class for all environments.

    An environment defines:
    - A dataset of tasks (raw Dataset or lazy DatasetBuilder)
    - A rubric for scoring
    - A rollout method (how the agent interacts)

    Key design (from Verifiers):
    - dataset can be a raw Dataset (eagerly built) or a DatasetBuilder (lazy callable)
    - eval_dataset falls back to dataset if not provided
    - _ensure_prompt creates prompt column from question + system_prompt + few_shot
    - State has input forwarding for ergonomic access
    """

    def __init__(
        self,
        dataset: Any | DatasetBuilder | None = None,
        eval_dataset: Any | DatasetBuilder | None = None,
        system_prompt: str | None = None,
        few_shot: Messages | None = None,
        parser: Parser | None = None,
        rubric: Rubric | None = None,
        sampling_args: SamplingArgs | None = None,
        tool_defs: list[Tool] | None = None,
        max_workers: int = 64,
        env_id: str | None = None,
        score_rollouts: bool = True,
        pass_threshold: float = 0.5,
        max_turns: int = 1,
        timeout_seconds: float | None = None,
        **kwargs,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.tool_defs = tool_defs
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        self.parser = parser or Parser()
        self.rubric = rubric or Rubric()
        self.env_id = env_id or ""
        self.max_workers = max_workers
        self.score_rollouts = score_rollouts
        self.pass_threshold = pass_threshold
        self.max_turns = max_turns
        self.timeout_seconds = timeout_seconds

        # Dataset storage (Verifiers pattern: lazy builders)
        self.dataset: Any | None = None
        self.eval_dataset: Any | None = None
        self.dataset_source: DatasetBuilder | None = None
        self.eval_dataset_source: DatasetBuilder | None = None

        if dataset is not None:
            if callable(dataset) and not isinstance(dataset, (list, dict)):
                self.dataset_source = dataset  # Lazy builder
            else:
                self.dataset_source = lambda ds=dataset: ds
                self.build_dataset()  # Eagerly build for raw datasets
        if eval_dataset is not None:
            if callable(eval_dataset) and not isinstance(eval_dataset, (list, dict)):
                self.eval_dataset_source = eval_dataset
            else:
                self.eval_dataset_source = lambda ds=eval_dataset: ds
                self.build_eval_dataset()

        # Sampling args
        self.sampling_args: SamplingArgs = {"n": 1, "extra_body": {}}
        if sampling_args is not None:
            for key, value in sampling_args.items():
                if key != "extra_body":
                    self.sampling_args[key] = value
                else:
                    self.sampling_args["extra_body"].update(value)

        # Lifecycle handlers (Verifiers decorator pattern)
        self._stop_conditions = discover_decorated(self, "stop")
        self._cleanup_handlers = discover_decorated(self, "cleanup")
        self._teardown_handlers = discover_decorated(self, "teardown")

    # ------------------------------------------------------------------
    # Dataset building (Verifiers pattern)
    # ------------------------------------------------------------------

    def build_dataset(self) -> Any | None:
        """Build and cache the training dataset from source if needed."""
        if self.dataset is not None:
            return self.dataset
        if self.dataset_source is None:
            return None
        self.dataset = self.dataset_source()
        self.dataset = self._ensure_prompt(self.dataset)
        return self.dataset

    def build_eval_dataset(self) -> Any | None:
        """Build and cache the evaluation dataset from source if needed."""
        if self.eval_dataset is not None:
            return self.eval_dataset
        if self.eval_dataset_source is None:
            return None
        self.eval_dataset = self.eval_dataset_source()
        self.eval_dataset = self._ensure_prompt(self.eval_dataset)
        return self.eval_dataset

    def _ensure_prompt(self, dataset: Any) -> Any:
        """Ensure prompt column exists in dataset.

        Converts question/input/query column to prompt messages.
        (From Verifiers environment.py _ensure_prompt pattern)
        """
        if dataset is None:
            return None

        # If it's a list of dicts, check if prompt exists
        if isinstance(dataset, list) and len(dataset) > 0 and isinstance(dataset[0], dict):
            if "prompt" in dataset[0]:
                return dataset  # Already has prompt

            # Create prompt from question/input/query
            rows = []
            for row in dataset:
                question = row.get("prompt", row.get("query", row.get("question", row.get("input", ""))))
                if isinstance(question, str):
                    messages = []
                    if self.system_prompt:
                        messages.append({"role": "system", "content": self.system_prompt})
                    if self.few_shot:
                        messages.extend(self.few_shot)
                    messages.append({"role": "user", "content": question})
                    row = {**row, "prompt": messages}
                rows.append(row)
            return rows

        # If it's a HuggingFace Dataset
        if hasattr(dataset, 'map') and hasattr(dataset, 'column_names'):
            if "prompt" in dataset.column_names:
                return dataset

            question_key = None
            for key in ["question", "query", "input", "text"]:
                if key in dataset.column_names:
                    question_key = key
                    break

            if question_key:
                system_prompt = self.system_prompt

                def format_prompt_fn(row):
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": row[question_key]})
                    return {"prompt": messages}

                dataset = dataset.map(format_prompt_fn)
            return dataset

        return dataset

    def get_dataset(self, n: int = -1, seed: int | None = None) -> Any:
        """Get the training dataset, optionally shuffled and sampled."""
        self.build_dataset()
        if self.dataset is None:
            raise ValueError("No dataset available")
        dataset = self.dataset
        if seed is not None and hasattr(dataset, 'shuffle'):
            dataset = dataset.shuffle(seed=seed)
        if n > 0 and hasattr(dataset, 'select'):
            n = min(n, len(dataset))
            return dataset.select(range(n))
        return dataset

    def get_eval_dataset(self, n: int = -1, seed: int | None = None) -> Any:
        """Get the eval dataset, falling back to train dataset."""
        self.build_eval_dataset()
        if self.eval_dataset is None:
            self.logger.warning("eval_dataset not set, falling back to train dataset")
            return self.get_dataset(n, seed)
        dataset = self.eval_dataset
        if seed is not None and hasattr(dataset, 'shuffle'):
            dataset = dataset.shuffle(seed=seed)
        if n > 0 and hasattr(dataset, 'select'):
            n = min(n, len(dataset))
            return dataset.select(range(n))
        return dataset

    # ------------------------------------------------------------------
    # Rollout pipeline (Verifiers pattern)
    # ------------------------------------------------------------------

    @abstractmethod
    async def rollout(
        self,
        input: RolloutInput,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """Run a single rollout. Subclasses implement this."""
        pass

    def _create_state(self, input: RolloutInput) -> State:
        """Create initial state from input.

        (From Verifiers environment.py init_state pattern)
        """
        state_input = deepcopy(input)
        if isinstance(state_input.info, str):
            try:
                state_input.info = json.loads(state_input.info)
            except (json.JSONDecodeError, TypeError):
                pass

        # Build state dict with all fields accessible at top level
        input_dict = state_input.model_dump() if hasattr(state_input, 'model_dump') else dict(state_input)
        state = State(input=input_dict)

        # Expose key fields at top level (Verifiers input forwarding pattern)
        state["task"] = dict(input_dict)
        state["prompt"] = input_dict.get("prompt", [])
        state["answer"] = input_dict.get("answer", "")
        state["example_id"] = input_dict.get("example_id", 0)
        state["info"] = input_dict.get("info", {})
        state["is_completed"] = False
        state["is_truncated"] = False
        state["trajectory"] = []
        state["completion"] = None
        state["trajectory_id"] = uuid.uuid4().hex
        state["reward"] = None
        state["metrics"] = None
        state["error"] = None
        state["timing"] = RolloutTiming()
        return state

    async def _score_state(self, state: State) -> None:
        """Score a completed rollout state."""
        if self.score_rollouts:
            await self.rubric.score_rollout(state)
        else:
            await self.rubric.dummy_score_rollout(state)

    async def _cleanup_state(self, state: State) -> None:
        """Run cleanup handlers on a state."""
        await self.rubric.cleanup(state)

    async def run_single(
        self,
        input: RolloutInput,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> RolloutOutput:
        """Run a single rollout and return the output."""
        state = await self.rollout(input, model, api_key, base_url, sampling_args)
        state["timing"].scoring.start = time.time()
        await self._score_state(state)
        state["timing"].scoring.end = time.time()
        await self._cleanup_state(state)
        return state_to_output(state)

    # ------------------------------------------------------------------
    # Evaluation runner (Verifiers generate() pattern simplified)
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        sampling_args: SamplingArgs | None = None,
        max_concurrent: int = -1,
        on_progress: ProgressCallback | None = None,
        on_log: LogCallback | None = None,
    ) -> GenerateOutputs:
        """Evaluate the model on the evaluation dataset.

        Follows Verifiers generate() pattern with semaphore concurrency.
        """
        dataset = self.get_eval_dataset()
        if dataset is None:
            raise ValueError("No dataset available for evaluation")

        # Convert dataset to RolloutInput list
        inputs: list[RolloutInput] = []
        if hasattr(dataset, 'to_list'):
            rows = dataset.to_list()
        elif isinstance(dataset, list):
            rows = dataset
        else:
            rows = list(dataset)

        for i, row in enumerate(rows):
            if isinstance(row, dict):
                prompt = row.get("prompt", row.get("query", row.get("question", row.get("input", ""))))
                if isinstance(prompt, str):
                    prompt = [UserMessage(content=prompt)]
                answer = row.get("answer", row.get("reference_answer", row.get("target", "")))
                inputs.append(RolloutInput(
                    prompt=prompt,
                    example_id=i,
                    answer=str(answer),
                    info=row.get("info", {}),
                ))
            else:
                inputs.append(row)

        # Set up concurrency (Verifiers semaphore pattern)
        sem = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None

        # Run rollouts
        outputs: list[RolloutOutput] = []
        total = len(inputs)
        completed = 0

        async def run_one(inp: RolloutInput) -> RolloutOutput:
            nonlocal completed
            if sem:
                async with sem:
                    result = await self.run_single(inp, model, api_key, base_url, sampling_args)
            else:
                result = await self.run_single(inp, model, api_key, base_url, sampling_args)
            completed += 1
            if on_progress:
                avg_reward = sum(o.get("reward", 0) for o in outputs) / max(len(outputs), 1)
                on_progress(outputs, [result], {"avg_reward": avg_reward})
            if on_log:
                on_log(f"Completed {completed}/{total}: reward={result.get('reward', 0):.3f}")
            return result

        tasks = [run_one(inp) for inp in inputs]
        outputs = await asyncio.gather(*tasks)

        # Build metadata
        rewards = [o.get("reward", 0) for o in outputs]
        metadata = GenerateMetadata(
            env_id=self.env_id,
            model=model,
            num_examples=len(inputs),
            rollouts_per_example=1,
            avg_reward=sum(rewards) / max(len(rewards), 1),
            avg_metrics={},
        )

        return GenerateOutputs(outputs=list(outputs), metadata=metadata)

    def evaluate_sync(self, model: str, **kwargs) -> GenerateOutputs:
        """Synchronous wrapper for evaluate()."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.evaluate(model, **kwargs))

        return asyncio.run(self.evaluate(model, **kwargs))
