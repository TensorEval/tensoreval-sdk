"""Rubric class for weighted multi-grader reward functions.

Ported from PrimeIntellect Verifiers (MIT License).
Adapted imports for TensorEval namespace.
"""

import asyncio
import inspect
import logging
from collections.abc import Callable, Mapping
from typing import Any, cast, get_origin

from tensoreval.core.decorators import discover_decorated
from tensoreval.core.types import (
    GroupRewardFunc,
    RewardFunc,
    RolloutScore,
    State,
)

ScoreObjectProvider = Callable[[State], Mapping[str, object]]
GroupScoreObjectProvider = Callable[[list[State]], Mapping[str, object]]


async def _maybe_await(func: Callable, *args, **kwargs):
    """Call func and await the result if it's a coroutine."""
    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def _maybe_call_with_named_args(func: Callable, **objects):
    """Call func with only the keyword arguments it declares."""
    sig = inspect.signature(func)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return await _maybe_await(func, **objects)
    allowed = {key: value for key, value in objects.items() if key in sig.parameters}
    return await _maybe_await(func, **allowed)


class Rubric:
    """Rubric class for reward functions.

    Each reward function takes:
    - prompt, completion, answer, state, info, task (as needed)
    - Returns: float (reward score)
    """

    def __init__(
        self,
        funcs: list[RewardFunc | GroupRewardFunc] | None = None,
        weights: list[float] | None = None,
        parser: Any | None = None,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.funcs = funcs or []
        self.weights = weights or []
        if not self.weights:
            self.weights = [1.0] * len(self.funcs)
        elif len(self.weights) != len(self.funcs):
            raise ValueError(
                f"Number of weights ({len(self.weights)}) must match number of functions ({len(self.funcs)})"
            )
        self.parser = parser
        self.class_objects: dict[str, Any] = {}
        if self.parser:
            self.class_objects["parser"] = self.parser
        self.score_object_providers: list[ScoreObjectProvider] = []
        self.group_score_object_providers: list[GroupScoreObjectProvider] = []
        self._cleanup_handlers = discover_decorated(self, "cleanup")
        self._teardown_handlers = discover_decorated(self, "teardown")

    def add_reward_func(self, func: RewardFunc, weight: float = 1.0):
        """Add a reward function with a weight."""
        self.funcs.append(func)
        self.weights.append(weight)

    def add_metric(self, func: RewardFunc, weight: float = 0.0):
        """Add a metric function (weight=0 means not summed into reward)."""
        self.funcs.append(func)
        self.weights.append(weight)

    def add_class_object(self, name: str, obj: Any):
        """Add an object available to reward functions."""
        self.class_objects[name] = obj

    def _is_group_func(self, func: RewardFunc) -> bool:
        """Check if a function is a GroupRewardFunc by inspecting its signature."""
        sig = inspect.signature(func)
        param_names = set(sig.parameters.keys())
        group_indicators = {"states", "prompts", "completions", "answers", "tasks", "infos"}
        return_annotation = sig.return_annotation
        returns_list = return_annotation is list or get_origin(return_annotation) is list
        return bool(param_names & group_indicators) or returns_list

    def score_objects(self, state: State) -> dict[str, Any]:
        """Build the argument namespace for reward functions."""
        objects = {
            "prompt": state.get("prompt"),
            "completion": state.get("completion"),
            "answer": state.get("answer", ""),
            "state": state,
            "info": state.get("info", {}),
            **self.class_objects,
        }
        objects["task"] = state.get("task")
        for provider in self.score_object_providers:
            objects.update(provider(state))
        return objects

    def group_score_objects(self, states: list[State]) -> dict[str, Any]:
        """Build the argument namespace for group reward functions."""
        objects = dict(
            prompts=[state.get("prompt") for state in states],
            completions=[state.get("completion") for state in states],
            answers=[state.get("answer", "") for state in states],
            states=states,
            tasks=[state.get("task") for state in states],
            infos=[state.get("info", {}) for state in states],
            **self.class_objects,
        )
        for provider in self.group_score_object_providers:
            objects.update(provider(states))
        return objects

    async def _call_individual_reward_func(self, func: RewardFunc, state: State) -> float:
        """Invoke a reward function with only the required arguments."""
        sig = inspect.signature(func)
        merged = self.score_objects(state)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            try:
                ans = float(await _maybe_await(func, **merged))
            except Exception as e:
                self.logger.error(f"Error calling reward function {func.__name__}: {e}")
                ans = 0.0
        else:
            allowed = {k: v for k, v in merged.items() if k in sig.parameters}
            try:
                ans = float(await _maybe_await(func, **allowed))
            except Exception as e:
                self.logger.error(f"Error calling reward function {func.__name__}: {e}")
                ans = 0.0
        return ans

    def _get_group_reward_funcs(self) -> list[GroupRewardFunc]:
        return cast(
            list[GroupRewardFunc],
            [func for func in self.funcs if self._is_group_func(func)],
        )

    @property
    def has_group_rewards(self) -> bool:
        return bool(self._get_group_reward_funcs())

    async def _call_group_reward_func(self, func: GroupRewardFunc, states: list[State]) -> list[float]:
        """Invoke a group reward function."""
        sig = inspect.signature(func)
        merged = self.group_score_objects(states)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            try:
                ans = await _maybe_await(func, **merged)
            except Exception as e:
                self.logger.error(f"Error calling group reward function {func.__name__}: {e}")
                ans = [0.0] * len(states)
        else:
            allowed = {k: v for k, v in merged.items() if k in sig.parameters}
            try:
                ans = await _maybe_await(func, **allowed)
            except Exception as e:
                self.logger.error(f"Error calling group reward function {func.__name__}: {e}")
                ans = [0.0] * len(states)
        return ans

    async def cleanup(self, state: State):
        """Run all cleanup handlers."""
        for handler in self._cleanup_handlers:
            await _maybe_call_with_named_args(handler, state=state, **self.class_objects)

    async def teardown(self):
        """Run all teardown handlers."""
        for handler in self._teardown_handlers:
            await handler()

    async def dummy_score_rollout(self, state: State):
        """Score a single rollout with dummy rewards."""
        state["reward"] = 0.0
        state["metrics"] = {}

    async def score_rollout(self, state: State):
        """Evaluate all reward functions for a single rollout."""
        reward_funcs = [func for func in self.funcs if not self._is_group_func(func)]
        group_reward_funcs = self._get_group_reward_funcs()
        assert len(reward_funcs) > 0 and len(group_reward_funcs) == 0, (
            "Rubric.score_rollout requires at least one individual-level reward function and no group-level reward functions"
        )
        reward_scores = []
        for func in reward_funcs:
            reward_scores.append(await self._call_individual_reward_func(func=func, state=state))
        filtered_weights = [
            weight for func, weight in zip(self.funcs, self.weights) if not self._is_group_func(func)
        ]
        rewards = RolloutScore(
            metrics={func.__name__: reward for func, reward in zip(reward_funcs, reward_scores)},
            reward=sum(reward * weight for reward, weight in zip(reward_scores, filtered_weights)),
        )
        state["reward"] = rewards["reward"]
        state["metrics"] = rewards["metrics"]

    async def dummy_score_group(self, states: list[State]):
        """Score a group of rollouts with dummy rewards."""
        for state in states:
            await self.dummy_score_rollout(state)

    async def score_group(self, states: list[State]):
        """Score a group of rollouts together. Computes advantages for GRPO."""
        num_states = len(states)
        if num_states == 0:
            self.logger.warning("No states to score")
            return
        aggregated_rewards = [0.0] * num_states
        aggregated_metrics: dict[str, list[float]] = {}

        for func, weight in zip(self.funcs, self.weights):
            is_group = self._is_group_func(func)
            if is_group:
                group_func = cast(GroupRewardFunc, func)
                scores = await self._call_group_reward_func(group_func, states)
                func_name = func.__name__
                if func_name not in aggregated_metrics:
                    aggregated_metrics[func_name] = [0.0] * num_states
                for i in range(num_states):
                    aggregated_rewards[i] += scores[i] * weight
                    aggregated_metrics[func_name][i] = scores[i]
            else:
                reward_func = cast(RewardFunc, func)
                score_tasks = [self._call_individual_reward_func(reward_func, state) for state in states]
                scores = await asyncio.gather(*score_tasks)
                func_name = func.__name__
                if func_name not in aggregated_metrics:
                    aggregated_metrics[func_name] = [0.0] * num_states
                for i in range(num_states):
                    aggregated_rewards[i] += scores[i] * weight
                    aggregated_metrics[func_name][i] = scores[i]

        avg_reward = sum(aggregated_rewards) / num_states
        for i, state in enumerate(states):
            state["reward"] = aggregated_rewards[i]
            state["advantage"] = aggregated_rewards[i] - avg_reward
            for t in state.get("trajectory", []):
                if t.get("advantage") is None:
                    t["advantage"] = state["advantage"]
                if t.get("reward") is None:
                    t["reward"] = state["reward"]
            state["metrics"] = {func_name: values[i] for func_name, values in aggregated_metrics.items()}
