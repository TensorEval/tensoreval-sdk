"""Multi-turn environment — model interacts over multiple turns.

Ported from PrimeIntellect Verifiers (MIT License).
"""

import asyncio
import time
from abc import abstractmethod
from typing import Any

from tensoreval.core.types import (
    Messages,
    RolloutInput,
    SamplingArgs,
    State,
    TrajectoryStep,
    UserMessage,
)
from tensoreval.envs.environment import Environment


class MultiTurnEnv(Environment):
    """Environment for multi-turn interactions.

    Subclasses must implement env_response() to provide environment feedback.
    """

    def __init__(self, max_turns: int = 10, timeout_seconds: float | None = None, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = max_turns
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        """Generate a response from the environment after the model's turn."""
        pass

    async def rollout(
        self,
        input: RolloutInput,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """Run a multi-turn rollout."""
        state = self._create_state(input)
        state["timing"].generation.start = time.time()

        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=api_key or "dummy",
            base_url=base_url or "https://api.openai.com/v1",
        )

        prompt = state.get("prompt", [])
        if isinstance(prompt, str):
            prompt = [UserMessage(content=prompt)]
        if self.system_prompt:
            from tensoreval.core.types import SystemMessage
            prompt = [SystemMessage(content=self.system_prompt)] + prompt

        merged_args = {**self.sampling_args, **(sampling_args or {})}
        current_messages = list(prompt)

        try:
            for turn in range(self.max_turns):
                # Call model
                messages_for_api = [m.model_dump() if hasattr(m, 'model_dump') else m for m in current_messages]
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages_for_api,
                    max_tokens=merged_args.get("max_tokens", 1024),
                    temperature=merged_args.get("temperature", 0.7),
                )
                content = response.choices[0].message.content or ""

                from tensoreval.core.types import AssistantMessage
                assistant_msg = AssistantMessage(content=content)
                completion = [assistant_msg]

                # Record trajectory step
                state["trajectory"].append(TrajectoryStep(
                    prompt=current_messages,
                    completion=completion,
                    trajectory_id=state["trajectory_id"],
                ))

                current_messages.append(assistant_msg)

                # Check if done (no tool calls, model gave final answer)
                if response.choices[0].finish_reason == "stop":
                    break

                # Get environment response
                env_msgs = await self.env_response(current_messages, state)
                if env_msgs:
                    current_messages.extend(env_msgs)

            state["completion"] = current_messages[len(prompt):]

        except Exception as e:
            state["error"] = e

        state["timing"].generation.end = time.time()
        state["is_completed"] = True
        return state
