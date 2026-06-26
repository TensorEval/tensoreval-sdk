"""Async utilities for TensorEval.

Ported from PrimeIntellect Verifiers (MIT License).
"""

import asyncio
import inspect
from collections.abc import Coroutine
from typing import Any, AsyncContextManager, Callable, Optional, TypeVar

T = TypeVar("T")


async def with_sem(sem: AsyncContextManager, coro: Coroutine[Any, Any, T]) -> T:
    """Wrap a coroutine with a semaphore context manager."""
    try:
        async with sem:
            return await coro
    finally:
        coro.close()


async def maybe_await(func: Callable, *args, **kwargs):
    """Call func and await the result if it's a coroutine."""
    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def maybe_call_with_named_args(func: Callable, **objects):
    """Call func with only the keyword arguments it declares."""
    sig = inspect.signature(func)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return await maybe_await(func, **objects)
    allowed = {key: value for key, value in objects.items() if key in sig.parameters}
    return await maybe_await(func, **allowed)


class NullAsyncContext:
    """No-op async context manager."""
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


async def maybe_semaphore(limit: Optional[int] = None) -> AsyncContextManager:
    """Return a real semaphore if limit > 0, else a no-op context manager."""
    if limit and limit > 0:
        return asyncio.Semaphore(limit)
    return NullAsyncContext()
