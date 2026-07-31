"""
Shared "call this with a hard wall-clock budget" helper.

Several call sites (ReAct routing decisions, GenAI treatment enhancement,
network health checks) need to bound a potentially slow/hanging call without
blocking on the still-running background thread if it doesn't finish in time.
A naive ``with ThreadPoolExecutor() as pool:`` blocks on ``__exit__`` (which
calls ``shutdown(wait=True)``) until the orphaned call finishes — defeating
the whole point of the timeout when the underlying call is genuinely stuck.
This helper gets that right once, so every caller does too.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from typing import Callable, TypeVar

T = TypeVar("T")


def call_with_timeout(fn: Callable[..., T], *args, timeout: float, **kwargs) -> T:
    """Call ``fn(*args, **kwargs)`` with a hard wall-clock budget.

    Raises ``TimeoutError`` if it doesn't complete in time. Uses
    ``shutdown(wait=False)`` deliberately: control returns immediately at the
    budget, regardless of how long the background thread keeps running. The
    orphaned thread is cleaned up by the interpreter at process exit.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(fn, *args, **kwargs)
        return future.result(timeout=timeout)
    except _FutureTimeout as e:
        raise TimeoutError(f"call exceeded {timeout}s budget") from e
    finally:
        pool.shutdown(wait=False)
