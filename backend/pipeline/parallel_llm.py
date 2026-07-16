"""Bounded parallel execution helpers for independent LLM requests."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, List, TypeVar


T = TypeVar("T")
R = TypeVar("R")


def get_llm_concurrency() -> int:
    """Return a conservative, user-overridable LLM request limit."""
    try:
        value = int(os.getenv("AUTOCLIP_LLM_CONCURRENCY", "3"))
    except (TypeError, ValueError):
        value = 3
    return max(1, min(value, 8))


def run_parallel_ordered(
    tasks: Iterable[T],
    worker: Callable[[T], R],
    on_completed: Callable[[int, int], None] | None = None,
) -> List[R]:
    """Run independent work concurrently and return results in input order."""
    task_list = list(tasks)
    if not task_list:
        return []

    workers = min(get_llm_concurrency(), len(task_list))
    if workers == 1:
        results = []
        for completed, task in enumerate(task_list, start=1):
            results.append(worker(task))
            if on_completed:
                on_completed(completed, len(task_list))
        return results

    indexed_results: Dict[int, R] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="autoclip-llm") as executor:
        futures = {
            executor.submit(worker, task): index
            for index, task in enumerate(task_list)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            indexed_results[futures[future]] = future.result()
            if on_completed:
                on_completed(completed, len(task_list))

    return [indexed_results[index] for index in range(len(task_list))]
