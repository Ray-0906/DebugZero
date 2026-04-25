from __future__ import annotations

import statistics
from collections import deque


# Global solve rate history buffer: {seed_id: deque(maxlen=20)}
solve_rate_history: dict[str, deque[float]] = {}


def reset_reward_history() -> None:
    solve_rate_history.clear()


def get_solve_rate(seed_id: str) -> float:
    if seed_id not in solve_rate_history or not solve_rate_history[seed_id]:
        return 0.5
    return statistics.mean(solve_rate_history[seed_id])


def record_solve_result(seed_id: str, solved: bool) -> None:
    if seed_id not in solve_rate_history:
        solve_rate_history[seed_id] = deque(maxlen=20)
    solve_rate_history[seed_id].append(1.0 if solved else 0.0)
