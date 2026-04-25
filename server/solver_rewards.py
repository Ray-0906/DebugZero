from __future__ import annotations

try:
    from .reward_history import record_solve_result
except ImportError:
    from reward_history import record_solve_result


def compute_solver_reward(meta: dict) -> float:
    solved = meta.get("tests_passed", False)
    syntax_error = meta.get("syntax_error", True)
    unsafe_code = meta.get("unsafe_code", False)

    record_solve_result(meta["seed_id"], solved and not syntax_error and not unsafe_code)

    if syntax_error or unsafe_code:
        return -0.5
    if solved:
        return 1.0
    return 0.0
