from __future__ import annotations

import ast
import statistics
from collections import deque

from thefuzz import fuzz


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


def is_effectively_unchanged(original_code: str, candidate_code: str) -> bool:
    try:
        return ast.dump(ast.parse(original_code)) == ast.dump(ast.parse(candidate_code))
    except SyntaxError:
        return original_code.strip() == candidate_code.strip()


def compute_ast_distance(original_code: str, mutated_code: str) -> float:
    """
    Computes the string similarity distance between the AST dumps of the original
    and mutated code using thefuzz (Levenshtein based).
    Zero edits = 0 score.
    Targeted (small) edit = high plausibility (closer to 1.0).
    Random / wide corruption = low score.
    """
    try:
        orig_ast = ast.dump(ast.parse(original_code))
        mut_ast = ast.dump(ast.parse(mutated_code))
    except SyntaxError:
        return 0.0

    ratio = fuzz.ratio(orig_ast, mut_ast)

    # The Fuzz defaults to percentage (0 to 100)
    # Empirical calibration: simple AST mutations typically result in
    # a fuzz ratio of 85-98%.
    if 85 <= ratio:
        return 1.0
    if 50 <= ratio < 85:
        return max(0.1, (ratio - 50) / 35.0)
    return 0.0


def compute_proposer_reward(meta: dict) -> float:
    if meta.get("syntax_error", False):
        return -0.5

    if meta.get("unsafe_code", False):
        return -0.5

    if meta.get("unchanged_code", False):
        return 0.0

    if meta.get("changed_but_passing", False):
        return -0.1

    if meta.get("tests_passed", True):
        return 0.0

    plausibility_bonus = meta.get("plausibility_score", 0.0)
    learnability_bonus = 0.0
    solve_rate = get_solve_rate(meta["seed_id"])
    if 0.2 <= solve_rate <= 0.8:
        learnability_bonus = 1.0

    return 1.0 + plausibility_bonus + learnability_bonus


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
