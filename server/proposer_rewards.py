from __future__ import annotations

import ast

try:
    from .reward_history import get_solve_rate
except ImportError:
    from reward_history import get_solve_rate


def is_effectively_unchanged(original_code: str, candidate_code: str) -> bool:
    try:
        return ast.dump(ast.parse(original_code)) == ast.dump(ast.parse(candidate_code))
    except SyntaxError:
        return original_code.strip() == candidate_code.strip()


def compute_proposer_reward(meta: dict) -> float:
    if meta.get("syntax_error", False):
        return -0.5

    if meta.get("unsafe_code", False):
        return -0.5

    if meta.get("unchanged_code", False):
        return 0.0

    if meta.get("tests_passed", True):
        return 0.0

    plausibility_bonus = meta.get("plausibility_score", 0.0)
    learnability_bonus = 0.0
    solve_rate = get_solve_rate(meta["seed_id"])
    if 0.2 <= solve_rate <= 0.8:
        learnability_bonus = 1.0

    return 1.0 + plausibility_bonus + learnability_bonus
