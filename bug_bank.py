from __future__ import annotations

import random
from dataclasses import dataclass

from seed_bank import SEED_BANK, SeedSpec

try:
    from server.bug_injector import inject_bug
    from server.executor import execute_code
    from server.graders import compute_ast_distance
except ImportError:
    from .server.bug_injector import inject_bug
    from .server.executor import execute_code
    from .server.graders import compute_ast_distance


V1_BUG_OPERATORS = (
    "wrong_operator",
    "wrong_builtin",
    "condition_negation",
    "off_by_one",
    "loop_boundary_shift",
    "slice_boundary_corruption",
)

MAX_VERIFIED_BUGS_PER_SEED = 4
HOLDOUT_BUGS_PER_SEED = 1
MAX_MUTATION_ATTEMPTS = 4

BUG_OPERATOR_PRIORITY = {
    "loop_boundary_shift": 6,
    "slice_boundary_corruption": 5,
    "condition_negation": 4,
    "wrong_operator": 3,
    "off_by_one": 2,
    "wrong_builtin": 1,
}


@dataclass(frozen=True)
class BugSample:
    seed_id: str
    original_code: str
    buggy_code: str
    bug_operator: str
    execution_result: str


@dataclass(frozen=True)
class BugBank:
    train_samples: tuple[BugSample, ...]
    eval_samples: tuple[BugSample, ...]


def validate_seed(seed: SeedSpec) -> None:
    result = execute_code(seed.original_code, seed.test)
    if result.syntax_error or not result.passed:
        raise ValueError(f"Seed {seed.seed_id} does not pass its canonical tests.")


def build_bug_bank(
    seeds: tuple[SeedSpec, ...] = SEED_BANK,
    max_verified_bugs_per_seed: int = MAX_VERIFIED_BUGS_PER_SEED,
    holdout_bugs_per_seed: int = HOLDOUT_BUGS_PER_SEED,
) -> BugBank:
    train_samples: list[BugSample] = []
    eval_samples: list[BugSample] = []

    for seed in seeds:
        validate_seed(seed)
        verified_samples = _collect_verified_bugs(seed)
        verified_samples = sorted(
            verified_samples,
            key=lambda sample: _bug_difficulty_score(seed, sample),
            reverse=True,
        )

        if len(verified_samples) <= holdout_bugs_per_seed:
            raise ValueError(
                f"Seed {seed.seed_id} only produced {len(verified_samples)} verified bugs."
            )

        eval_samples.extend(verified_samples[:holdout_bugs_per_seed])
        train_samples.extend(
            verified_samples[
                holdout_bugs_per_seed : holdout_bugs_per_seed + max_verified_bugs_per_seed
            ]
        )

    return BugBank(
        train_samples=tuple(train_samples),
        eval_samples=tuple(eval_samples),
    )


def _collect_verified_bugs(seed: SeedSpec) -> list[BugSample]:
    verified_samples: list[BugSample] = []
    seen_codes: set[str] = set()

    for bug_operator in V1_BUG_OPERATORS:
        for attempt in range(MAX_MUTATION_ATTEMPTS):
            random.seed(f"{seed.seed_id}:{bug_operator}:{attempt}")
            buggy_code, changed = inject_bug(seed.original_code, bug_operator)
            if not changed:
                continue
            if buggy_code in seen_codes:
                continue

            result = execute_code(buggy_code, seed.test)
            if result.syntax_error or result.passed:
                continue

            seen_codes.add(buggy_code)
            verified_samples.append(
                BugSample(
                    seed_id=seed.seed_id,
                    original_code=seed.original_code,
                    buggy_code=buggy_code,
                    bug_operator=bug_operator,
                    execution_result=result.output[:500] if result.output else "",
                )
            )

    return verified_samples


def _bug_difficulty_score(seed: SeedSpec, sample: BugSample) -> float:
    operator_score = BUG_OPERATOR_PRIORITY.get(sample.bug_operator, 0)
    ast_similarity = compute_ast_distance(seed.original_code, sample.buggy_code)
    execution_lines = _count_nonempty_lines(sample.execution_result)

    # Bias toward bugs that preserve the function shape but still require a real local repair.
    local_repair_score = ast_similarity
    execution_signal = min(execution_lines / 4.0, 1.0)
    return float(operator_score) + local_repair_score + execution_signal


def _count_nonempty_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())
