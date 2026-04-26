from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedSpec:
    seed_id: str
    entrypoint: str
    prompt: str
    canonical_solution: str
    test: str

    @property
    def original_code(self) -> str:
        return f"{self.prompt}\n{self.canonical_solution}"


SEED_BANK = (
    SeedSpec(
        seed_id="HumanEval/0",
        entrypoint="has_close_elements",
        prompt="def has_close_elements(numbers: list[float], threshold: float) -> bool:",
        canonical_solution=(
            "    for idx, elem in enumerate(numbers):\n"
            "        for idx2, elem2 in enumerate(numbers):\n"
            "            if idx != idx2:\n"
            "                distance = abs(elem - elem2)\n"
            "                if distance < threshold:\n"
            "                    return True\n"
            "    return False\n"
        ),
        test=(
            "def check(candidate):\n"
            "    assert candidate([1.0, 2.0, 3.0], 0.5) is False\n"
            "    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) is True\n\n"
            "check(has_close_elements)\n"
        ),
    ),
    SeedSpec(
        seed_id="DebugZero/1",
        entrypoint="sum_to_n",
        prompt="def sum_to_n(n: int) -> int:",
        canonical_solution=(
            "    total = 0\n"
            "    for value in range(n + 1):\n"
            "        total += value\n"
            "    return total\n"
        ),
        test=(
            "def check(candidate):\n"
            "    assert candidate(0) == 0\n"
            "    assert candidate(1) == 1\n"
            "    assert candidate(5) == 15\n"
            "    assert candidate(10) == 55\n\n"
            "check(sum_to_n)\n"
        ),
    ),
    SeedSpec(
        seed_id="DebugZero/2",
        entrypoint="middle_slice",
        prompt="def middle_slice(values: list[int]) -> list[int]:",
        canonical_solution=(
            "    if len(values) <= 2:\n"
            "        return []\n"
            "    return values[1:-1]\n"
        ),
        test=(
            "def check(candidate):\n"
            "    assert candidate([1]) == []\n"
            "    assert candidate([1, 2]) == []\n"
            "    assert candidate([1, 2, 3]) == [2]\n"
            "    assert candidate([1, 2, 3, 4, 5]) == [2, 3, 4]\n\n"
            "check(middle_slice)\n"
        ),
    ),
    SeedSpec(
        seed_id="DebugZero/3",
        entrypoint="is_non_decreasing",
        prompt="def is_non_decreasing(values: list[int]) -> bool:",
        canonical_solution=(
            "    return all(values[idx] <= values[idx + 1] for idx in range(len(values) - 1))\n"
        ),
        test=(
            "def check(candidate):\n"
            "    assert candidate([]) is True\n"
            "    assert candidate([5]) is True\n"
            "    assert candidate([1, 2, 2, 3]) is True\n"
            "    assert candidate([3, 2]) is False\n"
            "    assert candidate([1, 3, 2, 4]) is False\n\n"
            "check(is_non_decreasing)\n"
        ),
    ),
    SeedSpec(
        seed_id="DebugZero/4",
        entrypoint="count_nonempty",
        prompt="def count_nonempty(strings: list[str]) -> int:",
        canonical_solution=(
            "    total = 0\n"
            "    for text in strings:\n"
            "        if len(text) > 0:\n"
            "            total += 1\n"
            "    return total\n"
        ),
        test=(
            "def check(candidate):\n"
            "    assert candidate([]) == 0\n"
            "    assert candidate(['', '']) == 0\n"
            "    assert candidate(['a', '', 'bc', '']) == 2\n"
            "    assert candidate(['hi', 'there']) == 2\n\n"
            "check(count_nonempty)\n"
        ),
    ),
    SeedSpec(
        seed_id="DebugZero/5",
        entrypoint="running_max",
        prompt="def running_max(values: list[int]) -> int:",
        canonical_solution=(
            "    best = values[0]\n"
            "    for idx in range(1, len(values)):\n"
            "        if values[idx] > best:\n"
            "            best = values[idx]\n"
            "    return best\n"
        ),
        test=(
            "def check(candidate):\n"
            "    assert candidate([3]) == 3\n"
            "    assert candidate([3, 1, 5, 2]) == 5\n"
            "    assert candidate([-1, -4, -2]) == -1\n"
            "    assert candidate([0, 0, 0]) == 0\n\n"
            "check(running_max)\n"
        ),
    ),
)

SEED_BY_ID = {seed.seed_id: seed for seed in SEED_BANK}


def get_seed_by_id(seed_id: str) -> SeedSpec:
    return SEED_BY_ID[seed_id]


def legacy_seed_dict(seed: SeedSpec) -> dict[str, str]:
    return {
        "seed_id": seed.seed_id,
        "entrypoint": seed.entrypoint,
        "prompt": seed.prompt,
        "canonical_solution": seed.canonical_solution,
        "test": seed.test,
    }


HUMANEVAL_SEED = legacy_seed_dict(SEED_BANK[0])
