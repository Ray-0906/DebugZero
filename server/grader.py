# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Grading helpers for DebugZero environment submissions.

This module owns the server-side execution result shaping used by the
environment. It intentionally preserves the current environment reward behavior:
server observations return a neutral reward, while training reward shaping lives
in ``server/rewards.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .seed_bank import SeedSpec
except ImportError:
    from server.seed_bank import SeedSpec

try:
    from .executor import execute_code
except ImportError:
    from executor import execute_code


MAX_EXECUTION_OUTPUT_CHARS = 500
NEUTRAL_REWARD = 0.0


@dataclass(frozen=True)
class GradingResult:
    execution_result: str
    tests_passed: bool
    syntax_error: bool
    reward: float = NEUTRAL_REWARD


def grade_submission(code: str, seed: SeedSpec) -> GradingResult:
    result = execute_code(code, seed.test)
    return GradingResult(
        execution_result=result.output[:MAX_EXECUTION_OUTPUT_CHARS] if result.output else "",
        tests_passed=result.passed,
        syntax_error=result.syntax_error,
        reward=NEUTRAL_REWARD,
    )
