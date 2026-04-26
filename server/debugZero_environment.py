# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DebugZero Environment Implementation for adversarial bug-fixing self-play.
"""

from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import DebugzeroAction, DebugzeroObservation, DebugzeroState
    from .tasks import SEED_BANK, SeedSpec
except ImportError:
    from models import DebugzeroAction, DebugzeroObservation, DebugzeroState
    from server.tasks import SEED_BANK, SeedSpec

try:
    from .executor import execute_code
except ImportError:
    from executor import execute_code


class DebugzeroEnvironment(Environment):
    """
    Dual-role DebugZero Environment wrapping a Python sandbox execution
    for Proposer bug injection and Solver bug fixing.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._reset_count = 0
        self._current_seed = SEED_BANK[0]
        self._current_bug_operator: str | None = None
        self._state = self._build_state(self._current_seed)

    def reset(self) -> DebugzeroObservation:
        seed = SEED_BANK[self._reset_count % len(SEED_BANK)]
        self._reset_count += 1
        self._current_seed = seed
        self._current_bug_operator = None
        self._state = self._build_state(seed)

        return DebugzeroObservation(
            role_next="proposer",
            current_code=self._state.current_code,
            execution_result="",
            tests_passed=True,
            syntax_error=False,
            done=False,
            reward=0.0,
            metadata=self._observation_metadata(),
        )

    def step(self, action: DebugzeroAction) -> DebugzeroObservation:  # type: ignore[override]
        self._state.step_count += 1

        tests = self._current_seed.test

        if action.role == "proposer":
            self._state.current_code = action.code
            result = execute_code(self._state.current_code, tests)
            self._state.role_turn = "solver"

            return DebugzeroObservation(
                role_next="solver",
                current_code=self._state.current_code,
                execution_result=result.output[:500] if result.output else "",
                tests_passed=result.passed,
                syntax_error=result.syntax_error,
                done=False,
                reward=0.0,
                metadata=self._observation_metadata(),
            )

        if action.role == "solver":
            self._state.current_code = action.code
            result = execute_code(self._state.current_code, tests)
            self._state.role_turn = "end"

            return DebugzeroObservation(
                role_next="proposer",
                current_code=self._state.current_code,
                execution_result=result.output[:500] if result.output else "",
                tests_passed=result.passed,
                syntax_error=result.syntax_error,
                done=True,
                reward=0.0,
                metadata=self._observation_metadata(),
            )

        return DebugzeroObservation(
            role_next="end",
            current_code="",
            execution_result="",
            tests_passed=False,
            syntax_error=False,
            done=True,
            reward=0.0,
            metadata=self._observation_metadata(),
        )

    @property
    def state(self) -> DebugzeroState:
        return self._state

    def _build_state(self, seed: SeedSpec) -> DebugzeroState:
        return DebugzeroState(
            episode_id=str(uuid4()),
            step_count=0,
            seed_id=seed.seed_id,
            original_code=seed.original_code,
            current_code=seed.original_code,
            role_turn="proposer",
        )

    def _observation_metadata(self) -> dict[str, str]:
        metadata = {
            "seed_id": self._state.seed_id,
            "original_code": self._state.original_code,
        }
        if self._current_bug_operator:
            metadata["bug_operator"] = self._current_bug_operator
        return metadata
