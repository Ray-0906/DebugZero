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
    from .bug_injector import infer_bug_operator
    from .graders import (
        compute_ast_distance,
        compute_proposer_reward,
        compute_solver_reward,
        is_effectively_unchanged,
    )
    from .executor import execute_code
except ImportError:
    from bug_injector import infer_bug_operator
    from graders import (
        compute_ast_distance,
        compute_proposer_reward,
        compute_solver_reward,
        is_effectively_unchanged,
    )
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
        self._current_score = 0.0
        self._proposer_created_bug = False
        self._state = self._build_state(self._current_seed)

    def reset(self) -> DebugzeroObservation:
        seed = SEED_BANK[self._reset_count % len(SEED_BANK)]
        self._reset_count += 1
        self._current_seed = seed
        self._current_bug_operator = None
        self._current_score = 0.0
        self._proposer_created_bug = False
        self._state = self._build_state(seed)

        return self._build_observation(
            role_next="proposer",
            execution_result="",
            tests_passed=True,
            syntax_error=False,
            done=False,
            reward=0.0,
            score=0.0,
        )

    def step(self, action: DebugzeroAction) -> DebugzeroObservation:  # type: ignore[override]
        self._state.step_count += 1

        tests = self._current_seed.test

        if action.role == "proposer":
            self._state.current_code = action.code
            result = execute_code(self._state.current_code, tests)
            self._state.role_turn = "solver"
            reward, score = self._proposer_step_feedback(action.code, result)

            return self._build_observation(
                role_next="solver",
                execution_result=self._truncate_execution_output(result.output),
                tests_passed=result.passed,
                syntax_error=result.syntax_error,
                done=False,
                reward=reward,
                score=score,
            )

        if action.role == "solver":
            self._state.current_code = action.code
            result = execute_code(self._state.current_code, tests)
            self._state.role_turn = "end"
            reward, score = self._solver_step_feedback(result)

            return self._build_observation(
                role_next="proposer",
                execution_result=self._truncate_execution_output(result.output),
                tests_passed=result.passed,
                syntax_error=result.syntax_error,
                done=True,
                reward=reward,
                score=score,
            )

        self._current_score = 0.0
        self._proposer_created_bug = False
        return self._build_observation(
            role_next="end",
            execution_result="",
            tests_passed=False,
            syntax_error=False,
            done=True,
            reward=0.0,
            score=0.0,
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

    def _build_observation(
        self,
        *,
        role_next: str,
        execution_result: str,
        tests_passed: bool,
        syntax_error: bool,
        done: bool,
        reward: float,
        score: float,
    ) -> DebugzeroObservation:
        self._current_score = score
        return DebugzeroObservation(
            role_next=role_next,
            current_code=self._state.current_code,
            execution_result=execution_result,
            tests_passed=tests_passed,
            syntax_error=syntax_error,
            score=score,
            done=done,
            reward=reward,
            metadata=self._observation_metadata(),
        )

    def _proposer_step_feedback(self, candidate_code: str, result: object) -> tuple[float, float]:
        original_code = self._state.original_code
        execution_output = getattr(result, "output", "") or ""
        syntax_error = bool(getattr(result, "syntax_error", False))
        tests_passed = bool(getattr(result, "passed", False))
        unsafe_code = execution_output.startswith("Unsafe import detected.")

        unchanged_code = is_effectively_unchanged(original_code, candidate_code)
        changed_but_passing = (not unchanged_code) and tests_passed and (not syntax_error)
        plausibility_score = 0.0 if syntax_error else compute_ast_distance(original_code, candidate_code)

        reward = compute_proposer_reward(
            {
                "seed_id": self._state.seed_id,
                "tests_passed": tests_passed,
                "syntax_error": syntax_error,
                "unsafe_code": unsafe_code,
                "unchanged_code": unchanged_code,
                "changed_but_passing": changed_but_passing,
                "plausibility_score": plausibility_score,
            }
        )

        valid_bug = (not tests_passed) and (not syntax_error) and (not unsafe_code)
        self._proposer_created_bug = valid_bug
        self._current_bug_operator = infer_bug_operator(original_code, candidate_code) if valid_bug else None
        score = 0.5 if valid_bug else 0.0
        return reward, score

    def _solver_step_feedback(self, result: object) -> tuple[float, float]:
        execution_output = getattr(result, "output", "") or ""
        syntax_error = bool(getattr(result, "syntax_error", False))
        tests_passed = bool(getattr(result, "passed", False))
        unsafe_code = execution_output.startswith("Unsafe import detected.")

        reward = compute_solver_reward(
            {
                "seed_id": self._state.seed_id,
                "tests_passed": tests_passed,
                "syntax_error": syntax_error,
                "unsafe_code": unsafe_code,
            }
        )

        solved = tests_passed and (not syntax_error) and (not unsafe_code)
        score = 1.0 if solved else (0.5 if self._proposer_created_bug else 0.0)
        return reward, score

    def _truncate_execution_output(self, output: str) -> str:
        return output[:500] if output else ""

    def _observation_metadata(self) -> dict[str, str]:
        metadata = {
            "seed_id": self._state.seed_id,
            "original_code": self._state.original_code,
        }
        if self._current_bug_operator:
            metadata["bug_operator"] = self._current_bug_operator
        return metadata
