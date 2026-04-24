# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DebugZero Environment Implementation for Absolute Zero debugging self-play.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DebugzeroAction, DebugzeroObservation, DebugzeroState
except ImportError:
    from models import DebugzeroAction, DebugzeroObservation, DebugzeroState

try:
    from .executor import execute_code, ExecutionResult
    from .bug_injector import inject_bug
    from .plausibility import compute_ast_distance
except ImportError:
    from executor import execute_code, ExecutionResult
    from bug_injector import inject_bug
    from plausibility import compute_ast_distance

# Stub for HumanEval dataset holding
HUMANEVAL_SEED = {
    "seed_id": "HumanEval/0",
    "prompt": "def has_close_elements(numbers: list[float], threshold: float) -> bool:",
    "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False\n",
    "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n"
}

class DebugzeroEnvironment(Environment):
    """
    Dual-role DebugZero Environment wrapping a Python sandbox execution
    for Proposer bug injection and Solver bug fixing.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = DebugzeroState(
            episode_id=str(uuid4()), 
            step_count=0,
            seed_id=HUMANEVAL_SEED["seed_id"],
            original_code=f'{HUMANEVAL_SEED["prompt"]}\n{HUMANEVAL_SEED["canonical_solution"]}',
            current_code=f'{HUMANEVAL_SEED["prompt"]}\n{HUMANEVAL_SEED["canonical_solution"]}',
            role_turn="proposer"
        )
        self._reset_count = 0

    def reset(self) -> DebugzeroObservation:
        self._state = DebugzeroState(
            episode_id=str(uuid4()), 
            step_count=0,
            seed_id=HUMANEVAL_SEED["seed_id"],
            original_code=f'{HUMANEVAL_SEED["prompt"]}\n{HUMANEVAL_SEED["canonical_solution"]}',
            current_code=f'{HUMANEVAL_SEED["prompt"]}\n{HUMANEVAL_SEED["canonical_solution"]}',
            role_turn="proposer"
        )
        self._reset_count += 1

        return DebugzeroObservation(
            role_next="proposer",
            current_code=self._state.current_code,
            execution_result="",
            tests_passed=True,
            syntax_error=False,
            done=False,
            reward=0.0,
        )

    def step(self, action: DebugzeroAction) -> DebugzeroObservation:  # type: ignore[override]
        self._state.step_count += 1
        
        tests = HUMANEVAL_SEED["test"]
        
        if action.role == "proposer":
            # The proposer injects buggy code here
            # In our full implementation this string will be parsed and injected into the original code
            # using AST operators from the bug injector
            self._state.current_code = action.code
            
            # evaluate
            result = execute_code(self._state.current_code, tests)
            
            self._state.role_turn = "solver"
            
            # The actual reward will be calculated by the training loop using executor results 
            # including the `compute_ast_distance` plausibility score.
            return DebugzeroObservation(
                role_next="solver",
                current_code=self._state.current_code,
                execution_result=result.output[:500],  # truncate avoiding overflow
                tests_passed=result.passed,
                syntax_error=result.syntax_error,
                done=False,
                reward=0.0
            )
            
        elif action.role == "solver":
            # The solver fixes the bug
            self._state.current_code = action.code
            
            # evaluate
            result = execute_code(self._state.current_code, tests)
            
            self._state.role_turn = "end"
            
            return DebugzeroObservation(
                role_next="end",
                current_code=self._state.current_code,
                execution_result=result.output[:500],
                tests_passed=result.passed,
                syntax_error=result.syntax_error,
                done=True,
                reward=0.0
            )
            
        return DebugzeroObservation(role_next="end", done=True)
            
    @property
    def state(self) -> DebugzeroState:
        return self._state
