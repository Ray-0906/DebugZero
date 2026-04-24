# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the DebugZero Environment.

The debugZero environment implements the Absolute Zero paradigm for debugging self-play.
"""

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field
from typing import Optional


class DebugzeroAction(Action):
    """Action for the DebugZero environment representing the Proposer or Solver inputs."""

    role: str = Field(..., description="Role taking action: 'proposer' or 'solver'")
    code: str = Field(..., description="Code injected (by proposer) or fixed (by solver)")


class DebugzeroObservation(Observation):
    """Observation from the DebugZero environment following sandbox execution."""

    role_next: str = Field(default="proposer", description="The role supposed to play next")
    current_code: str = Field(default="", description="The current state of the python code")
    execution_result: str = Field(default="", description="Result of evaluating tests in the sandbox")
    tests_passed: bool = Field(default=False, description="Whether the tests passed")
    syntax_error: bool = Field(default=False, description="Whether the code had a parse/syntax error")

class DebugzeroState(State):
    """State for the DebugZero environment, extending default state with seed context."""
    seed_id: str = Field(default="", description="ID of the HumanEval function")
    original_code: str = Field(default="", description="Original clean seed code")
    current_code: str = Field(default="", description="Current code after Proposer/Solver turn")
    role_turn: str = Field(default="proposer", description="Current turn's role")
