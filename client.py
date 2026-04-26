# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Debugzero Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import DebugzeroAction, DebugzeroObservation, DebugzeroState
except ImportError:
    from models import DebugzeroAction, DebugzeroObservation, DebugzeroState


class DebugzeroEnv(
    EnvClient[DebugzeroAction, DebugzeroObservation, DebugzeroState]
):
    """
    Client for the DebugZero Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions for Proposer/Solver roles.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with DebugzeroEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.current_code)
        ...
        ...     result = client.step(DebugzeroAction(role="proposer", code="buggy code"))
        ...     print(result.observation.tests_passed)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = DebugzeroEnv.from_docker_image("debugZero-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(DebugzeroAction(role="proposer", code="import os"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: DebugzeroAction) -> Dict:
        """
        Convert DebugzeroAction to JSON payload for step message.

        Args:
            action: DebugzeroAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "role": action.role,
            "code": action.code,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DebugzeroObservation]:
        """
        Parse server response into StepResult[DebugzeroObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with DebugzeroObservation
        """
        obs_data = payload.get("observation", {})
        reward_value = payload.get("reward", obs_data.get("reward"))
        done_value = payload.get("done", obs_data.get("done", False))
        observation = DebugzeroObservation(
            role_next=obs_data.get("role_next", "proposer"),
            current_code=obs_data.get("current_code", ""),
            execution_result=obs_data.get("execution_result", ""),
            tests_passed=obs_data.get("tests_passed", False),
            syntax_error=obs_data.get("syntax_error", False),
            score=obs_data.get("score", 0.0),
            done=done_value,
            reward=reward_value,
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=reward_value,
            done=done_value,
        )

    def _parse_state(self, payload: Dict) -> DebugzeroState:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object
        """
        return DebugzeroState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            seed_id=payload.get("seed_id", ""),
            original_code=payload.get("original_code", ""),
            current_code=payload.get("current_code", ""),
            role_turn=payload.get("role_turn", "proposer"),
        )
