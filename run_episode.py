import asyncio
from typing import cast
import subprocess
from server.debugZero_environment import DebugzeroEnvironment
from models import DebugzeroAction, DebugzeroObservation

def test_local_env():
    print("Testing DebugzeroEnvironment directly...")
    env = DebugzeroEnvironment()
    obs = env.reset()
    print(f"\nInitial state (role_next={obs.role_next})")
    
    # Proposer turn: Send invalid code to test sandbox
    print("\n--- Proposer turn (malicious code) ---")
    action = DebugzeroAction(role="proposer", code="import os\nos.system('echo hacked')")
    obs = env.step(action)
    print(f"Status: done={obs.done}, reward={obs.reward}, syntax_error={obs.syntax_error}")
    print(f"Execution Result:\n{obs.execution_result.strip()}")
    
    obs = env.reset()
    # Proposer turn: Inject valid bug
    print("\n--- Proposer turn (valid bug) ---")
    buggy_code = "def has_close_elements(numbers, threshold):\n    pass"
    action = DebugzeroAction(role="proposer", code=buggy_code)
    obs = env.step(action)
    print(f"Status: role_next={obs.role_next}, done={obs.done}, reward={obs.reward}, syntax_error={obs.syntax_error}")
    
    # Solver turn: Fix the bug
    print("\n--- Solver turn (fix bug) ---")
    valid_code = obs.current_code.replace("pass", "return False")
    action = DebugzeroAction(role="solver", code=valid_code)
    obs = env.step(action)
    print(f"Status: role_next={obs.role_next}, done={obs.done}, reward={obs.reward}, tests_passed={obs.tests_passed}")

if __name__ == "__main__":
    test_local_env()