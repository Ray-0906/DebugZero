import pytest
from server.bug_injector import inject_bug

SEED = """def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
"""

def test_missing_base_case():
    mutated, changed = inject_bug(SEED, "missing_base_case")
    # Base case should be pass
    assert changed == True
    assert "return 1" not in mutated
    assert "pass" in mutated
    
def test_off_by_one():
    # Value 0 or 1 should be shifted
    mutated, changed = inject_bug(SEED, "off_by_one")
    assert changed == True
    assert mutated != SEED
    
SEED2 = "def count_up(n): return range(n)"

def test_loop_boundary_shift():
    mutated, changed = inject_bug(SEED2, "loop_boundary_shift")
    assert changed == True
    assert "range(n + 1)" in mutated
    
SEED3 = "def do_any(arr): return any(arr)"

def test_wrong_builtin():
    mutated, changed = inject_bug(SEED3, "wrong_builtin")
    assert changed == True
    assert "all(arr)" in mutated
    
SEED4 = "def check(x): return x > 0"
def test_condition_negation():
    # Should wrap in not
    mutated, changed = inject_bug("if x > 0:\n    pass", "condition_negation")
    assert changed == True
    assert "if not x > 0:" in mutated

def test_safety_check_blocks():
    # Injecting import os directly shouldn't be possible through AST operations, but let's test if the safety injector logic would block it if we injected it manually.
    from server.bug_injector import is_safe_injection
    unsafe = SEED + "\nimport os"
    assert is_safe_injection(unsafe) == False
    assert is_safe_injection(SEED) == True
