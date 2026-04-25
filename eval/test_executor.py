import pytest
from server.executor import is_safe, execute_code

def test_executor_is_safe():
    assert is_safe("def add(a,b):\n    return a+b") == True
    
    # Blocked imports
    assert is_safe("import os") == False
    assert is_safe("from os import path") == False
    assert is_safe("import sys\nsys.exit(1)") == False
    assert is_safe("import subprocess as sp") == False
    
    # Syntax errors
    assert is_safe("def add(a b):") == False
    
def test_execute_code():
    code = "def add(a, b): return a + b"
    tests = "assert add(1, 2) == 3"
    res = execute_code(code, tests)
    assert res.passed == True
    assert res.syntax_error == False
    
    code_buggy = "def add(a, b): return a - b"
    res2 = execute_code(code_buggy, tests)
    assert res2.passed == False
    assert res2.syntax_error == False

def test_execute_code_calls_humaneval_check():
    tests = "def check(candidate):\n    assert candidate(1, 2) == 3"

    good = "def add(a, b): return a + b"
    good_res = execute_code(good, tests)
    assert good_res.passed == True
    assert good_res.syntax_error == False

    buggy = "def add(a, b): return a - b"
    buggy_res = execute_code(buggy, tests)
    assert buggy_res.passed == False
    assert buggy_res.syntax_error == False
