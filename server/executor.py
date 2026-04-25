import os
import subprocess
import sys
import tempfile
import ast

BLOCKED_IMPORTS = ["os", "sys", "subprocess", "shutil", "pathlib"]
BLOCKED_BUILTINS = ["__import__", "eval", "exec", "open"]

def is_safe(code: str) -> bool:
    """
    Check if the code contains any blocked imports strings. 
    Also performs a quick AST parse check to see if it parses.
    """
    # 1. Simple text matching for basic imports
    for mod in BLOCKED_IMPORTS:
        if f"import {mod}" in code or f"from {mod}" in code:
            return False
            
    # 2. Block built-in execution loopholes
    for b in BLOCKED_BUILTINS:
        if b in code:
            return False
            
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
        
    # 3. Deep AST walk to find dynamic imports disguised in functions/aliases
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in BLOCKED_IMPORTS:
                    return False
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in BLOCKED_IMPORTS:
                return False
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_BUILTINS:
                return False
                
    return True

class ExecutionResult:
    def __init__(self, passed: bool, output: str, syntax_error: bool = False, timeout_error: bool = False):
        self.passed = passed
        self.output = output
        self.syntax_error = syntax_error
        self.timeout_error = timeout_error

def execute_code(code: str, tests: str, timeout: int = 5) -> ExecutionResult:
    """
    Executes the provided python code alongside its tests in an isolated subprocess.
    Returns the execution results.
    """
    full_code = f"{code}\n\n{tests}"
    
    if not is_safe(full_code):
        # We need to distinguish between unsafe imports and actual syntax errors
        try:
            ast.parse(full_code)
            return ExecutionResult(passed=False, output="Unsafe import detected.", syntax_error=False)
        except SyntaxError as e:
            return ExecutionResult(passed=False, output=f"SyntaxError: {e}", syntax_error=True)
            
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "exec_script.py")
        with open(temp_file, "w") as f:
            f.write(full_code)
            
        try:
            # We run the process completely isolated with no stdout buffers blocking us
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                return ExecutionResult(passed=True, output=result.stdout)
            else:
                syntax_error = "SyntaxError" in result.stderr
                return ExecutionResult(passed=False, output=result.stderr, syntax_error=syntax_error)
                
        except subprocess.TimeoutExpired:
            return ExecutionResult(passed=False, output="Execution timed out.", timeout_error=True)
        except Exception as e:
            return ExecutionResult(passed=False, output=str(e))
