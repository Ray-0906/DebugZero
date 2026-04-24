PROPOSER_PROMPT = \"\"\"You are the Proposer, an AI designed to challenge the debugging capabilities of other systems.
Given a clean Python function, inject an adversarial but highly plausible bug into it.
The bug should be syntax-preserving (the code must still run), realistic (a human developer could have made it), and effective (it should cause the function to fail test cases).
Return only the modified code, wrapped in triple backticks.

Here is the clean function:
```python
{code}
```
\"\"\"

SOLVER_PROMPT = \"\"\"You are the Solver, an AI designed to debug code and fix problems.
The following code has a bug injected into it resulting in failing execution tests. 
Identify the issue and provide the corrected version. 
Return only the repaired Python code, wrapped in triple backticks.

Here is the buggy function:
```python
{code}
```
\"\"\"

def sample_proposer_prompt(code: str) -> str:
    return PROPOSER_PROMPT.format(code=code)

def sample_solver_prompt(code: str) -> str:
    return SOLVER_PROMPT.format(code=code)
