from __future__ import annotations

import re


PROPOSER_PROMPT = """You are the Proposer in a debugging self-play game.
Given a clean Python function, inject a realistic logical bug into it.
Rules:
- Make exactly one small logical change.
- Keep the code valid Python.
- Keep the same function signature.
- Preserve the overall structure and formatting as much as possible.
- Do not rewrite the whole function or add unrelated edits.
- Prefer realistic bug types such as boundary, comparison, condition, or slice mistakes.
- Return only the full modified Python code inside triple backticks.

Clean function:
```python
{code}
```
"""

SOLVER_PROMPT_FULL = """You are the Solver in a debugging self-play game.
The following Python code is failing its tests.
Repair the bug and return the full fixed Python code inside triple backticks.

Buggy function:
```python
{code}
```

Observed failure:
{execution_result}
"""

SOLVER_PROMPT_CONCISE = """You are the Solver in a debugging self-play game.
Fix the bug with the smallest correct local change and return only the full fixed Python code inside triple backticks.

Buggy function:
```python
{code}
```

Failure summary:
{execution_result}
"""

TRACEBACK_HINTS = (
    "Traceback",
    "AssertionError",
    "SyntaxError",
    "TypeError",
    "NameError",
    "ValueError",
    "IndexError",
    "KeyError",
    "ZeroDivisionError",
    "RuntimeError",
    "Timeout",
)


def summarize_failure_output(execution_result: str, *, max_lines: int = 3, max_chars: int = 220) -> str:
    text = execution_result.strip()
    if not text:
        return "No failure output provided."

    if text in {"Unsafe import detected.", "Execution timed out."} or text.startswith("SyntaxError:"):
        return _truncate_text(text, max_chars)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "No failure output provided."

    traceback_positions = [idx for idx, line in enumerate(lines) if "Traceback" in line]
    if traceback_positions:
        tail = lines[traceback_positions[-1] :]
        if len(tail) > max_lines:
            lines = [tail[0], *tail[-(max_lines - 1) :]]
        else:
            lines = tail
    else:
        interesting_lines = [line for line in lines if any(hint in line for hint in TRACEBACK_HINTS)]
        if interesting_lines:
            lines = interesting_lines[-max_lines:]
        else:
            lines = lines[-max_lines:]

    summary = "\n".join(lines)
    return _truncate_text(summary, max_chars)


def _truncate_text(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"[ \t]+", " ", text.strip())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(0, max_chars - 3)].rstrip() + "..."


def sample_proposer_prompt(code: str) -> str:
    return PROPOSER_PROMPT.format(code=code)


def sample_solver_prompt(
    code: str,
    execution_result: str = "",
    *,
    mode: str = "concise",
) -> str:
    failure_output = summarize_failure_output(execution_result)
    if mode == "full":
        failure_output = execution_result.strip() if execution_result.strip() else "No failure output provided."
        return SOLVER_PROMPT_FULL.format(code=code, execution_result=failure_output)
    return SOLVER_PROMPT_CONCISE.format(code=code, execution_result=failure_output)
