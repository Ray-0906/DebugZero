import sys
import subprocess
print("\n[DebugZero] Launch Request Received. Checking A100 Hardware and allocating VRAM...", flush=True)
# 1. Models
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

class DebugzeroAction(Action):
    role: str = Field(..., description="Role taking action: 'proposer' or 'solver'")
    code: str = Field(..., description="Code injected (by proposer) or fixed (by solver)")

class DebugzeroObservation(Observation):
    role_next: str = Field(default="proposer", description="The role supposed to play next")
    current_code: str = Field(default="", description="The current state of the python code")
    execution_result: str = Field(default="", description="Result of evaluating tests")
    tests_passed: bool = Field(default=False, description="Whether the tests passed")
    syntax_error: bool = Field(default=False, description="Whether the code had a parse/syntax error")

class DebugzeroState(State):
    seed_id: str = Field(default="", description="ID of the HumanEval function")
    original_code: str = Field(default="", description="Original clean seed code")
    current_code: str = Field(default="", description="Current code after turn")
    role_turn: str = Field(default="proposer", description="Current turn's role")

# 2. Seed Bank
from dataclasses import dataclass

@dataclass(frozen=True)
class SeedSpec:
    seed_id: str
    entrypoint: str
    prompt: str
    canonical_solution: str
    test: str

    @property
    def original_code(self) -> str:
        return f"{self.prompt}\n{self.canonical_solution}"

SEED_BANK = (
    SeedSpec("HumanEval/0", "has_close_elements", "def has_close_elements(numbers: list[float], threshold: float) -> bool:", "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False\n", "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) is False\n    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) is True\n\ncheck(has_close_elements)\n"),
    SeedSpec("DebugZero/1", "sum_to_n", "def sum_to_n(n: int) -> int:", "    total = 0\n    for value in range(n + 1):\n        total += value\n    return total\n", "def check(candidate):\n    assert candidate(0) == 0\n    assert candidate(1) == 1\n    assert candidate(5) == 15\n    assert candidate(10) == 55\n\ncheck(sum_to_n)\n"),
    SeedSpec("DebugZero/2", "middle_slice", "def middle_slice(values: list[int]) -> list[int]:", "    if len(values) <= 2:\n        return []\n    return values[1:-1]\n", "def check(candidate):\n    assert candidate([1]) == []\n    assert candidate([1, 2]) == []\n    assert candidate([1, 2, 3]) == [2]\n    assert candidate([1, 2, 3, 4, 5]) == [2, 3, 4]\n\ncheck(middle_slice)\n"),
    SeedSpec("DebugZero/3", "is_non_decreasing", "def is_non_decreasing(values: list[int]) -> bool:", "    return all(values[idx] <= values[idx + 1] for idx in range(len(values) - 1))\n", "def check(candidate):\n    assert candidate([]) is True\n    assert candidate([5]) is True\n    assert candidate([1, 2, 2, 3]) is True\n    assert candidate([3, 2]) is False\n    assert candidate([1, 3, 2, 4]) is False\n\ncheck(is_non_decreasing)\n"),
    SeedSpec("DebugZero/4", "count_nonempty", "def count_nonempty(strings: list[str]) -> int:", "    total = 0\n    for text in strings:\n        if len(text) > 0:\n            total += 1\n    return total\n", "def check(candidate):\n    assert candidate([]) == 0\n    assert candidate(['', '']) == 0\n    assert candidate(['a', '', 'bc', '']) == 2\n    assert candidate(['hi', 'there']) == 2\n\ncheck(count_nonempty)\n"),
    SeedSpec("DebugZero/5", "running_max", "def running_max(values: list[int]) -> int:", "    best = values[0]\n    for idx in range(1, len(values)):\n        if values[idx] > best:\n            best = values[idx]\n    return best\n", "def check(candidate):\n    assert candidate([3]) == 3\n    assert candidate([3, 1, 5, 2]) == 5\n    assert candidate([-1, -4, -2]) == -1\n    assert candidate([0, 0, 0]) == 0\n\ncheck(running_max)\n"),
)
SEED_BY_ID = {seed.seed_id: seed for seed in SEED_BANK}
def get_seed_by_id(seed_id: str) -> SeedSpec: return SEED_BY_ID[seed_id]

# 3. Executor
import os
import subprocess
import sys
import tempfile
import ast

BLOCKED_IMPORTS = ["os", "sys", "subprocess", "shutil", "pathlib"]
BLOCKED_BUILTINS = ["__import__", "eval", "exec", "open"]

def is_safe(code: str) -> bool:
    for mod in BLOCKED_IMPORTS:
        if f"import {mod}" in code or f"from {mod}" in code:
            return False
    for b in BLOCKED_BUILTINS:
        if b in code: return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in BLOCKED_IMPORTS: return False
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in BLOCKED_IMPORTS: return False
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_BUILTINS: return False
    return True

class ExecutionResult:
    def __init__(self, passed: bool, output: str, syntax_error: bool = False, timeout_error: bool = False):
        self.passed = passed
        self.output = output
        self.syntax_error = syntax_error
        self.timeout_error = timeout_error

def execute_code(code: str, tests: str, timeout: int = 5) -> ExecutionResult:
    full_code = f"{code}\n\n{tests}"
    if not is_safe(full_code):
        try:
            ast.parse(full_code)
            return ExecutionResult(passed=False, output="Unsafe import detected.", syntax_error=False)
        except SyntaxError as e:
            return ExecutionResult(passed=False, output=f"SyntaxError: {e}", syntax_error=True)
            
    try:
        # Run fully in memory via sys.executable -c
        result = subprocess.run([sys.executable, "-c", full_code], capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return ExecutionResult(passed=True, output=result.stdout)
        else:
            syntax_error = "SyntaxError" in result.stderr
            return ExecutionResult(passed=False, output=result.stderr, syntax_error=syntax_error)
    except subprocess.TimeoutExpired:
        return ExecutionResult(passed=False, output="Execution timed out.", timeout_error=True)
    except Exception as e:
        return ExecutionResult(passed=False, output=str(e))

# 4. Plausibility Score
from thefuzz import fuzz

def compute_ast_distance(original_code: str, mutated_code: str) -> float:
    try:
        orig_ast = ast.dump(ast.parse(original_code))
        mut_ast = ast.dump(ast.parse(mutated_code))
    except SyntaxError:
        return 0.0
    ratio = fuzz.ratio(orig_ast, mut_ast)
    if 85 <= ratio: return 1.0 
    elif 50 <= ratio < 85: return max(0.1, (ratio - 50) / 35.0)
    else: return 0.0

# 5. Prompts and Summarization (Dual Role Sampler)
import re

PROPOSER_PROMPT = """You are the Proposer in a debugging self-play game.
Given a clean Python function, inject a realistic logical bug into it.
Rules:
- Make exactly one small logical change.
- Keep the code valid Python.
- Keep the same function signature.
- Preserve the overall structure and formatting as much as possible.
- Prefer one of these mutation families: off_by_one, wrong_operator, wrong_builtin,
  condition_negation, loop_boundary_shift, or slice_boundary_corruption.
- Aim for an edge-case behavior change, not a cosmetic refactor.
- Avoid helper extraction, renaming-only edits, comment-only changes, or multi-line rewrites.
- Return only the full modified Python code inside triple backticks.
{focus_instruction}

Clean function:
```python
{code}
```
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

TRACEBACK_HINTS = ("Traceback", "AssertionError", "SyntaxError", "TypeError", "NameError", "ValueError", "IndexError", "KeyError", "ZeroDivisionError", "RuntimeError", "Timeout",)

def _truncate_text(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"[ \t]+", " ", text.strip())
    if len(cleaned) <= max_chars: return cleaned
    return cleaned[: max(0, max_chars - 3)].rstrip() + "..."

def summarize_failure_output(execution_result: str, *, max_lines: int = 3, max_chars: int = 220) -> str:
    text = execution_result.strip()
    if not text: return "No failure output provided."
    if text in {"Unsafe import detected.", "Execution timed out."} or text.startswith("SyntaxError:"):
        return _truncate_text(text, max_chars)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines: return "No failure output provided."
    traceback_positions = [idx for idx, line in enumerate(lines) if "Traceback" in line]
    if traceback_positions:
        tail = lines[traceback_positions[-1] :]
        if len(tail) > max_lines: lines = [tail[0], *tail[-(max_lines - 1) :]]
        else: lines = tail
    else:
        interesting_lines = [line for line in lines if any(hint in line for hint in TRACEBACK_HINTS)]
        if interesting_lines: lines = interesting_lines[-max_lines:]
        else: lines = lines[-max_lines:]
    return _truncate_text("\n".join(lines), max_chars)

def sample_proposer_prompt(code: str, bug_focus: str | None = None) -> str:
    focus_instruction = ""
    if bug_focus:
        focus_instruction = f"- Focus specifically on the `{bug_focus}` mutation family.\n- Keep the edit local so the bug can be repaired with a small fix."
    return PROPOSER_PROMPT.format(code=code, focus_instruction=focus_instruction)

def sample_solver_prompt(code: str, execution_result: str = "", *, mode: str = "concise") -> str:
    failure_output = summarize_failure_output(execution_result)
    return SOLVER_PROMPT_CONCISE.format(code=code, execution_result=failure_output)

# 6. Bug Operations & Injector
import random
import copy

BUILTIN_PAIRS = {"min": "max", "max": "min", "any": "all", "all": "any", "sum": "len", "len": "sum"}

def is_safe_injection(code: str) -> bool:
    for blocked in BLOCKED_IMPORTS:
        if f"import {blocked}" in code or f"from {blocked}" in code:
            return False
    return True

class BugInjectorVisitor(ast.NodeTransformer):
    def __init__(self, target_operator: str):
        super().__init__()
        self.target_operator = target_operator
        self.mutated = False

    def visit_Constant(self, node):
        self.generic_visit(node)
        if self.mutated: return node
        if self.target_operator == "off_by_one" and isinstance(node.value, int) and not isinstance(node.value, bool):
            node.value += random.choice([-1, 1])
            self.mutated = True
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if self.mutated: return node
        if self.target_operator == "wrong_operator":
            if isinstance(node.ops[0], ast.Lt):
                node.ops[0] = ast.GtE()
                self.mutated = True
            elif isinstance(node.ops[0], ast.LtE):
                node.ops[0] = ast.Gt()
                self.mutated = True
            elif isinstance(node.ops[0], ast.Gt):
                node.ops[0] = ast.LtE()
                self.mutated = True
            elif isinstance(node.ops[0], ast.GtE):
                node.ops[0] = ast.Lt()
                self.mutated = True
            elif isinstance(node.ops[0], ast.Eq):
                node.ops[0] = ast.NotEq()
                self.mutated = True
            elif isinstance(node.ops[0], ast.NotEq):
                node.ops[0] = ast.Eq()
                self.mutated = True
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if self.mutated: return node
        if self.target_operator == "wrong_operator":
            if isinstance(node.op, ast.Add):
                node.op = ast.Sub()
                self.mutated = True
            elif isinstance(node.op, ast.Sub):
                node.op = ast.Add()
                self.mutated = True
            elif isinstance(node.op, ast.Mult):
                node.op = ast.FloorDiv()
                self.mutated = True
            elif isinstance(node.op, ast.Div):
                node.op = ast.Mult()
                self.mutated = True
        return node
        
    def visit_Call(self, node):
        self.generic_visit(node)
        if self.mutated: return node
        if isinstance(node.func, ast.Name):
            if self.target_operator == "wrong_builtin" and node.func.id in BUILTIN_PAIRS:
                node.func.id = BUILTIN_PAIRS[node.func.id]
                self.mutated = True
            elif self.target_operator == "loop_boundary_shift" and node.func.id == "range":
                if len(node.args) == 1:
                    node.args[0] = ast.BinOp(left=node.args[0], op=ast.Add(), right=ast.Constant(value=1))
                    self.mutated = True
                elif len(node.args) == 2:
                    node.args[0] = ast.BinOp(left=node.args[0], op=ast.Sub(), right=ast.Constant(value=1))
                    self.mutated = True
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if self.mutated: return node
        if self.target_operator == "condition_negation":
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            self.mutated = True
        if self.target_operator == "missing_base_case":
            for idx, child in enumerate(node.body):
                if isinstance(child, ast.Return):
                    node.body[idx] = ast.Pass()
                    self.mutated = True
                    break
        return node
        
    def visit_Slice(self, node):
        self.generic_visit(node)
        if self.mutated: return node
        if self.target_operator == "slice_boundary_corruption":
            if node.lower is not None:
                node.lower = ast.BinOp(left=node.lower, op=ast.Add(), right=ast.Constant(value=1))
                self.mutated = True
            elif node.upper is not None:
                node.upper = ast.BinOp(left=node.upper, op=ast.Sub(), right=ast.Constant(value=1))
                self.mutated = True
        return node
        
    def visit_Assign(self, node):
        self.generic_visit(node)
        if self.mutated: return node
        if self.target_operator == "variable_swap" and getattr(node, "targets", None):
            if isinstance(node.targets[0], ast.Tuple) and len(node.targets[0].elts) >= 2:
                node.targets[0].elts[0], node.targets[0].elts[1] = node.targets[0].elts[1], node.targets[0].elts[0]
                self.mutated = True
        return node

def inject_bug(original_code: str, proposed_operator: str) -> tuple[str, bool]:
    try: tree = ast.parse(original_code)
    except SyntaxError: return original_code, False
    injector = BugInjectorVisitor(proposed_operator)
    mutated_tree = injector.visit(copy.deepcopy(tree))
    ast.fix_missing_locations(mutated_tree)
    mutated_code = ast.unparse(mutated_tree)
    if mutated_code.strip() == original_code.strip(): return original_code, False
    if not is_safe_injection(mutated_code): return original_code, False
    try: ast.parse(mutated_code)
    except SyntaxError: return original_code, False
    return mutated_code, True

def infer_bug_operator(original_code: str, candidate_code: str) -> str | None:
    try:
        original_tree = ast.parse(original_code)
        candidate_tree = ast.parse(candidate_code)
    except SyntaxError:
        return None
    if ast.dump(original_tree) == ast.dump(candidate_tree):
        return None
    return "unknown" # simplified for inline layout logic

# 7. Environment & Bug Bank
V1_BUG_OPERATORS = ("wrong_operator", "wrong_builtin", "condition_negation", "off_by_one", "loop_boundary_shift", "slice_boundary_corruption",)
MAX_VERIFIED_BUGS_PER_SEED = 4
HOLDOUT_BUGS_PER_SEED = 1
MAX_MUTATION_ATTEMPTS = 4
BUG_OPERATOR_PRIORITY = {"loop_boundary_shift": 6, "slice_boundary_corruption": 5, "condition_negation": 4, "wrong_operator": 3, "off_by_one": 2, "wrong_builtin": 1}

@dataclass(frozen=True)
class BugSample:
    seed_id: str
    original_code: str
    buggy_code: str
    bug_operator: str
    execution_result: str

@dataclass(frozen=True)
class BugBank:
    train_samples: tuple[BugSample, ...]
    eval_samples: tuple[BugSample, ...]

def validate_seed(seed: SeedSpec) -> None:
    result = execute_code(seed.original_code, seed.test)
    if result.syntax_error or not result.passed:
        raise ValueError(f"Seed {seed.seed_id} does not pass.")

def _count_nonempty_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())

def _bug_difficulty_score(seed: SeedSpec, sample: BugSample) -> float:
    operator_score = BUG_OPERATOR_PRIORITY.get(sample.bug_operator, 0)
    ast_similarity = compute_ast_distance(seed.original_code, sample.buggy_code)
    execution_lines = _count_nonempty_lines(sample.execution_result)
    return float(operator_score) + ast_similarity + min(execution_lines / 4.0, 1.0)

def _collect_verified_bugs(seed: SeedSpec) -> list[BugSample]:
    verified_samples: list[BugSample] = []
    seen_codes: set[str] = set()
    for bug_operator in V1_BUG_OPERATORS:
        for attempt in range(MAX_MUTATION_ATTEMPTS):
            random.seed(f"{seed.seed_id}:{bug_operator}:{attempt}")
            buggy_code, changed = inject_bug(seed.original_code, bug_operator)
            if not changed or buggy_code in seen_codes: continue
            result = execute_code(buggy_code, seed.test)
            if result.syntax_error or result.passed: continue
            seen_codes.add(buggy_code)
            verified_samples.append(BugSample(seed.seed_id, seed.original_code, buggy_code, bug_operator, result.output[:500] if result.output else ""))
    return verified_samples

def build_bug_bank() -> BugBank:
    train_samples, eval_samples = [], []
    for seed in SEED_BANK:
        validate_seed(seed)
        verified_samples = sorted(_collect_verified_bugs(seed), key=lambda sample: _bug_difficulty_score(seed, sample), reverse=True)
        if len(verified_samples) <= HOLDOUT_BUGS_PER_SEED: raise ValueError(f"Seed {seed.seed_id} under-produced.")
        eval_samples.extend(verified_samples[:HOLDOUT_BUGS_PER_SEED])
        train_samples.extend(verified_samples[HOLDOUT_BUGS_PER_SEED : HOLDOUT_BUGS_PER_SEED + MAX_VERIFIED_BUGS_PER_SEED])
    return BugBank(tuple(train_samples), tuple(eval_samples))

# 8. Training Rewards
import statistics
from collections import deque

solve_rate_history: dict[str, deque[float]] = {}
def reset_reward_history() -> None: solve_rate_history.clear()
def get_solve_rate(seed_id: str) -> float: return statistics.mean(solve_rate_history[seed_id]) if solve_rate_history.get(seed_id) else 0.5
def record_solve_result(seed_id: str, solved: bool) -> None:
    if seed_id not in solve_rate_history: solve_rate_history[seed_id] = deque(maxlen=20)
    solve_rate_history[seed_id].append(1.0 if solved else 0.0)

def is_effectively_unchanged(original_code: str, candidate_code: str) -> bool:
    try: return ast.dump(ast.parse(original_code)) == ast.dump(ast.parse(candidate_code))
    except SyntaxError: return original_code.strip() == candidate_code.strip()

def compute_proposer_reward(meta: dict) -> float:
    if meta.get("syntax_error", False) or meta.get("unsafe_code", False): return -0.5
    if meta.get("unchanged_code", False) or meta.get("tests_passed", True): return 0.0
    if meta.get("changed_but_passing", False): return -0.1
    plausibility_bonus = meta.get("plausibility_score", 0.0)
    learnability_bonus = 1.0 if 0.2 <= get_solve_rate(meta["seed_id"]) <= 0.8 else 0.0
    return 1.0 + plausibility_bonus + learnability_bonus

def compute_solver_reward(meta: dict) -> float:
    solved = meta.get("tests_passed", False)
    syntax_error = meta.get("syntax_error", True)
    unsafe_code = meta.get("unsafe_code", False)
    record_solve_result(meta["seed_id"], solved and not syntax_error and not unsafe_code)
    if syntax_error or unsafe_code: return -0.5
    if solved: return 2.0
    return 0.0

# 9. Build the Dataset
import math
from datasets import Dataset
from collections import Counter, defaultdict

DEFAULT_SOLVER_WEIGHT = 2
TARGETED_PROMPT_RATIO = 0.75

def choose_proposer_bug_focus(seed_id: str, operators: list, operator_counts: Counter, focus_counters: Counter, row_index: int, total_rows: int) -> str | None:
    unique_operators = sorted(set(operators), key=lambda op: (operator_counts[op], op))
    if not unique_operators: return None
    if row_index >= math.ceil(total_rows * TARGETED_PROMPT_RATIO): return None
    chosen = min(unique_operators, key=lambda op: (focus_counters[op], operator_counts[op], op))
    focus_counters[chosen] += 1
    return chosen

def build_weighted_proposer_rows(bug_bank, target_proposer_rows: int) -> list:
    if target_proposer_rows <= 0: return []
    operator_counts = Counter(sample.bug_operator for sample in bug_bank.train_samples)
    seed_to_operators = defaultdict(list)
    for sample in bug_bank.train_samples:
        seed_to_operators[sample.seed_id].append(sample.bug_operator)
        
    seed_weights = {seed.seed_id: 2 for seed in SEED_BANK} # Default weight for inline
    rows = []
    focus_counters = Counter()
    ordered_seeds = sorted(SEED_BANK, key=lambda seed: (-seed_weights[seed.seed_id], seed.seed_id))

    for seed in SEED_BANK[:target_proposer_rows]:
        bug_focus = choose_proposer_bug_focus(seed.seed_id, seed_to_operators[seed.seed_id], operator_counts, focus_counters, len(rows), target_proposer_rows)
        prompt_text = sample_proposer_prompt(seed.original_code, bug_focus=bug_focus)
        rows.append({"prompt": [{"role": "user", "content": prompt_text}], "role": "proposer", "seed_id": seed.seed_id, "original_code": seed.original_code, "bug_focus": bug_focus if bug_focus else ""})

    while len(rows) < target_proposer_rows:
        for seed in ordered_seeds:
            extra_weight = max(0, seed_weights[seed.seed_id] - 1)
            for _ in range(extra_weight):
                if len(rows) >= target_proposer_rows: break
                bug_focus = choose_proposer_bug_focus(seed.seed_id, seed_to_operators[seed.seed_id], operator_counts, focus_counters, len(rows), target_proposer_rows)
                prompt_text = sample_proposer_prompt(seed.original_code, bug_focus=bug_focus)
                rows.append({"prompt": [{"role": "user", "content": prompt_text}], "role": "proposer", "seed_id": seed.seed_id, "original_code": seed.original_code, "bug_focus": bug_focus if bug_focus else ""})
            if len(rows) >= target_proposer_rows: break
    return rows

def build_mixed_role_dataset(bug_bank) -> Dataset:
    rows = []
    for bug_sample in bug_bank.train_samples:
        prompt_text = sample_solver_prompt(bug_sample.buggy_code, bug_sample.execution_result)
        rows.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "role": "solver", "seed_id": bug_sample.seed_id, "original_code": bug_sample.original_code, "buggy_code": bug_sample.buggy_code
        })
    target_proposer_rows = max(1, math.ceil(len(rows) / DEFAULT_SOLVER_WEIGHT)) if rows else len(SEED_BANK)
    rows.extend(build_weighted_proposer_rows(bug_bank, target_proposer_rows))
    return Dataset.from_list(rows)

dataset, bug_bank = build_mixed_role_dataset(build_bug_bank()), build_bug_bank()
print("Dataset size:", len(dataset))


# 10. TRL GRPO Training Setup
import torch
import importlib.util
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import re

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct" # Recommended default from DebugZero
DEFAULT_MAX_PROMPT_LENGTH = 768
DEFAULT_MAX_COMPLETION_LENGTH = 256

def extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s(.*?)```", text, flags=re.DOTALL)
    if match: return match.group(1).strip()
    return text.strip()

def completion_to_text(completion) -> str:
    if isinstance(completion, list) and completion:
        item = completion[0]
        return item.get("content", "") if isinstance(item, dict) else str(item)
    return str(completion)

def execute_candidate(seed: SeedSpec, candidate_code: str) -> dict[str, object]:
    result = execute_code(candidate_code, seed.test)
    execution_result = result.output[:500] if result.output else ""
    return {
        "tests_passed": result.passed, "syntax_error": result.syntax_error,
        "unsafe_code": execution_result.startswith("Unsafe import detected."),
        "execution_result": execution_result,
    }

from concurrent.futures import ThreadPoolExecutor

def proposer_reward_fn(prompts, completions, **kwargs):
    roles = kwargs.get("role", [])
    seed_ids = kwargs.get("seed_id", [])
    original_codes = kwargs.get("original_code", [])
    
    def evaluate_single(args):
        i, completion = args
        role = roles[i] if i < len(roles) else roles[0]
        if role != "proposer":
            return 0.0
            
        seed_id = seed_ids[i] if i < len(seed_ids) else seed_ids[0]
        original_code = original_codes[i] if i < len(original_codes) else original_codes[0]
        seed = get_seed_by_id(seed_id)
        candidate_code = extract_python_code(completion_to_text(completion))
        exec_meta = execute_candidate(seed, candidate_code)
        
        unchanged = is_effectively_unchanged(original_code, candidate_code)
        proposer_meta = {
            "seed_id": seed.seed_id, "tests_passed": exec_meta["tests_passed"], "syntax_error": exec_meta["syntax_error"],
            "unsafe_code": exec_meta["unsafe_code"], "unchanged_code": unchanged,
            "changed_but_passing": (not unchanged) and exec_meta["tests_passed"] and (not exec_meta["syntax_error"]),
            "plausibility_score": 0.0 if exec_meta["syntax_error"] else compute_ast_distance(original_code, candidate_code)
        }
        return compute_proposer_reward(proposer_meta)
        
    with ThreadPoolExecutor() as executor:
        return list(executor.map(evaluate_single, enumerate(completions)))

def solver_reward_fn(prompts, completions, **kwargs):
    roles = kwargs.get("role", [])
    seed_ids = kwargs.get("seed_id", [])
    
    def evaluate_single(args):
        i, completion = args
        role = roles[i] if i < len(roles) else roles[0]
        if role != "solver":
            return 0.0
            
        seed_id = seed_ids[i] if i < len(seed_ids) else seed_ids[0]
        seed = get_seed_by_id(seed_id)
        candidate_code = extract_python_code(completion_to_text(completion))
        exec_meta = execute_candidate(seed, candidate_code)
        
        return compute_solver_reward({"seed_id": seed.seed_id, "tests_passed": exec_meta["tests_passed"], "syntax_error": exec_meta["syntax_error"], "unsafe_code": exec_meta["unsafe_code"]})
        
    with ThreadPoolExecutor() as executor:
        return list(executor.map(evaluate_single, enumerate(completions)))

# Load Model
model, tokenizer = None, None
try:
    from unsloth import FastLanguageModel, PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)
    model, tokenizer = FastLanguageModel.from_pretrained(model_name="unsloth/Qwen2.5-Coder-3B-Instruct", max_seq_length=1024, load_in_4bit=False, fast_inference=True)
    model = FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_alpha=16, bias="none", use_gradient_checkpointing="unsloth")
except ImportError:
    # Unsloth is failing to load (e.g., due to Kaggle/Colab CUDA mismatch).
    # Falling back to standard HuggingFace PEFT (LoRA).
    from peft import LoraConfig, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    peft_config = LoraConfig(r=16, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)

if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token


# 11. Run GRPO Training & Plot Metrics
has_bitsandbytes = importlib.util.find_spec("bitsandbytes") is not None

training_args = GRPOConfig(
    output_dir="debugzero_model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    max_steps=200,
    num_generations=8,
    max_prompt_length=DEFAULT_MAX_PROMPT_LENGTH,
    max_completion_length=DEFAULT_MAX_COMPLETION_LENGTH,
    bf16=True, fp16=False,
    logging_steps=1,
    optim="adamw_torch",
    report_to="none",
    remove_unused_columns=False,
    disable_tqdm=False,
)

from transformers import TrainerCallback
from tqdm.auto import tqdm

class TQDMMetricsCallback(TrainerCallback):
    def __init__(self, max_steps):
        self.pbar = tqdm(total=max_steps, desc="GRPO Training", dynamic_ncols=True, leave=True)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs.get("loss", 0.0)
            epoch = logs.get("epoch", 0.0)
            
            p_reward = next((v for k, v in logs.items() if "proposer" in k and "reward" in k and isinstance(v, (int, float))), 0.0)
            s_reward = next((v for k, v in logs.items() if "solver" in k and "reward" in k and isinstance(v, (int, float))), 0.0)
            
            total_reward = logs.get("env/reward_mean", logs.get("reward", p_reward + s_reward))
            
            self.pbar.set_postfix({
                "Epoch": f"{epoch:.2f}",
                "Loss": f"{loss:.4f}",
                "P_Rew": f"{p_reward:.3f}",
                "S_Rew": f"{s_reward:.3f}",
                "Total": f"{total_reward:.3f}"
            })
            self.pbar.update(state.global_step - self.pbar.n)
            
    def on_train_end(self, args, state, control, **kwargs):
        self.pbar.close()

trainer = GRPOTrainer(

    callbacks=[TQDMMetricsCallback(training_args.max_steps)],

    model=model,
    reward_funcs=[proposer_reward_fn, solver_reward_fn],
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print(f"Starting GRPO training for {training_args.max_steps} episodes (steps)...")
print("To change the number of episodes, modify 'max_steps' in GRPOConfig above.")
train_result = trainer.train()
print("Training Complete! View debugzero_model for artifacts.")

# 12. Plot Metrics natively in Colab
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log_history = trainer.state.log_history
steps = [entry["step"] for entry in log_history if "loss" in entry]
losses = [entry["loss"] for entry in log_history if "loss" in entry]

p_rewards = []
s_rewards = []

for entry in log_history:
    if "loss" in entry:
        p_val = next((v for k, v in entry.items() if "proposer" in k and "reward" in k and isinstance(v, (int, float))), 0.0)
        s_val = next((v for k, v in entry.items() if "solver" in k and "reward" in k and isinstance(v, (int, float))), 0.0)
        p_rewards.append(p_val)
        s_rewards.append(s_val)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss Plot
if steps and losses:
    axes[0].plot(steps[:len(losses)], losses, marker='o', color='purple', label="Total Loss")
    axes[0].set_title("GRPO Training Loss")
    axes[0].set_xlabel("Steps (Episodes)")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

# Rewards Plot
if steps and (p_rewards or s_rewards):
    if p_rewards:
        axes[1].plot(steps[:len(p_rewards)], p_rewards, marker='s', color='orange', label="Proposer Reward")
    if s_rewards:
        axes[1].plot(steps[:len(s_rewards)], s_rewards, marker='^', color='green', label="Solver Reward")
        
    axes[1].set_title("GRPO Environment Rewards Evolution")
    axes[1].set_xlabel("Steps (Episodes)")
    axes[1].set_ylabel("Reward")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

plt.tight_layout()
plt.savefig("/data/metrics.png")
print("Saved training metrics plot to metrics.png")

# 13. Interactive Verification
# We wrap tqdm around some final manual checks to give a visual indicator for eval.
from tqdm.auto import tqdm

print("Running final evaluations across the holdout set:")
model.eval()

# Testing Solver
correct = 0
total_evals = len(bug_bank.eval_samples)

print(f"Validating {total_evals} Holdout bugs...")
for sample in tqdm(bug_bank.eval_samples, desc="Solver Eval"):
    prompt = sample_solver_prompt(sample.buggy_code, sample.execution_result)
    
    prompt_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    out = model.generate(**encoded, max_new_tokens=200, pad_token_id=tokenizer.pad_token_id, do_sample=False)
    
    generated_code = tokenizer.decode(out[0][encoded.input_ids.shape[-1]:], skip_special_tokens=True)
    clean_code = extract_python_code(generated_code)
    
    # Check if the generated solution passes the test
    seed = get_seed_by_id(sample.seed_id)
    exec_meta = execute_candidate(seed, clean_code)
    
    if exec_meta["tests_passed"] and not exec_meta["syntax_error"]:
        correct += 1

print(f"Holdout Set Solver Pass Rate: {correct}/{total_evals} ({correct/total_evals:.1%})")

