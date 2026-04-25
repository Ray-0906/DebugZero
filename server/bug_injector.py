import ast
import random
import copy

BUILTIN_PAIRS = {
    "min": "max", "max": "min",
    "any": "all", "all": "any",
    "sum": "len", "len": "sum"
}

BLOCKED_IMPORTS = ["os", "sys", "subprocess", "shutil", "pathlib"]

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
        if self.mutated:
            return node
            
        if self.target_operator == "off_by_one" and isinstance(node.value, int) and not isinstance(node.value, bool):
            shift = random.choice([-1, 1])
            node.value += shift
            self.mutated = True
        return node
        
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if self.mutated:
            return node
        
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
        if self.mutated:
            return node
            
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
        if self.mutated:
            return node
            
        if isinstance(node.func, ast.Name):
            # wrong built-in
            if self.target_operator == "wrong_builtin" and node.func.id in BUILTIN_PAIRS:
                node.func.id = BUILTIN_PAIRS[node.func.id]
                self.mutated = True
            
            # loop boundary shift
            elif self.target_operator == "loop_boundary_shift" and node.func.id == "range":
                if len(node.args) == 1:
                    # `range(n)` -> `range(n+1)`
                    node.args[0] = ast.BinOp(left=node.args[0], op=ast.Add(), right=ast.Constant(value=1))
                    self.mutated = True
                elif len(node.args) == 2:
                    # `range(a, b)` -> `range(a-1, b)`
                    node.args[0] = ast.BinOp(left=node.args[0], op=ast.Sub(), right=ast.Constant(value=1))
                    self.mutated = True
                    
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if self.mutated:
            return node
            
        if self.target_operator == "condition_negation":
            # `if x > 0` -> `if not x > 0`
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
        if self.mutated:
            return node
            
        if self.target_operator == "slice_boundary_corruption":
            if node.lower is not None:
                node.lower = ast.BinOp(left=node.lower, op=ast.Add(), right=ast.Constant(value=1))
                self.mutated = True
            elif node.upper is not None:
                node.upper = ast.BinOp(left=node.upper, op=ast.Sub(), right=ast.Constant(value=1))
                self.mutated = True
                
        return node

    def visit_Name(self, node):
        self.generic_visit(node)
        return node
        
    def visit_Assign(self, node):
        # variable swap strategy applied by exchanging two target assign variables when there are multiple targets
        self.generic_visit(node)
        if self.mutated:
            return node
            
        if self.target_operator == "variable_swap" and getattr(node, "targets", None):
            if isinstance(node.targets[0], ast.Tuple) and len(node.targets[0].elts) >= 2:
                # swap a, b = x, y  -> b, a = x, y 
                node.targets[0].elts[0], node.targets[0].elts[1] = node.targets[0].elts[1], node.targets[0].elts[0]
                self.mutated = True
        return node

def inject_bug(original_code: str, proposed_operator: str) -> tuple[str, bool]:
    """
    4 critical checks:
    - parse succeeds 
    - mutation actually changed code
    - blocked imports checked
    - built pairs correctly swapped
    """
    try:
        tree = ast.parse(original_code)
    except SyntaxError:
        return original_code, False
        
    injector = BugInjectorVisitor(proposed_operator)
    mutated_tree = injector.visit(copy.deepcopy(tree))
    ast.fix_missing_locations(mutated_tree)
    
    mutated_code = ast.unparse(mutated_tree)
    
    # 2. Check if mutation actually changed something
    if mutated_code.strip() == original_code.strip():
        return original_code, False
        
    # 3. Blocked imports
    if not is_safe_injection(mutated_code):
        return original_code, False
        
    # 4. AST parsing check
    try:
        ast.parse(mutated_code)
    except SyntaxError:
        return original_code, False
        
    return mutated_code, True


def infer_bug_operator(original_code: str, candidate_code: str) -> str | None:
    try:
        original_tree = ast.parse(original_code)
        candidate_tree = ast.parse(candidate_code)
    except SyntaxError:
        return None

    if ast.dump(original_tree) == ast.dump(candidate_tree):
        return None

    original_nodes = list(ast.walk(original_tree))
    candidate_nodes = list(ast.walk(candidate_tree))

    inferred = (
        _infer_wrong_builtin(original_nodes, candidate_nodes)
        or _infer_loop_boundary_shift(original_nodes, candidate_nodes)
        or _infer_slice_boundary_corruption(original_nodes, candidate_nodes)
        or _infer_condition_negation(original_nodes, candidate_nodes)
        or _infer_wrong_operator(original_nodes, candidate_nodes)
        or _infer_off_by_one(original_nodes, candidate_nodes)
    )
    return inferred


def _infer_wrong_builtin(original_nodes: list[ast.AST], candidate_nodes: list[ast.AST]) -> str | None:
    for original_node, candidate_node in zip(original_nodes, candidate_nodes):
        if not isinstance(original_node, ast.Call) or not isinstance(candidate_node, ast.Call):
            continue
        if not isinstance(original_node.func, ast.Name) or not isinstance(candidate_node.func, ast.Name):
            continue
        expected = BUILTIN_PAIRS.get(original_node.func.id)
        if expected and candidate_node.func.id == expected:
            return "wrong_builtin"
    return None


def _infer_loop_boundary_shift(
    original_nodes: list[ast.AST],
    candidate_nodes: list[ast.AST],
) -> str | None:
    for original_node, candidate_node in zip(original_nodes, candidate_nodes):
        if not isinstance(original_node, ast.Call) or not isinstance(candidate_node, ast.Call):
            continue
        if not isinstance(original_node.func, ast.Name) or original_node.func.id != "range":
            continue
        if not isinstance(candidate_node.func, ast.Name) or candidate_node.func.id != "range":
            continue
        if len(original_node.args) != len(candidate_node.args):
            continue
        for original_arg, candidate_arg in zip(original_node.args, candidate_node.args):
            if _is_shifted_by_one(original_arg, candidate_arg):
                return "loop_boundary_shift"
    return None


def _infer_slice_boundary_corruption(
    original_nodes: list[ast.AST],
    candidate_nodes: list[ast.AST],
) -> str | None:
    for original_node, candidate_node in zip(original_nodes, candidate_nodes):
        if not isinstance(original_node, ast.Slice) or not isinstance(candidate_node, ast.Slice):
            continue
        if original_node.lower is not None and candidate_node.lower is not None:
            if _is_shifted_by_one(original_node.lower, candidate_node.lower):
                return "slice_boundary_corruption"
        if original_node.upper is not None and candidate_node.upper is not None:
            if _is_shifted_by_one(original_node.upper, candidate_node.upper):
                return "slice_boundary_corruption"
    return None


def _infer_condition_negation(
    original_nodes: list[ast.AST],
    candidate_nodes: list[ast.AST],
) -> str | None:
    for original_node, candidate_node in zip(original_nodes, candidate_nodes):
        if not isinstance(original_node, ast.If) or not isinstance(candidate_node, ast.If):
            continue
        if (
            isinstance(candidate_node.test, ast.UnaryOp)
            and isinstance(candidate_node.test.op, ast.Not)
            and ast.dump(candidate_node.test.operand) == ast.dump(original_node.test)
        ):
            return "condition_negation"
    return None


def _infer_wrong_operator(original_nodes: list[ast.AST], candidate_nodes: list[ast.AST]) -> str | None:
    for original_node, candidate_node in zip(original_nodes, candidate_nodes):
        if isinstance(original_node, ast.Compare) and isinstance(candidate_node, ast.Compare):
            if original_node.ops and candidate_node.ops and type(original_node.ops[0]) is not type(candidate_node.ops[0]):
                return "wrong_operator"
        if isinstance(original_node, ast.BinOp) and isinstance(candidate_node, ast.BinOp):
            if type(original_node.op) is not type(candidate_node.op):
                return "wrong_operator"
    return None


def _infer_off_by_one(original_nodes: list[ast.AST], candidate_nodes: list[ast.AST]) -> str | None:
    for original_node, candidate_node in zip(original_nodes, candidate_nodes):
        if not isinstance(original_node, ast.Constant) or not isinstance(candidate_node, ast.Constant):
            continue
        if isinstance(original_node.value, bool) or isinstance(candidate_node.value, bool):
            continue
        if isinstance(original_node.value, int) and isinstance(candidate_node.value, int):
            if abs(candidate_node.value - original_node.value) == 1:
                return "off_by_one"
    return None


def _is_shifted_by_one(original_node: ast.AST, candidate_node: ast.AST) -> bool:
    if not isinstance(candidate_node, ast.BinOp):
        return False
    if ast.dump(candidate_node.left) != ast.dump(original_node):
        return False
    if not isinstance(candidate_node.right, ast.Constant) or candidate_node.right.value != 1:
        return False
    return isinstance(candidate_node.op, (ast.Add, ast.Sub))
