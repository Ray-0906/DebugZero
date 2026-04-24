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
