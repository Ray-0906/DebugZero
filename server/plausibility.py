import ast
from thefuzz import fuzz

def compute_ast_distance(original_code: str, mutated_code: str) -> float:
    """
    Computes the string similarity distance between the AST dumps of the original
    and mutated code using thefuzz (Levenshtein based).
    Zero edits = 0 score.
    Targeted (small) edit = high plausibility (closer to 1.0).
    Random / wide corruption = low score.
    """
    try:
        orig_ast = ast.dump(ast.parse(original_code))
        mut_ast = ast.dump(ast.parse(mutated_code))
    except SyntaxError:
        return 0.0

    ratio = fuzz.ratio(orig_ast, mut_ast)
    
    # The Fuzz defaults to percentage (0 to 100)
    # Empirical calibration: simple AST mutations typically result in 
    # a fuzz ratio of 85-98%.
    
    if 85 <= ratio:
        return 1.0  # Perfect sweet spot
    elif 50 <= ratio < 85:
        # Linearly decay from 1.0 at 85 down to 0.1 at 50
        return max(0.1, (ratio - 50) / 35.0)
    else:
        return 0.0
