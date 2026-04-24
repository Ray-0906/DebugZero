from collections import deque
import statistics

# Global solve rate history buffer: {seed_id: deque(maxlen=20)}
solve_rate_history = {}

def get_solve_rate(seed_id: str) -> float:
    if seed_id not in solve_rate_history or len(solve_rate_history[seed_id]) == 0:
        return 0.5  # default baseline if no history yet
        
    return statistics.mean(solve_rate_history[seed_id])

def record_solve_result(seed_id: str, solved: bool):
    if seed_id not in solve_rate_history:
        solve_rate_history[seed_id] = deque(maxlen=20)
    solve_rate_history[seed_id].append(1.0 if solved else 0.0)

def compute_proposer_reward(meta: dict) -> float:
    # meta requires: tests_passed, syntax_error, plausibility_score, seed_id
    validity = 0.0
    if meta.get("syntax_error", False):
        validity = -1.0
    elif not meta.get("tests_passed", True):
        validity = 1.0  # Successfully broke tests
    else:
        validity = 0.0  # Ran fine, didn't break tests
        
    plausibility = meta.get("plausibility_score", 0.0)
    
    solve_rate = get_solve_rate(meta["seed_id"])
    learnability = 0.0
    if 0.1 <= solve_rate <= 0.9:
        learnability = 1.0
        
    return validity + plausibility + learnability

def compute_solver_reward(meta: dict) -> float:
    # meta requires: tests_passed, syntax_error, seed_id
    solved = meta.get("tests_passed", False) and not meta.get("syntax_error", True)
    
    record_solve_result(meta["seed_id"], solved)
    
    if solved:
        return 1.0
    return 0.0
