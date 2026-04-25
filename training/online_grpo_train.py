"""AZR-inspired online self-play GRPO training for DebugZero.

Instead of building the dataset once (offline), this module re-generates
training data every iteration using the *model itself* as proposer —
the core innovation from Absolute Zero Reasoner.

Usage:
    python training/online_grpo_train.py
    python training/online_grpo_train.py --dry_run
"""

from __future__ import annotations

import ast as _ast
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- vLLM V1 workaround for T4 / older GPUs (compute capability < 8.0) ---
try:
    import torch as _torch
    if _torch.cuda.is_available():
        _cc = _torch.cuda.get_device_capability()
        if _cc[0] < 8:
            os.environ.setdefault("VLLM_USE_V1", "0")
            os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
except Exception:
    pass

from datasets import Dataset

# Re-use everything from the original trainer
from training.grpo_train import (
    DEFAULT_MAX_COMPLETION_LENGTH,
    DEFAULT_MAX_PROMPT_LENGTH,
    DEFAULT_MODEL_ID,
    DEFAULT_NUM_GENERATIONS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOLVER_WEIGHT,
    build_mixed_role_dataset,
    create_dataset,
    create_trainer,
    evaluate_proposer_fixed_set,
    evaluate_solver_fixed_set,
    execute_candidate,
    extract_python_code,
    generate_code,
    is_effectively_unchanged,
    load_training_model_and_tokenizer,
    reward_fn,
    save_results_plot,
)

try:
    from server.bug_bank import BugBank, BugSample, build_bug_bank
    from server.seed_bank import SEED_BANK, get_seed_by_id
    from server.rewards import reset_reward_history
    from training.dual_role_sampler import sample_proposer_prompt, sample_solver_prompt
except ImportError:
    from ..server.bug_bank import BugBank, BugSample, build_bug_bank
    from ..server.seed_bank import SEED_BANK, get_seed_by_id
    from ..server.rewards import reset_reward_history
    from .dual_role_sampler import sample_proposer_prompt, sample_solver_prompt


# ---------------------------------------------------------------------------
# Online self-play constants
# ---------------------------------------------------------------------------
DEFAULT_ONLINE_ITERATIONS = 3
DEFAULT_STEPS_PER_ITERATION = 30
DRY_RUN_ONLINE_ITERATIONS = 2
DRY_RUN_STEPS_PER_ITERATION = 2
DEFAULT_CANDIDATES_PER_SEED = 4


# ---------------------------------------------------------------------------
# ModelBugPool — simplified version of AZR's DatasetManager
# ---------------------------------------------------------------------------

class ModelBugPool:
    """Growing pool of model-generated bugs.

    Analogous to AZR's Ray-based DatasetManager, but runs in-process.
    Deduplicates by normalised AST so near-identical bugs aren't repeated.
    """

    def __init__(self) -> None:
        self._samples: list[BugSample] = []
        self._seen_asts: set[str] = set()
        self.stats: dict[str, int] = {
            "attempted": 0,
            "valid": 0,
            "duplicate": 0,
            "syntax_error": 0,
            "still_passing": 0,
            "unchanged": 0,
        }

    @property
    def samples(self) -> list[BugSample]:
        return list(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def try_add(self, sample: BugSample) -> bool:
        """Add a bug if it is novel (by AST). Returns True if accepted."""
        self.stats["attempted"] += 1
        try:
            ast_key = _ast.dump(_ast.parse(sample.buggy_code))
        except SyntaxError:
            ast_key = sample.buggy_code.strip()

        if ast_key in self._seen_asts:
            self.stats["duplicate"] += 1
            return False

        self._seen_asts.add(ast_key)
        self._samples.append(sample)
        self.stats["valid"] += 1
        return True

    def summary(self) -> str:
        return (
            f"Pool size: {len(self)} | "
            f"attempted={self.stats['attempted']} valid={self.stats['valid']} "
            f"dup={self.stats['duplicate']} syntax={self.stats['syntax_error']} "
            f"passing={self.stats['still_passing']} unchanged={self.stats['unchanged']}"
        )


# ---------------------------------------------------------------------------
# Online data generation  (AZR's "gen" phase)
# ---------------------------------------------------------------------------

def generate_bugs_with_model(
    model,
    tokenizer,
    *,
    candidates_per_seed: int = DEFAULT_CANDIDATES_PER_SEED,
    max_new_tokens: int = DEFAULT_MAX_COMPLETION_LENGTH,
) -> list[dict]:
    """Use the current model as proposer to generate candidate bugs.

    Mirrors AZR's generation phase where the model creates its own data.
    """
    candidates: list[dict] = []
    for seed in SEED_BANK:
        prompt = sample_proposer_prompt(seed.original_code)
        for _ in range(candidates_per_seed):
            code = generate_code(
                model, tokenizer, prompt,
                do_sample=True, max_new_tokens=max_new_tokens,
            )
            candidates.append({
                "seed_id": seed.seed_id,
                "original_code": seed.original_code,
                "candidate_code": code,
            })
    return candidates


def validate_and_collect(candidates: list[dict], pool: ModelBugPool) -> int:
    """Validate model outputs and add real test-breaking bugs to the pool."""
    accepted = 0
    for c in candidates:
        seed = get_seed_by_id(c["seed_id"])
        code = c["candidate_code"]

        if is_effectively_unchanged(c["original_code"], code):
            pool.stats["unchanged"] += 1
            continue

        result = execute_candidate(seed, code)
        if result["syntax_error"]:
            pool.stats["syntax_error"] += 1
            continue
        if result["tests_passed"]:
            pool.stats["still_passing"] += 1
            continue

        sample = BugSample(
            seed_id=c["seed_id"],
            original_code=c["original_code"],
            buggy_code=code,
            bug_operator="model_generated",
            execution_result=result["execution_result"][:500],
        )
        if pool.try_add(sample):
            accepted += 1
    return accepted


# ---------------------------------------------------------------------------
# Online dataset builder  (merges bug bank + model pool)
# ---------------------------------------------------------------------------

def build_online_dataset(
    bug_bank: BugBank,
    pool: ModelBugPool,
    solver_weight: int = DEFAULT_SOLVER_WEIGHT,
) -> Dataset:
    """Build mixed-role dataset from bug-bank bugs + model-generated bugs."""
    all_solver_bugs = list(bug_bank.train_samples) + pool.samples
    rows: list[dict] = []

    for bug in all_solver_bugs:
        rows.append({
            "prompt": sample_solver_prompt(
                bug.buggy_code, bug.execution_result, mode="concise",
            ),
            "metadata": {
                "role": "solver",
                "seed_id": bug.seed_id,
                "original_code": bug.original_code,
                "buggy_code": bug.buggy_code,
                "bug_operator": bug.bug_operator,
                "execution_result": bug.execution_result,
            },
        })

    proposer_rows: list[dict] = []
    for seed in SEED_BANK:
        proposer_rows.append({
            "prompt": sample_proposer_prompt(seed.original_code),
            "metadata": {
                "role": "proposer",
                "seed_id": seed.seed_id,
                "original_code": seed.original_code,
            },
        })

    target = max(1, math.ceil(len(rows) / solver_weight)) if rows else len(proposer_rows)
    while len(proposer_rows) < target:
        proposer_rows.extend(proposer_rows[:max(1, target - len(proposer_rows))])
    rows.extend(proposer_rows[:target])

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Main online workflow
# ---------------------------------------------------------------------------

def run_online_workflow(
    dry_run: bool = False,
    iterations: int | None = None,
    steps_per_iter: int | None = None,
) -> dict:
    """AZR-inspired online self-play training loop.

    Each iteration:
      1. GENERATE — model proposes new bugs on all seeds
      2. VALIDATE — keep only real test-breaking bugs, add to pool
      3. REBUILD  — merge bug bank + model-generated pool into dataset
      4. TRAIN    — run GRPO for `steps_per_iter` steps
      5. EVALUATE — measure solver & proposer quality
    """
    if iterations is None:
        iterations = DRY_RUN_ONLINE_ITERATIONS if dry_run else DEFAULT_ONLINE_ITERATIONS
    if steps_per_iter is None:
        steps_per_iter = DRY_RUN_STEPS_PER_ITERATION if dry_run else DEFAULT_STEPS_PER_ITERATION

    # --- Initial setup ---
    dataset, bug_bank = create_dataset()
    model, tokenizer = load_training_model_and_tokenizer(dry_run, dataset, bug_bank)
    pool = ModelBugPool()

    reset_reward_history()
    pre_solver = evaluate_solver_fixed_set(model, tokenizer, bug_bank)
    pre_proposer = evaluate_proposer_fixed_set(model, tokenizer)
    print(f"\n{'='*60}")
    print(f"PRE-TRAINING METRICS")
    print(f"  Solver:   {pre_solver}")
    print(f"  Proposer: {pre_proposer}")
    print(f"{'='*60}\n")

    all_log_history: list[dict] = []
    iter_metrics: list[dict] = []

    for it in range(iterations):
        print(f"\n{'='*60}")
        print(f"ONLINE ITERATION {it + 1}/{iterations}")
        print(f"{'='*60}")

        # --- Phase 1: GENERATE bugs with current model ---
        print(f"  [GEN] Generating candidates with model as proposer...")
        candidates = generate_bugs_with_model(
            model, tokenizer,
            candidates_per_seed=DEFAULT_CANDIDATES_PER_SEED if not dry_run else 2,
        )

        # --- Phase 2: VALIDATE and collect ---
        accepted = validate_and_collect(candidates, pool)
        print(f"  [VAL] Accepted {accepted} new bugs | {pool.summary()}")

        # --- Phase 3: REBUILD dataset ---
        online_ds = build_online_dataset(bug_bank, pool)
        print(f"  [DATA] Dataset size: {len(online_ds)} rows "
              f"(bank={len(bug_bank.train_samples)}, model={len(pool)})")

        # --- Phase 4: TRAIN ---
        reset_reward_history()
        trainer = create_trainer(model, tokenizer, online_ds, dry_run,
                                 max_steps_override=steps_per_iter)
        train_result = trainer.train()
        model = trainer.model  # keep updated weights
        all_log_history.extend(trainer.state.log_history)

        # --- Phase 5: EVALUATE ---
        solver_m = evaluate_solver_fixed_set(model, tokenizer, bug_bank)
        proposer_m = evaluate_proposer_fixed_set(model, tokenizer)
        iter_metrics.append({
            "iteration": it + 1,
            "solver": solver_m,
            "proposer": proposer_m,
            "pool_size": len(pool),
            "dataset_size": len(online_ds),
        })
        print(f"  [EVAL] Solver: {solver_m}")
        print(f"  [EVAL] Proposer: {proposer_m}")

    # --- Final summary ---
    post_solver = iter_metrics[-1]["solver"] if iter_metrics else pre_solver
    post_proposer = iter_metrics[-1]["proposer"] if iter_metrics else pre_proposer

    plot_path = save_results_plot(
        pre_solver, post_solver,
        pre_proposer, post_proposer,
        all_log_history,
    )

    print(f"\n{'='*60}")
    print(f"ONLINE TRAINING COMPLETE")
    print(f"  Iterations: {iterations}")
    print(f"  Model-generated bugs: {len(pool)}")
    print(f"  Final dataset size: {iter_metrics[-1]['dataset_size'] if iter_metrics else 0}")
    print(f"  Pool stats: {pool.summary()}")
    if plot_path:
        print(f"  Plot saved: {plot_path}")
    print(f"{'='*60}\n")

    return {
        "pre_solver_metrics": pre_solver,
        "post_solver_metrics": post_solver,
        "pre_proposer_metrics": pre_proposer,
        "post_proposer_metrics": post_proposer,
        "iteration_metrics": iter_metrics,
        "pool_stats": pool.stats,
        "pool_size": len(pool),
        "plot_path": str(plot_path) if plot_path else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="DebugZero online self-play GRPO training")
    parser.add_argument("--dry_run", action="store_true", help="Quick smoke test")
    parser.add_argument("--iterations", type=int, default=None,
                        help=f"Number of online iterations (default: {DEFAULT_ONLINE_ITERATIONS})")
    parser.add_argument("--steps_per_iter", type=int, default=None,
                        help=f"GRPO steps per iteration (default: {DEFAULT_STEPS_PER_ITERATION})")
    args = parser.parse_args()

    run_online_workflow(
        dry_run=args.dry_run,
        iterations=args.iterations,
        steps_per_iter=args.steps_per_iter,
    )


if __name__ == "__main__":
    main()
