"""
DebugZero GRPO training — AZR philosophy, TRL + Unsloth stack.

  python training/grpo_train.py --dry_run
  python training/grpo_train.py --model unsloth/Qwen2.5-Coder-3B-Instruct
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets import Dataset

from server.bug_bank import BugBank, BugSample, build_bug_bank
from server.executor import execute_code
from server.plausibility import compute_ast_distance
from server.rewards import (
    compute_proposer_reward,
    compute_solver_reward,
    is_effectively_unchanged,
    reset_reward_history,
)
from server.seed_bank import SEED_BANK, SeedSpec, get_seed_by_id
from training.dual_role_sampler import sample_proposer_prompt, sample_solver_prompt


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "unsloth/Qwen2.5-Coder-3B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/debugzero_grpo"
DEFAULT_STEPS = 50
DEFAULT_G = 4
DEFAULT_BATCH_SIZE = 2


# ---------------------------------------------------------------------------
# Step 1 — Unsloth model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_name: str, max_seq_len: int = 1280):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=None,          # auto: bf16 on Ampere+, fp16 otherwise
        load_in_4bit=True,   # QLoRA — fits 3B on a single 16 GB GPU
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",   # 30 % less VRAM
        random_state=42,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # required for GRPO generation
    return model, tokenizer


# ---------------------------------------------------------------------------
# Step 2 — Build the mixed-role dataset
# ---------------------------------------------------------------------------
def build_dataset(bug_bank: BugBank, solver_ratio: float = 0.67) -> Dataset:
    """
    AZR philosophy: one unified dataset, two roles.
    Each row carries metadata the reward function needs.
    """
    rows: list[dict] = []

    # --- Solver rows: see buggy code + failure, fix it ---
    for s in bug_bank.train_samples:
        prompt_text = sample_solver_prompt(
            s.buggy_code, s.execution_result, mode="concise",
        )
        rows.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "role": "solver",
            "seed_id": s.seed_id,
            "original_code": s.original_code,
            "buggy_code": s.buggy_code,
            "execution_result": s.execution_result,
        })

    # --- Proposer rows: see clean code, inject a realistic bug ---
    n_prop = max(1, int(len(rows) * (1 - solver_ratio) / solver_ratio))
    for i in range(n_prop):
        seed = SEED_BANK[i % len(SEED_BANK)]
        prompt_text = sample_proposer_prompt(seed.original_code)
        rows.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "role": "proposer",
            "seed_id": seed.seed_id,
            "original_code": seed.original_code,
            "buggy_code": "",
            "execution_result": "",
        })

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Step 3 — Code extraction helper
# ---------------------------------------------------------------------------
def extract_code(text: str) -> str:
    """Pull the first ```python block out, or return raw text."""
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Step 4 — TRR++ (the core AZR trick)
# ---------------------------------------------------------------------------
def trr_plus_plus(rewards: list[float], roles: list[str]) -> list[float]:
    """
    Task-Relative REINFORCE++ (TRR++).

    AZR computes 6 baselines (3 task types × 2 roles).
    DebugZero has 2 roles, so we compute 2 baselines.

    Without this: proposer rewards (0 to ~2.5) and solver rewards
    (binary -0.5/0/1) are on different scales. The proposer's higher
    variance dominates the gradient and kills solver learning.

    With this: each role's rewards are z-scored within that role's
    slice of the batch, giving equal gradient weight to both roles.
    """
    r = np.array(rewards, dtype=np.float32)
    out = np.zeros_like(r)
    for role in ("proposer", "solver"):
        mask = np.array([x == role for x in roles])
        if mask.sum() < 2:
            out[mask] = r[mask]
            continue
        slice_ = r[mask]
        out[mask] = (slice_ - slice_.mean()) / (slice_.std() + 1e-8)
    return out.tolist()


# ---------------------------------------------------------------------------
# Step 5 — Reward function wired to the DebugZero environment
# ---------------------------------------------------------------------------
def make_reward_fn(dataset: Dataset):
    """
    Build the reward function that GRPOTrainer will call.

    TRL signature: fn(prompts, completions, **kwargs) -> list[float]

    We apply TRR++ so that by the time TRL uses these as advantages,
    they're already role-normalized.
    """
    # Fast lookup: prompt content → row metadata
    meta_by_prompt: dict[str, dict] = {}
    for row in dataset:
        # The prompt is a list of dicts; use the user content as key
        key = row["prompt"][0]["content"] if isinstance(row["prompt"], list) else row["prompt"]
        meta_by_prompt[key] = row

    def reward_fn(prompts, completions, **kwargs):
        rewards, roles = [], []

        for prompt, completion in zip(prompts, completions):
            # Resolve prompt key — TRL may pass the formatted string
            if isinstance(prompt, list):
                key = prompt[0]["content"]
            else:
                key = prompt

            meta = meta_by_prompt.get(key, {})
            role = meta.get("role", "solver")

            # Extract code from the model's completion
            if isinstance(completion, list):
                comp_text = completion[0]["content"] if completion else ""
            elif isinstance(completion, dict):
                comp_text = completion.get("content", str(completion))
            else:
                comp_text = str(completion)
            code = extract_code(comp_text)

            seed = get_seed_by_id(meta["seed_id"])
            result = execute_code(code, seed.test)
            unsafe = not __import__("server.executor", fromlist=["is_safe"]).is_safe(code)

            if role == "solver":
                solver_meta = {
                    "seed_id": seed.seed_id,
                    "tests_passed": result.passed,
                    "syntax_error": result.syntax_error,
                    "unsafe_code": unsafe,
                }
                r = compute_solver_reward(solver_meta)
            else:
                proposer_meta = {
                    "seed_id": seed.seed_id,
                    "tests_passed": result.passed,
                    "syntax_error": result.syntax_error,
                    "unsafe_code": unsafe,
                    "unchanged_code": is_effectively_unchanged(
                        meta.get("original_code", ""), code,
                    ),
                    "plausibility_score": 0.0,
                }
                if not result.syntax_error:
                    proposer_meta["plausibility_score"] = compute_ast_distance(
                        meta.get("original_code", ""), code,
                    )
                r = compute_proposer_reward(proposer_meta)

            rewards.append(float(r))
            roles.append(role)

        # Apply TRR++ before returning to TRL
        return trr_plus_plus(rewards, roles)

    return reward_fn


# ---------------------------------------------------------------------------
# Step 6 — Quick holdout eval
# ---------------------------------------------------------------------------
def quick_eval(model, tokenizer, bug_bank: BugBank, n: int = 6):
    import torch
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    device = next(model.parameters()).device
    s_hits, p_hits = [], []

    with torch.no_grad():
        # Solver eval
        for samp in bug_bank.eval_samples[:n]:
            prompt = sample_solver_prompt(
                samp.buggy_code, samp.execution_result, mode="concise",
            )
            messages = [{"role": "user", "content": prompt}]
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            ids = tokenizer(chat_text, return_tensors="pt",
                            truncation=True, max_length=768).to(device)
            out = model.generate(**ids, max_new_tokens=512, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
            code = extract_code(
                tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                 skip_special_tokens=True))
            seed = get_seed_by_id(samp.seed_id)
            result = execute_code(code, seed.test)
            s_hits.append(result.passed)

        # Proposer eval
        for seed in SEED_BANK[:3]:
            prompt = sample_proposer_prompt(seed.original_code)
            messages = [{"role": "user", "content": prompt}]
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            ids = tokenizer(chat_text, return_tensors="pt",
                            truncation=True, max_length=768).to(device)
            out = model.generate(**ids, max_new_tokens=512, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
            code = extract_code(
                tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                 skip_special_tokens=True))
            result = execute_code(code, seed.test)
            valid_bug = (not result.passed) and (not result.syntax_error)
            p_hits.append(valid_bug)

    FastLanguageModel.for_training(model)
    return (
        float(np.mean(s_hits)) if s_hits else 0.0,
        float(np.mean(p_hits)) if p_hits else 0.0,
    )


# ---------------------------------------------------------------------------
# Step 7 — Main training function
# ---------------------------------------------------------------------------
def train(args):
    import torch

    print(f"\n{'=' * 55}")
    print(f"  DebugZero GRPO  |  {args.model}")
    print(f"  Steps: {args.steps}  |  G rollouts: {args.g}")
    print(f"{'=' * 55}\n")

    # ── data ──────────────────────────────────────────────────
    bug_bank = build_bug_bank()
    dataset = build_dataset(bug_bank)
    n_solver = sum(1 for r in dataset["role"] if r == "solver")
    n_proposer = sum(1 for r in dataset["role"] if r == "proposer")
    print(f"Bug bank : {len(bug_bank.train_samples)} train / "
          f"{len(bug_bank.eval_samples)} eval")
    print(f"Dataset  : {len(dataset)} rows ({n_solver} solver / {n_proposer} proposer)")

    # ── model ─────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.model)

    # ── reward fn ─────────────────────────────────────────────
    reward_fn = make_reward_fn(dataset)

    # ── GRPO config ───────────────────────────────────────────
    from trl import GRPOConfig, GRPOTrainer

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.steps,
        num_generations=args.g,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        max_completion_length=512,
        max_prompt_length=768,
        temperature=0.8,
        top_p=0.95,
        learning_rate=1e-6,               # same as AZR paper
        kl_coef=0.01,
        optim="adamw_8bit",
        lr_scheduler_type="constant",      # AZR uses constant LR
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_steps=25,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
    )

    # ── pre-eval ──────────────────────────────────────────────
    print("\n--- Pre-training eval ---")
    reset_reward_history()
    pre_s, pre_p = quick_eval(model, tokenizer, bug_bank)
    print(f"  Solver  fix rate  : {pre_s:.0%}")
    print(f"  Proposer bug rate : {pre_p:.0%}")

    # ── train ─────────────────────────────────────────────────
    print("\n--- Training ---")
    reset_reward_history()
    trainer.train()

    # ── post-eval ─────────────────────────────────────────────
    print("\n--- Post-training eval ---")
    post_s, post_p = quick_eval(trainer.model, tokenizer, bug_bank)
    print(f"  Solver  fix rate  : {post_s:.0%}  (delta {post_s - pre_s:+.0%})")
    print(f"  Proposer bug rate : {post_p:.0%}  (delta {post_p - pre_p:+.0%})")

    # Save merged weights (unsloth specific)
    model.save_pretrained_merged(
        args.output_dir + "/merged",
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"\nSaved to {args.output_dir}/merged")

    return {
        "pre_solver": pre_s,
        "post_solver": post_s,
        "pre_proposer": pre_p,
        "post_proposer": post_p,
    }


# ---------------------------------------------------------------------------
# Step 8 — Dry run + entrypoint
# ---------------------------------------------------------------------------
def dry_run():
    print("\n=== DRY RUN ===")
    bug_bank = build_bug_bank()
    ds = build_dataset(bug_bank)
    print(f"Bug bank : {len(bug_bank.train_samples)} train / "
          f"{len(bug_bank.eval_samples)} eval")
    print(f"Dataset  : {len(ds)} rows")

    # Test TRR++
    rewards = [1.4, 0.0, 1.1, 0.0, 1.0, 0.0, 1.0, 0.0]
    roles = ["proposer", "proposer", "proposer", "proposer",
             "solver", "solver", "solver", "solver"]
    normed = trr_plus_plus(rewards, roles)
    print(f"\nTRR++ proposer in : {rewards[:4]}")
    print(f"TRR++ proposer out: {[f'{x:.2f}' for x in normed[:4]]}")
    print(f"TRR++ solver   in : {rewards[4:]}")
    print(f"TRR++ solver   out: {[f'{x:.2f}' for x in normed[4:]]}")

    # Test reward fn on one real sample
    samp = bug_bank.train_samples[0]
    seed = get_seed_by_id(samp.seed_id)
    result = execute_code(seed.original_code, seed.test)
    r = compute_solver_reward({
        "seed_id": seed.seed_id,
        "tests_passed": result.passed,
        "syntax_error": result.syntax_error,
        "unsafe_code": False,
    })
    print(f"\nSolver reward (canonical as fix) : {r}  (expect 1.0)")
    assert r == 1.0, f"Expected 1.0, got {r}"
    print("\n=== DRY RUN PASSED ===")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DebugZero GRPO training")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--g", type=int, default=DEFAULT_G,
                   help="num_generations (G rollouts per prompt, like AZR)")
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = p.parse_args()

    if args.dry_run:
        dry_run()
    else:
        train(args)
