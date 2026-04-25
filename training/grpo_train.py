from __future__ import annotations

import importlib.util
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- vLLM V1 workaround for T4 / older GPUs (compute capability < 8.0) ---
# The V1 engine uses torch.compile which crashes on T4 with:
#   "RuntimeError: Tried to erase Node size_3 but it still had 1 users in the graph"
# Disabling V1 and using eager mode fixes this.
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

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported

    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

    def is_bfloat16_supported() -> bool:
        return False


try:
    from unsloth import PatchFastRL

    PatchFastRL("GRPO", FastLanguageModel)
except ImportError:
    pass

try:
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
except ImportError:
    from ..server.bug_bank import BugBank, BugSample, build_bug_bank
    from ..server.executor import execute_code
    from ..server.plausibility import compute_ast_distance
    from ..server.rewards import (
        compute_proposer_reward,
        compute_solver_reward,
        is_effectively_unchanged,
        reset_reward_history,
    )
    from ..server.seed_bank import SEED_BANK, SeedSpec, get_seed_by_id
    from .dual_role_sampler import sample_proposer_prompt, sample_solver_prompt


DEFAULT_MODEL_ID = "unsloth/Qwen2.5-Coder-3B-Instruct"
DEFAULT_FALLBACK_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
DEFAULT_OUTPUT_DIR = Path("debugzero_model")
DEFAULT_SOLVER_WEIGHT = 2
DEFAULT_NUM_GENERATIONS = 4
DEFAULT_MAX_STEPS = 80
DEFAULT_MAX_PROMPT_LENGTH = 768
DEFAULT_MAX_COMPLETION_LENGTH = 256
DRY_RUN_MAX_STEPS = 2


def extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def completion_to_text(completion) -> str:
    if isinstance(completion, list) and completion:
        item = completion[0]
        if isinstance(item, dict):
            return item.get("content", "")
        return str(item)
    return str(completion)


def execute_candidate(seed: SeedSpec, candidate_code: str) -> dict[str, object]:
    result = execute_code(candidate_code, seed.test)
    execution_result = result.output[:500] if result.output else ""
    unsafe_code = execution_result.startswith("Unsafe import detected.")
    return {
        "tests_passed": result.passed,
        "syntax_error": result.syntax_error,
        "unsafe_code": unsafe_code,
        "execution_result": execution_result,
    }


def build_mixed_role_dataset(
    bug_bank: BugBank,
    solver_weight: int = DEFAULT_SOLVER_WEIGHT,
) -> Dataset:
    rows: list[dict[str, object]] = []

    for bug_sample in bug_bank.train_samples:
        rows.append(
            {
                "prompt": sample_solver_prompt(
                    bug_sample.buggy_code,
                    bug_sample.execution_result,
                    mode="concise",
                ),
                "metadata": {
                    "role": "solver",
                    "seed_id": bug_sample.seed_id,
                    "original_code": bug_sample.original_code,
                    "buggy_code": bug_sample.buggy_code,
                    "bug_operator": bug_sample.bug_operator,
                    "execution_result": bug_sample.execution_result,
                },
            }
        )

    proposer_rows: list[dict[str, object]] = []
    for seed in SEED_BANK:
        proposer_rows.append(
            {
                "prompt": sample_proposer_prompt(seed.original_code),
                "metadata": {
                    "role": "proposer",
                    "seed_id": seed.seed_id,
                    "original_code": seed.original_code,
                },
            }
        )

    target_proposer_rows = max(1, math.ceil(len(rows) / solver_weight)) if rows else len(proposer_rows)
    while len(proposer_rows) < target_proposer_rows:
        proposer_rows.extend(proposer_rows[: max(1, target_proposer_rows - len(proposer_rows))])

    rows.extend(proposer_rows[:target_proposer_rows])
    return Dataset.from_list(rows)


def create_dataset() -> tuple[Dataset, BugBank]:
    bug_bank = build_bug_bank()
    return build_mixed_role_dataset(bug_bank), bug_bank


def reward_fn(prompts, completions, **kwargs):
    rewards: list[float] = []
    metadata = kwargs.get("metadata", [])

    for completion, meta in zip(completions, metadata):
        seed = get_seed_by_id(meta["seed_id"])
        candidate_code = extract_python_code(completion_to_text(completion))
        execution_meta = execute_candidate(seed, candidate_code)

        if meta["role"] == "proposer":
            proposer_meta = {
                "seed_id": seed.seed_id,
                "tests_passed": execution_meta["tests_passed"],
                "syntax_error": execution_meta["syntax_error"],
                "unsafe_code": execution_meta["unsafe_code"],
                "unchanged_code": is_effectively_unchanged(
                    meta["original_code"],
                    candidate_code,
                ),
                "plausibility_score": 0.0,
            }
            if not execution_meta["syntax_error"]:
                proposer_meta["plausibility_score"] = compute_ast_distance(
                    meta["original_code"],
                    candidate_code,
                )
            rewards.append(compute_proposer_reward(proposer_meta))
            continue

        solver_meta = {
            "seed_id": seed.seed_id,
            "tests_passed": execution_meta["tests_passed"],
            "syntax_error": execution_meta["syntax_error"],
            "unsafe_code": execution_meta["unsafe_code"],
        }
        rewards.append(compute_solver_reward(solver_meta))

    return rewards


def evaluate_bug_sample(candidate_code: str, bug_sample: BugSample) -> dict[str, object]:
    seed = get_seed_by_id(bug_sample.seed_id)
    evaluation = execute_candidate(seed, candidate_code)
    reward = compute_solver_reward(
        {
            "seed_id": bug_sample.seed_id,
            "tests_passed": evaluation["tests_passed"],
            "syntax_error": evaluation["syntax_error"],
            "unsafe_code": evaluation["unsafe_code"],
        }
    )
    return {**evaluation, "reward": reward}


def evaluate_solver_fixed_set(model, tokenizer, bug_bank: BugBank) -> dict[str, float]:
    results = []
    for bug_sample in bug_bank.eval_samples:
        prompt = sample_solver_prompt(
            bug_sample.buggy_code,
            bug_sample.execution_result,
            mode="concise",
        )
        candidate_code = generate_code(model, tokenizer, prompt, do_sample=False)
        results.append(evaluate_bug_sample(candidate_code, bug_sample))
    return summarize_solver_results(results)


def evaluate_proposer_fixed_set(model, tokenizer) -> dict[str, float]:
    results = []
    for seed in SEED_BANK:
        prompt = sample_proposer_prompt(seed.original_code)
        candidate_code = generate_code(model, tokenizer, prompt, do_sample=False)
        evaluation = execute_candidate(seed, candidate_code)
        unchanged_code = is_effectively_unchanged(seed.original_code, candidate_code)
        valid_bug = (not evaluation["tests_passed"]) and (not evaluation["syntax_error"])
        changed_but_passing = (
            (not unchanged_code)
            and evaluation["tests_passed"]
            and (not evaluation["syntax_error"])
        )
        reward = compute_proposer_reward(
            {
                "seed_id": seed.seed_id,
                "tests_passed": evaluation["tests_passed"],
                "syntax_error": evaluation["syntax_error"],
                "unsafe_code": evaluation["unsafe_code"],
                "unchanged_code": unchanged_code,
                "plausibility_score": 0.0
                if evaluation["syntax_error"]
                else compute_ast_distance(seed.original_code, candidate_code),
            }
        )
        results.append(
            {
                **evaluation,
                "reward": reward,
                "unchanged_code": unchanged_code,
                "valid_bug": valid_bug,
                "changed_but_passing": changed_but_passing,
            }
        )
    return summarize_proposer_results(results)


def summarize_solver_results(results: list[dict[str, object]]) -> dict[str, float]:
    total = len(results) or 1
    passed = sum(1 for result in results if result["tests_passed"])
    syntax_errors = sum(1 for result in results if result["syntax_error"])
    mean_reward = sum(float(result["reward"]) for result in results) / total
    return {
        "pass_rate": passed / total,
        "syntax_error_rate": syntax_errors / total,
        "mean_reward": mean_reward,
    }


def summarize_proposer_results(results: list[dict[str, object]]) -> dict[str, float]:
    total = len(results) or 1
    bug_rate = sum(
        1 for result in results if (not result["tests_passed"]) and (not result["syntax_error"])
    )
    unchanged = sum(1 for result in results if result.get("unchanged_code"))
    changed_but_passing = sum(1 for result in results if result.get("changed_but_passing"))
    syntax_errors = sum(1 for result in results if result["syntax_error"])
    mean_reward = sum(float(result["reward"]) for result in results) / total
    return {
        "break_rate": bug_rate / total,
        "valid_bug_rate": bug_rate / total,
        "unchanged_rate": unchanged / total,
        "changed_but_passing_rate": changed_but_passing / total,
        "syntax_error_rate": syntax_errors / total,
        "mean_reward": mean_reward,
    }


def generate_code(
    model,
    tokenizer,
    prompt: str,
    *,
    do_sample: bool,
    max_new_tokens: int = DEFAULT_MAX_COMPLETION_LENGTH,
) -> str:
    import torch

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=DEFAULT_MAX_PROMPT_LENGTH)
    model_device = next(model.parameters()).device
    encoded = {key: value.to(model_device) for key, value in encoded.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = 0.7
        generation_kwargs["top_p"] = 0.95

    with torch.no_grad():
        output = model.generate(**encoded, **generation_kwargs)

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    completion = decoded[len(prompt) :] if decoded.startswith(prompt) else decoded
    return extract_python_code(completion)


def get_training_profile(dry_run: bool) -> dict[str, int | float | bool | str]:
    has_bitsandbytes = importlib.util.find_spec("bitsandbytes") is not None
    return {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-5,
        "max_steps": DRY_RUN_MAX_STEPS if dry_run else DEFAULT_MAX_STEPS,
        "num_generations": 2 if dry_run else DEFAULT_NUM_GENERATIONS,
        "max_completion_length": DEFAULT_MAX_COMPLETION_LENGTH,
        "report_to": "none",
        "optim": "adamw_torch" if dry_run or not has_bitsandbytes else "adamw_8bit",
    }


def load_training_model_and_tokenizer(
    dry_run: bool,
    dataset: Dataset,
    bug_bank: BugBank,
):
    if dry_run:
        return build_tiny_local_model_and_tokenizer(dataset, bug_bank)

    if HAS_UNSLOTH:
        print("Initializing Unsloth FastLanguageModel...")
        # T4 GPUs (compute capability < 8.0) crash with vLLM's torch.compile,
        # so we disable fast_inference (vLLM) and fall back to HF generate.
        import torch as _t
        use_fast = True
        if _t.cuda.is_available() and _t.cuda.get_device_capability()[0] < 8:
            print("T4 GPU detected (CC<8) — disabling vLLM fast_inference to avoid torch.compile crash")
            use_fast = False
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=DEFAULT_MODEL_ID,
            max_seq_length=DEFAULT_MAX_PROMPT_LENGTH + DEFAULT_MAX_COMPLETION_LENGTH,
            load_in_4bit=True,
            fast_inference=use_fast,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        return model, tokenizer

    print("Unsloth not available. Falling back to standard Transformers loading.")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_FALLBACK_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(DEFAULT_FALLBACK_MODEL_ID)
    return model, tokenizer


def build_tiny_local_model_and_tokenizer(dataset: Dataset, bug_bank: BugBank):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    corpus = [row["prompt"] for row in dataset]
    corpus.extend(sample.original_code for sample in bug_bank.train_samples)
    corpus.extend(sample.buggy_code for sample in bug_bank.train_samples)
    corpus.extend(sample.original_code for sample in bug_bank.eval_samples)
    corpus.extend(sample.buggy_code for sample in bug_bank.eval_samples)
    corpus.extend(seed.test for seed in SEED_BANK)

    tokenizer_object = Tokenizer(WordLevel(unk_token="<unk>"))
    tokenizer_object.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        min_frequency=1,
    )
    tokenizer_object.train_from_iterator(corpus, trainer=trainer)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_object,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )

    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=DEFAULT_MAX_PROMPT_LENGTH + DEFAULT_MAX_COMPLETION_LENGTH,
        n_ctx=DEFAULT_MAX_PROMPT_LENGTH + DEFAULT_MAX_COMPLETION_LENGTH,
        n_embd=128,
        n_layer=2,
        n_head=2,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(model_config)
    return model, tokenizer


def get_trl_classes():
    if os.name == "nt" and not sys.flags.utf8_mode:
        print("Windows detected. Use `python -X utf8` when running this file locally.")
    from trl import GRPOConfig, GRPOTrainer

    return GRPOConfig, GRPOTrainer


def create_trainer(model, tokenizer, dataset: Dataset, dry_run: bool, max_steps_override: int | None = None):
    GRPOConfig, GRPOTrainer = get_trl_classes()
    profile = get_training_profile(dry_run)

    max_steps = max_steps_override if max_steps_override is not None else profile["max_steps"]

    training_args = GRPOConfig(
        output_dir=str(DEFAULT_OUTPUT_DIR),
        per_device_train_batch_size=profile["per_device_train_batch_size"],
        gradient_accumulation_steps=profile["gradient_accumulation_steps"],
        learning_rate=profile["learning_rate"],
        max_steps=max_steps,
        num_generations=profile["num_generations"],
        max_completion_length=profile["max_completion_length"],
        bf16=(not dry_run) and HAS_UNSLOTH and is_bfloat16_supported(),
        fp16=(not dry_run) and not is_bfloat16_supported(),
        use_cpu=dry_run,
        logging_steps=1 if dry_run else 5,
        optim=profile["optim"],
        report_to=profile["report_to"],
    )

    return GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )


def save_results_plot(
    pre_solver_metrics: dict[str, float],
    post_solver_metrics: dict[str, float],
    pre_proposer_metrics: dict[str, float],
    post_proposer_metrics: dict[str, float],
    log_history: list[dict[str, float]],
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skipping plot generation.")
        return None

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = DEFAULT_OUTPUT_DIR / "debugzero_results.png"

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(
        ["solver pre", "solver post", "proposer pre", "proposer post"],
        [
            pre_solver_metrics["pass_rate"],
            post_solver_metrics["pass_rate"],
            pre_proposer_metrics["break_rate"],
            post_proposer_metrics["break_rate"],
        ],
        color=["#4f81bd", "#4f81bd", "#c0504d", "#c0504d"],
    )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Fixed Eval Rates")
    axes[0].set_ylabel("Rate")

    steps = [entry["step"] for entry in log_history if "step" in entry]
    losses = [entry["loss"] for entry in log_history if "loss" in entry]
    if steps and losses:
        axes[1].plot(steps[: len(losses)], losses, marker="o")
        axes[1].set_title("Training Loss")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Loss")
    else:
        axes[1].bar(
            ["solver reward pre", "solver reward post"],
            [
                pre_solver_metrics["mean_reward"],
                post_solver_metrics["mean_reward"],
            ],
            color=["#9bbb59", "#9bbb59"],
        )
        axes[1].set_title("Solver Mean Reward")

    figure.tight_layout()
    figure.savefig(plot_path)
    plt.close(figure)
    return plot_path


def run_workflow(dry_run: bool = False) -> dict[str, object]:
    dataset, bug_bank = create_dataset()
    print(
        f"Built dataset with {len(dataset)} rows from "
        f"{len(bug_bank.train_samples)} training bugs and {len(bug_bank.eval_samples)} eval bugs."
    )

    model, tokenizer = load_training_model_and_tokenizer(dry_run, dataset, bug_bank)
    trainer = create_trainer(model, tokenizer, dataset, dry_run)

    reset_reward_history()
    pre_solver_metrics = evaluate_solver_fixed_set(model, tokenizer, bug_bank)
    pre_proposer_metrics = evaluate_proposer_fixed_set(model, tokenizer)

    print("Pre-training solver metrics:", pre_solver_metrics)
    print("Pre-training proposer metrics:", pre_proposer_metrics)

    reset_reward_history()
    train_result = trainer.train()

    post_solver_metrics = evaluate_solver_fixed_set(trainer.model, tokenizer, bug_bank)
    post_proposer_metrics = evaluate_proposer_fixed_set(trainer.model, tokenizer)

    plot_path = save_results_plot(
        pre_solver_metrics,
        post_solver_metrics,
        pre_proposer_metrics,
        post_proposer_metrics,
        trainer.state.log_history,
    )

    results = {
        "train_result": train_result,
        "pre_solver_metrics": pre_solver_metrics,
        "post_solver_metrics": post_solver_metrics,
        "pre_proposer_metrics": pre_proposer_metrics,
        "post_proposer_metrics": post_proposer_metrics,
        "plot_path": str(plot_path) if plot_path else None,
        "dataset_size": len(dataset),
        "train_bug_count": len(bug_bank.train_samples),
        "eval_bug_count": len(bug_bank.eval_samples),
    }

    print("Post-training solver metrics:", post_solver_metrics)
    print("Post-training proposer metrics:", post_proposer_metrics)
    if plot_path:
        print(f"Saved plot to {plot_path}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Run a tiny local GRPO smoke test.")
    args = parser.parse_args()

    run_workflow(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
