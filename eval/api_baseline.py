import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bug_bank import build_bug_bank
from models import DebugzeroAction
from seed_bank import SEED_BANK, get_seed_by_id
from server.bug_injector import infer_bug_operator
from server.executor import execute_code
from server.plausibility import compute_ast_distance
from training.dual_role_sampler import sample_proposer_prompt, sample_solver_prompt
from training.rewards import (
    compute_proposer_reward,
    compute_solver_reward,
    is_effectively_unchanged,
    reset_reward_history,
)


def extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def execute_candidate(seed_id: str, code: str) -> dict[str, object]:
    seed = get_seed_by_id(seed_id)
    result = execute_code(code, seed.test)
    execution_result = result.output[:500] if result.output else ""
    return {
        "tests_passed": result.passed,
        "syntax_error": result.syntax_error,
        "unsafe_code": execution_result.startswith("Unsafe import detected."),
        "execution_result": execution_result,
    }


def classify_proposer_attempt(
    original_code: str,
    candidate_code: str,
    *,
    tests_passed: bool,
    syntax_error: bool,
) -> dict[str, bool]:
    unchanged_code = is_effectively_unchanged(original_code, candidate_code)
    valid_bug = (not tests_passed) and (not syntax_error)
    changed_but_passing = (not unchanged_code) and tests_passed and (not syntax_error)
    return {
        "unchanged_code": unchanged_code,
        "valid_bug": valid_bug,
        "changed_but_passing": changed_but_passing,
    }


def run_deterministic_controls() -> dict[str, object]:
    bug_bank = build_bug_bank()
    controls = []

    print("=" * 80)
    print("Deterministic controls")
    print("=" * 80)

    for seed in SEED_BANK:
        eval_bug = next(sample for sample in bug_bank.eval_samples if sample.seed_id == seed.seed_id)
        canonical_result = execute_candidate(seed.seed_id, seed.original_code)
        buggy_result = execute_candidate(seed.seed_id, eval_bug.buggy_code)
        syntax_result = execute_candidate(seed.seed_id, "def broken(: pass")

        controls.append(
            {
                "seed_id": seed.seed_id,
                "canonical_passes": canonical_result["tests_passed"] and not canonical_result["syntax_error"],
                "bug_fails": (not buggy_result["tests_passed"]) and not buggy_result["syntax_error"],
                "syntax_detected": syntax_result["syntax_error"],
            }
        )

    canonical_passes = sum(1 for item in controls if item["canonical_passes"])
    bug_failures = sum(1 for item in controls if item["bug_fails"])
    syntax_detected = sum(1 for item in controls if item["syntax_detected"])

    summary = {
        "seed_count": len(SEED_BANK),
        "canonical_pass_count": canonical_passes,
        "verified_bug_fail_count": bug_failures,
        "syntax_detect_count": syntax_detected,
        "controls": controls,
        "bug_bank": bug_bank,
    }

    print(f"Canonical pass count: {canonical_passes}/{len(SEED_BANK)}")
    print(f"Verified bug fail count: {bug_failures}/{len(SEED_BANK)}")
    print(f"Syntax detection count: {syntax_detected}/{len(SEED_BANK)}")
    return summary


async def run_live_api_probe(bug_bank) -> dict[str, object] | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    model_name = os.environ.get("OPENAI_MODEL")
    env_url = os.environ.get("DEBUGZERO_ENV_URL", "http://localhost:8000")

    if not api_key:
        print("Skipping live API probe: OPENAI_API_KEY is not set.")
        return None
    if not model_name:
        print("Skipping live API probe: OPENAI_MODEL is not set.")
        return None

    from openai import OpenAI
    from client import DebugzeroEnv

    client = OpenAI(api_key=api_key, base_url=base_url)
    env = DebugzeroEnv(base_url=env_url)

    reset_reward_history()
    proposer_feedback = ""
    solver_feedback = ""

    metrics = {
        "episodes": len(SEED_BANK),
        "proposer_attempts": 0,
        "solver_attempts": 0,
        "proposer_successes": 0,
        "solver_successes": 0,
        "proposer_step1_successes": 0,
        "proposer_late_successes": 0,
        "proposer_valid_bug_attempts": 0,
        "proposer_unchanged_attempts": 0,
        "proposer_changed_but_passing_attempts": 0,
        "proposer_syntax_errors": 0,
        "solver_syntax_errors": 0,
        "proposer_rewards": [],
        "solver_rewards": [],
        "proposer_bug_family_attempts": {},
        "episode_details": [],
        "representative_success": None,
        "representative_failure": None,
    }

    print("=" * 80)
    print("Live API probe")
    print("=" * 80)
    print(f"API base URL: {base_url}")
    print(f"Model: {model_name}")

    try:
        for episode in range(len(SEED_BANK)):
            result = await env.reset()
            obs = result.observation
            seed_id = obs.metadata.get("seed_id", SEED_BANK[episode].seed_id)
            original_code = obs.metadata.get("original_code", get_seed_by_id(seed_id).original_code)

            print(f"\nEpisode {episode + 1}/{len(SEED_BANK)} | seed={seed_id}")

            proposer_succeeded = False
            for proposer_step in range(1, 5):
                metrics["proposer_attempts"] += 1
                proposer_prompt = sample_proposer_prompt(obs.current_code)
                if proposer_feedback:
                    proposer_prompt = f"{proposer_feedback}\n\n{proposer_prompt}"

                response = client.chat.completions.create(
                    model=model_name,
                    max_tokens=1024,
                    temperature=0.7,
                    messages=[
                        {"role": "system", "content": "You are an expert Python coder."},
                        {"role": "user", "content": proposer_prompt},
                    ],
                )
                proposer_code = extract_python_code(response.choices[0].message.content or "")
                result = await env.step(DebugzeroAction(role="proposer", code=proposer_code))
                obs = result.observation
                proposer_attempt = classify_proposer_attempt(
                    original_code,
                    proposer_code,
                    tests_passed=obs.tests_passed,
                    syntax_error=obs.syntax_error,
                )

                proposer_meta = {
                    "seed_id": seed_id,
                    "tests_passed": obs.tests_passed,
                    "syntax_error": obs.syntax_error,
                    "unsafe_code": obs.execution_result.startswith("Unsafe import detected."),
                    "unchanged_code": proposer_attempt["unchanged_code"],
                    "changed_but_passing": proposer_attempt["changed_but_passing"],
                    "plausibility_score": 0.0
                    if obs.syntax_error
                    else compute_ast_distance(original_code, proposer_code),
                }
                proposer_reward = compute_proposer_reward(proposer_meta)
                metrics["proposer_rewards"].append(proposer_reward)
                likely_bug_family = infer_bug_operator(original_code, proposer_code) or "unknown"
                if proposer_attempt["valid_bug"]:
                    metrics["proposer_valid_bug_attempts"] += 1
                    metrics["proposer_bug_family_attempts"][likely_bug_family] = (
                        metrics["proposer_bug_family_attempts"].get(likely_bug_family, 0) + 1
                    )
                if proposer_attempt["unchanged_code"]:
                    metrics["proposer_unchanged_attempts"] += 1
                if proposer_attempt["changed_but_passing"]:
                    metrics["proposer_changed_but_passing_attempts"] += 1

                if obs.syntax_error:
                    metrics["proposer_syntax_errors"] += 1
                    proposer_feedback = "Your last attempt caused a syntax error. Keep the code valid and preserve the signature."
                elif proposer_attempt["valid_bug"]:
                    proposer_feedback = "You created a valid failing bug. Keep the change small and realistic."
                    proposer_succeeded = True
                    metrics["proposer_successes"] += 1
                    if proposer_step == 1:
                        metrics["proposer_step1_successes"] += 1
                    else:
                        metrics["proposer_late_successes"] += 1
                    metrics["episode_details"].append(
                        {
                            "seed_id": seed_id,
                            "role": "proposer",
                            "step": proposer_step,
                            "likely_bug_family": likely_bug_family,
                            "reward": proposer_reward,
                        }
                    )
                    if metrics["representative_success"] is None:
                        metrics["representative_success"] = {
                            "role": "proposer",
                            "seed_id": seed_id,
                            "reward": proposer_reward,
                            "code": proposer_code,
                        }
                    print(f"  proposer succeeded on step {proposer_step} with reward {proposer_reward:.2f}")
                    break
                elif proposer_attempt["unchanged_code"]:
                    proposer_feedback = (
                        "Your last attempt did not change behavior. Make exactly one small boundary, "
                        "comparison, condition, or slice bug."
                    )
                else:
                    proposer_feedback = (
                        "The tests still passed. Keep exactly one small local edit, but make it "
                        "behavior-changing."
                    )

            if not proposer_succeeded:
                metrics["solver_rewards"].append(0.0)
                if metrics["representative_failure"] is None:
                    metrics["representative_failure"] = {
                        "role": "proposer",
                        "seed_id": seed_id,
                        "reason": "failed_to_break_tests",
                    }
                print("  proposer did not create a failing bug; solver skipped.")
                continue

            for solver_step in range(1, 5):
                metrics["solver_attempts"] += 1
                solver_prompt = sample_solver_prompt(
                    obs.current_code,
                    obs.execution_result,
                    mode="concise",
                )
                if solver_feedback:
                    solver_prompt = f"{solver_feedback}\n\n{solver_prompt}"

                response = client.chat.completions.create(
                    model=model_name,
                    max_tokens=1024,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": "You are an expert Python coder."},
                        {"role": "user", "content": solver_prompt},
                    ],
                )
                solver_code = extract_python_code(response.choices[0].message.content or "")
                result = await env.step(DebugzeroAction(role="solver", code=solver_code))
                obs = result.observation

                solver_meta = {
                    "seed_id": seed_id,
                    "tests_passed": obs.tests_passed,
                    "syntax_error": obs.syntax_error,
                    "unsafe_code": obs.execution_result.startswith("Unsafe import detected."),
                }
                solver_reward = compute_solver_reward(solver_meta)
                metrics["solver_rewards"].append(solver_reward)

                if obs.syntax_error:
                    metrics["solver_syntax_errors"] += 1
                    solver_feedback = "The fix caused a syntax error. Return a valid full function."
                elif obs.tests_passed:
                    solver_feedback = "The fix passed the tests."
                    metrics["solver_successes"] += 1
                    if metrics["representative_success"] is None:
                        metrics["representative_success"] = {
                            "role": "solver",
                            "seed_id": seed_id,
                            "reward": solver_reward,
                            "code": solver_code,
                        }
                    print(f"  solver succeeded on step {solver_step} with reward {solver_reward:.2f}")
                    break
                else:
                    solver_feedback = "The bug is still present. Focus on the failing behavior in the traceback."
                    if metrics["representative_failure"] is None:
                        metrics["representative_failure"] = {
                            "role": "solver",
                            "seed_id": seed_id,
                            "reason": "tests_still_failing",
                            "execution_result": obs.execution_result,
                        }

        return metrics
    finally:
        await env.close()


def print_live_summary(metrics: dict[str, object]) -> None:
    episodes = int(metrics["episodes"]) or 1
    proposer_attempts = int(metrics["proposer_attempts"]) or 1
    solver_attempts = int(metrics["solver_attempts"]) or 1
    proposer_rewards = metrics["proposer_rewards"]
    solver_rewards = metrics["solver_rewards"]

    print("\n" + "=" * 80)
    print("Live API summary")
    print("=" * 80)
    print(f"Proposer success rate: {metrics['proposer_successes'] / episodes:.2%}")
    print(f"Solver success rate:   {metrics['solver_successes'] / episodes:.2%}")
    print(f"Proposer step-1 success rate: {metrics['proposer_step1_successes'] / episodes:.2%}")
    print(f"Proposer late success rate:   {metrics['proposer_late_successes'] / episodes:.2%}")
    print(f"Proposer valid bug rate: {metrics['proposer_valid_bug_attempts'] / proposer_attempts:.2%}")
    print(f"Proposer unchanged rate: {metrics['proposer_unchanged_attempts'] / proposer_attempts:.2%}")
    print(
        f"Proposer changed-pass rate: "
        f"{metrics['proposer_changed_but_passing_attempts'] / proposer_attempts:.2%}"
    )
    print(f"Proposer syntax rate:  {metrics['proposer_syntax_errors'] / proposer_attempts:.2%}")
    print(f"Solver syntax rate:    {metrics['solver_syntax_errors'] / solver_attempts:.2%}")
    print(
        f"Average proposer reward: "
        f"{(sum(proposer_rewards) / len(proposer_rewards)) if proposer_rewards else 0.0:.2f}"
    )
    print(
        f"Average solver reward:   "
        f"{(sum(solver_rewards) / len(solver_rewards)) if solver_rewards else 0.0:.2f}"
    )
    print(f"Proposer bug families:  {metrics['proposer_bug_family_attempts']}")
    print(f"Representative success: {metrics['representative_success']}")
    print(f"Representative failure: {metrics['representative_failure']}")


async def main() -> None:
    control_summary = run_deterministic_controls()
    metrics = await run_live_api_probe(control_summary["bug_bank"])
    if metrics is not None:
        print_live_summary(metrics)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
