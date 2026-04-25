import asyncio
import inspect
import json
import os
import re
import sys
from typing import Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from client import DebugzeroEnv
from models import DebugzeroAction
from server.graders import (
    compute_ast_distance,
    compute_proposer_reward,
    compute_solver_reward,
    is_effectively_unchanged,
    reset_reward_history,
)
from training.dual_role_sampler import sample_proposer_prompt, sample_solver_prompt

load_dotenv()


API_BASE_URL = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL", "meta-llama/llama-3.1-8b-instruct")
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("DEBUGZERO_ENV_URL", "http://localhost:8000")
TASK_NAME = os.getenv("DEBUGZERO_TASK", "debugging-self-play")
BENCHMARK = os.getenv("DEBUGZERO_BENCHMARK", "debugzero")
BUG_FOCUS = os.getenv("DEBUGZERO_BUG_FOCUS")

NUM_EPISODES = int(os.getenv("NUM_EPISODES", "3"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
MAX_PROPOSER_STEPS = int(os.getenv("MAX_PROPOSER_STEPS", str(max(1, MAX_STEPS // 2))))
MAX_SOLVER_STEPS = int(os.getenv("MAX_SOLVER_STEPS", str(max(1, MAX_STEPS - MAX_PROPOSER_STEPS))))
PROPOSER_TEMPERATURE = float(os.getenv("PROPOSER_TEMPERATURE", "0.7"))
SOLVER_TEMPERATURE = float(os.getenv("SOLVER_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))


def extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def compact_action_string(role: str, code: str) -> str:
    obj = {"role": role, "code": code}
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error is not None else "null"
    action_str = action.replace("\n", "\\n")
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


def summarize_error(text: str, max_chars: int = 220) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return "null"
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def extract_env_error(observation: Any) -> Optional[str]:
    execution_result = getattr(observation, "execution_result", "") or ""
    if not execution_result:
        return None
    if getattr(observation, "syntax_error", False):
        return summarize_error(execution_result)
    if execution_result.startswith("Unsafe import detected."):
        return execution_result
    if not getattr(observation, "tests_passed", False):
        return summarize_error(execution_result)
    return None


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


def clamp_score(score: float) -> float:
    return max(0.0001, min(0.9999, float(score)))


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def call_env_method(obj: Any, method_name: str, *args: Any) -> Any:
    method = getattr(obj, method_name)
    result = method(*args)
    return await maybe_await(result)


def get_model_code(
    client: OpenAI,
    *,
    role: str,
    current_code: str,
    execution_result: str,
    feedback: str,
) -> str:
    if role == "proposer":
        prompt = sample_proposer_prompt(current_code, bug_focus=BUG_FOCUS)
        temperature = PROPOSER_TEMPERATURE
    else:
        prompt = sample_solver_prompt(current_code, execution_result, mode="concise")
        temperature = SOLVER_TEMPERATURE

    if feedback:
        prompt = f"{feedback}\n\n{prompt}"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert Python coder."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=MAX_TOKENS,
    )

    return extract_python_code(response.choices[0].message.content or "")


async def make_env() -> Any:
    max_retries = 30

    if LOCAL_IMAGE_NAME:
        for attempt in range(max_retries):
            try:
                env = DebugzeroEnv.from_docker_image(LOCAL_IMAGE_NAME)
                return await maybe_await(env)
            except Exception as exc:
                print(
                    f"[SYSTEM ERROR] Failed to start Docker environment (attempt {attempt + 1}/{max_retries}): {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(5.0)
                else:
                    raise

    for attempt in range(max_retries):
        try:
            return DebugzeroEnv(base_url=ENV_URL)
        except Exception as exc:
            print(
                f"[SYSTEM ERROR] Env connection to {ENV_URL} failed (attempt {attempt + 1}/{max_retries}): {exc}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(5.0)
            else:
                raise


async def main() -> None:
    if not API_KEY:
        print("[SYSTEM ERROR] Missing API key. Set API_KEY, OPENAI_API_KEY, or HF_TOKEN.", file=sys.stderr, flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = None

    try:
        env = await make_env()
        reset_reward_history()

        for _episode in range(1, NUM_EPISODES + 1):
            rewards: List[float] = []
            steps_taken = 0
            proposer_feedback = ""
            solver_feedback = ""
            proposer_succeeded = False
            success = False
            score = 0.0001

            log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

            try:
                result = await call_env_method(env, "reset")
                obs = result.observation
                original_code = obs.metadata.get("original_code", obs.current_code)

                for _proposer_step in range(1, MAX_PROPOSER_STEPS + 1):
                    if steps_taken >= MAX_STEPS:
                        break
                    steps_taken += 1

                    try:
                        proposer_code = await asyncio.to_thread(
                            get_model_code,
                            client,
                            role="proposer",
                            current_code=obs.current_code,
                            execution_result=obs.execution_result,
                            feedback=proposer_feedback,
                        )
                    except Exception as exc:
                        print(f"[SYSTEM ERROR] Proposer generation failed: {exc}", file=sys.stderr, flush=True)
                        proposer_code = obs.current_code

                    action = DebugzeroAction(role="proposer", code=proposer_code)
                    result = await call_env_method(env, "step", action)
                    obs = result.observation

                    proposer_attempt = classify_proposer_attempt(
                        original_code,
                        proposer_code,
                        tests_passed=obs.tests_passed,
                        syntax_error=obs.syntax_error,
                    )

                    reward = compute_proposer_reward(
                        {
                            "seed_id": obs.metadata.get("seed_id", ""),
                            "tests_passed": obs.tests_passed,
                            "syntax_error": obs.syntax_error,
                            "unsafe_code": obs.execution_result.startswith("Unsafe import detected."),
                            "unchanged_code": proposer_attempt["unchanged_code"],
                            "changed_but_passing": proposer_attempt["changed_but_passing"],
                            "plausibility_score": 0.0
                            if obs.syntax_error
                            else compute_ast_distance(original_code, proposer_code),
                        }
                    )
                    rewards.append(reward)

                    log_step(
                        step=steps_taken,
                        action=compact_action_string("proposer", proposer_code),
                        reward=reward,
                        done=bool(getattr(result, "done", False)),
                        error=extract_env_error(obs),
                    )

                    if obs.syntax_error:
                        proposer_feedback = (
                            "Your last attempt caused a syntax error. Keep the code valid and preserve the signature."
                        )
                    elif proposer_attempt["valid_bug"]:
                        proposer_succeeded = True
                        score = 0.5000
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

                if proposer_succeeded:
                    for _solver_step in range(1, MAX_SOLVER_STEPS + 1):
                        if steps_taken >= MAX_STEPS:
                            break
                        steps_taken += 1

                        try:
                            solver_code = await asyncio.to_thread(
                                get_model_code,
                                client,
                                role="solver",
                                current_code=obs.current_code,
                                execution_result=obs.execution_result,
                                feedback=solver_feedback,
                            )
                        except Exception as exc:
                            print(f"[SYSTEM ERROR] Solver generation failed: {exc}", file=sys.stderr, flush=True)
                            solver_code = obs.current_code

                        action = DebugzeroAction(role="solver", code=solver_code)
                        result = await call_env_method(env, "step", action)
                        obs = result.observation

                        reward = compute_solver_reward(
                            {
                                "seed_id": obs.metadata.get("seed_id", ""),
                                "tests_passed": obs.tests_passed,
                                "syntax_error": obs.syntax_error,
                                "unsafe_code": obs.execution_result.startswith("Unsafe import detected."),
                            }
                        )
                        rewards.append(reward)

                        log_step(
                            step=steps_taken,
                            action=compact_action_string("solver", solver_code),
                            reward=reward,
                            done=bool(getattr(result, "done", False)),
                            error=extract_env_error(obs),
                        )

                        if obs.syntax_error:
                            solver_feedback = "The fix caused a syntax error. Return a valid full function."
                        elif obs.tests_passed:
                            success = True
                            score = 0.9999
                            break
                        else:
                            solver_feedback = (
                                "The bug is still present. Focus on the failing behavior in the traceback."
                            )

                log_end(success=success, steps=steps_taken, score=clamp_score(score), rewards=rewards)

            except Exception as exc:
                print(f"[SYSTEM ERROR] {exc}", file=sys.stderr, flush=True)
                log_end(success=False, steps=steps_taken, score=clamp_score(score), rewards=rewards)

    except Exception as exc:
        print(f"[SYSTEM ERROR] {exc}", file=sys.stderr, flush=True)
    finally:
        try:
            if env is not None and hasattr(env, "close"):
                await call_env_method(env, "close")
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
        sys.exit(0)
    except Exception as exc:
        print(f"[CRITICAL VALIDATION ERROR] {exc}", file=sys.stderr, flush=True)
        sys.exit(0)
    except BaseException as base_exc:
        print(f"[BASE EXCEPTION] {base_exc}", file=sys.stderr, flush=True)
        sys.exit(0)
