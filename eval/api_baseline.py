import asyncio
import inspect
import json
import os
import sys
import textwrap
from typing import Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from openai import OpenAI

from client import DebugzeroEnv
from models import DebugzeroAction

load_dotenv()


API_BASE_URL = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL", "meta-llama/llama-3.1-8b-instruct")
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
ENV_URL = os.getenv("DEBUGZERO_ENV_URL", "http://localhost:8000")

NUM_EPISODES = int(os.getenv("NUM_EPISODES", "6"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
PROPOSER_TEMPERATURE = float(os.getenv("PROPOSER_TEMPERATURE", "0.7"))
SOLVER_TEMPERATURE = float(os.getenv("SOLVER_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
BUG_FOCUS = os.getenv("DEBUGZERO_BUG_FOCUS")


def extract_python_code(text: str) -> str:
    content = (text or "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
    if content.endswith("```"):
        content = content.rsplit("\n", 1)[0]
    return content.strip()


def summarize_error(text: str, max_chars: int = 220) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return "null"
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def extract_env_error(result: Any) -> Optional[str]:
    for attr in ("last_action_error", "error", "message"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if value:
                return str(value)

    obs = getattr(result, "observation", None)
    if obs is None:
        return None

    for attr in ("last_action_error", "error"):
        if hasattr(obs, attr):
            value = getattr(obs, attr)
            if value:
                return str(value)

    execution_result = getattr(obs, "execution_result", "")
    if isinstance(execution_result, str) and execution_result:
        if getattr(obs, "syntax_error", False):
            return summarize_error(execution_result)
        if execution_result.startswith("Unsafe import detected."):
            return execution_result
        if not getattr(obs, "tests_passed", False):
            return summarize_error(execution_result)

    return None


def compact_action_string(role: str, code: str) -> str:
    return json.dumps({"role": role, "code": code}, separators=(",", ":"), ensure_ascii=False)


def build_prompt(obs_dict: dict[str, Any], history: list[str]) -> str:
    role = str(obs_dict.get("role_next", "proposer"))
    current_code = str(obs_dict.get("current_code", ""))
    execution_result = str(obs_dict.get("execution_result", ""))
    tests_passed = bool(obs_dict.get("tests_passed", False))
    syntax_error = bool(obs_dict.get("syntax_error", False))
    metadata = obs_dict.get("metadata", {}) or {}
    seed_id = metadata.get("seed_id", "unknown")
    history_block = "\n".join(history[-4:]) if history else "None"

    if role == "proposer":
        focus_line = ""
        if BUG_FOCUS:
            focus_line = f"- Focus specifically on the `{BUG_FOCUS}` mutation family.\n"
        instructions = textwrap.dedent(
            f"""
            You are the Proposer in a debugging self-play environment.
            Return a full Python function with exactly one small logical bug injected.

            Rules:
            - Keep the code valid Python.
            - Keep the same function signature.
            - Preserve the overall structure and formatting as much as possible.
            - Make exactly one small local behavioral change.
            - Avoid comments, explanations, markdown outside the code block, and broad rewrites.
            {focus_line}- Your goal is to make tests fail without creating a syntax error.
            """
        ).strip()
    else:
        instructions = textwrap.dedent(
            """
            You are the Solver in a debugging self-play environment.
            Return the full fixed Python function.

            Rules:
            - Keep the code valid Python.
            - Keep the same function signature.
            - Make the smallest correct local fix you can.
            - Use the failure output to guide the repair.
            - Avoid comments, explanations, markdown outside the code block, and unrelated refactors.
            """
        ).strip()

    return textwrap.dedent(
        f"""
        {instructions}

        Current environment state:
        - seed_id: {seed_id}
        - role_next: {role}
        - tests_passed: {tests_passed}
        - syntax_error: {syntax_error}

        Current code:
        ```python
        {current_code}
        ```

        Execution result:
        {execution_result if execution_result else "None"}

        Previous actions:
        {history_block}

        Return only the full Python code inside triple backticks.
        """
    ).strip()


def get_model_code(client: OpenAI, obs_dict: dict[str, Any], history: list[str]) -> str:
    role = str(obs_dict.get("role_next", "proposer"))
    prompt = build_prompt(obs_dict, history)
    temperature = PROPOSER_TEMPERATURE if role == "proposer" else SOLVER_TEMPERATURE

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


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def call_env_method(obj: Any, method_name: str, *args: Any) -> Any:
    method = getattr(obj, method_name)
    result = method(*args)
    return await maybe_await(result)


async def make_env() -> Any:
    max_retries = 30
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


def print_live_summary(metrics: dict[str, Any]) -> None:
    episodes = max(1, int(metrics["episodes"]))
    proposer_attempts = max(1, int(metrics["proposer_attempts"]))
    solver_attempts = max(1, int(metrics["solver_attempts"]))
    rewards = metrics["rewards"]
    average_reward = (sum(rewards) / len(rewards)) if rewards else 0.0

    print("\n" + "=" * 80)
    print("Live API summary")
    print("=" * 80)
    print(f"Episode success rate:  {metrics['episode_successes'] / episodes:.2%}")
    print(f"Proposer syntax rate:  {metrics['proposer_syntax_errors'] / proposer_attempts:.2%}")
    print(f"Solver syntax rate:    {metrics['solver_syntax_errors'] / solver_attempts:.2%}")
    print(f"Average step reward:   {average_reward:.2f}")
    print(f"Average steps/episode: {metrics['total_steps'] / episodes:.2f}")
    print(f"Representative success: {metrics['representative_success']}")
    print(f"Representative failure: {metrics['representative_failure']}")


async def run_live_api_probe() -> dict[str, Any] | None:
    if not API_KEY:
        print("Skipping live API probe: OPENAI_API_KEY/API_KEY is not set.")
        return None
    if not MODEL_NAME:
        print("Skipping live API probe: OPENAI_MODEL/MODEL_NAME is not set.")
        return None

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env = await make_env()

    metrics = {
        "episodes": NUM_EPISODES,
        "episode_successes": 0,
        "proposer_attempts": 0,
        "solver_attempts": 0,
        "proposer_syntax_errors": 0,
        "solver_syntax_errors": 0,
        "rewards": [],
        "total_steps": 0,
        "representative_success": None,
        "representative_failure": None,
    }

    print("=" * 80)
    print("Live API probe")
    print("=" * 80)
    print(f"API base URL: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Env URL: {ENV_URL}")

    try:
        for episode in range(1, NUM_EPISODES + 1):
            result = await call_env_method(env, "reset")
            obs = getattr(result, "observation", None)
            done = bool(getattr(result, "done", False))
            history: list[str] = []
            success = False

            seed_id = "unknown"
            if obs is not None:
                metadata = getattr(obs, "metadata", {}) or {}
                seed_id = metadata.get("seed_id", "unknown")

            print(f"\nEpisode {episode}/{NUM_EPISODES} | seed={seed_id}")

            for step in range(1, MAX_STEPS + 1):
                if done or obs is None:
                    break

                obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
                role = str(obs_dict.get("role_next", "proposer"))
                if role == "proposer":
                    metrics["proposer_attempts"] += 1
                else:
                    metrics["solver_attempts"] += 1

                try:
                    code = await asyncio.to_thread(get_model_code, client, obs_dict, history)
                except Exception as exc:
                    print(f"[SYSTEM ERROR] Model generation failed: {exc}", file=sys.stderr, flush=True)
                    code = str(obs_dict.get("current_code", ""))

                action = DebugzeroAction(role=role, code=code)
                action_str = compact_action_string(role, code)
                result = await call_env_method(env, "step", action)
                obs = getattr(result, "observation", None)
                done = bool(getattr(result, "done", False))
                reward = float(getattr(result, "reward", 0.0) or 0.0)
                error = extract_env_error(result)

                metrics["rewards"].append(reward)
                metrics["total_steps"] += 1

                if obs is not None and getattr(obs, "syntax_error", False):
                    if role == "proposer":
                        metrics["proposer_syntax_errors"] += 1
                    else:
                        metrics["solver_syntax_errors"] += 1

                print(
                    f"  step={step} role={role} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
                    flush=True,
                )
                history.append(f"Step {step}: {action_str} -> reward {reward:.2f}")

                if done and obs is not None:
                    success = bool(getattr(obs, "tests_passed", False)) and not bool(
                        getattr(obs, "syntax_error", False)
                    )
                    if success:
                        metrics["episode_successes"] += 1
                        if metrics["representative_success"] is None:
                            metrics["representative_success"] = {
                                "seed_id": getattr(obs, "metadata", {}).get("seed_id", "unknown"),
                                "steps": step,
                                "reward": reward,
                            }
                    elif metrics["representative_failure"] is None:
                        metrics["representative_failure"] = {
                            "seed_id": getattr(obs, "metadata", {}).get("seed_id", "unknown"),
                            "steps": step,
                            "execution_result": getattr(obs, "execution_result", ""),
                        }
                    break

            if not success and metrics["representative_failure"] is None:
                failure_seed = seed_id
                failure_output = getattr(obs, "execution_result", "") if obs is not None else ""
                metrics["representative_failure"] = {
                    "seed_id": failure_seed,
                    "steps": min(MAX_STEPS, len(history)),
                    "execution_result": failure_output,
                }

        return metrics
    finally:
        await call_env_method(env, "close")


async def main() -> None:
    metrics = await run_live_api_probe()
    if metrics is not None:
        print_live_summary(metrics)


if __name__ == "__main__":
    asyncio.run(main())
