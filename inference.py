import asyncio
import inspect
import json
import os
import sys
import textwrap
from typing import Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from client import DebugzeroEnv
from models import DebugzeroAction

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
PROPOSER_TEMPERATURE = float(os.getenv("PROPOSER_TEMPERATURE", "0.7"))
SOLVER_TEMPERATURE = float(os.getenv("SOLVER_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))


def extract_python_code(text: str) -> str:
    content = (text or "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
    if content.endswith("```"):
        content = content.rsplit("\n", 1)[0]
    return content.strip()


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


def build_prompt(obs_dict: dict[str, Any], history: List[str]) -> str:
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
        task_block = textwrap.dedent(
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
        task_block = textwrap.dedent(
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
        {task_block}

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


def get_model_code(client: OpenAI, obs_dict: dict[str, Any], history: List[str]) -> str:
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

        for _episode in range(1, NUM_EPISODES + 1):
            history: List[str] = []
            rewards: List[float] = []
            steps_taken = 0
            score = 0.0
            success = False

            log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

            try:
                result = await call_env_method(env, "reset")
                done = bool(getattr(result, "done", False))
                obs = getattr(result, "observation", None)

                for step in range(1, MAX_STEPS + 1):
                    if done or obs is None:
                        break

                    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
                    role = str(obs_dict.get("role_next", "proposer"))

                    try:
                        code = await asyncio.to_thread(get_model_code, client, obs_dict, history)
                        env_action = DebugzeroAction(role=role, code=code)
                        action_str = compact_action_string(role, code)
                    except Exception as exc:
                        print(f"[SYSTEM ERROR] Model generation failed: {exc}", file=sys.stderr, flush=True)
                        code = obs_dict.get("current_code", "")
                        env_action = DebugzeroAction(role=role, code=code)
                        action_str = compact_action_string(role, code)

                    result = await call_env_method(env, "step", env_action)
                    obs = getattr(result, "observation", None)
                    done = bool(getattr(result, "done", False))
                    reward = float(getattr(result, "reward", 0.0) or 0.0)

                    rewards.append(reward)
                    steps_taken = step

                    error = extract_env_error(result)

                    if obs is not None:
                        score = float(getattr(obs, "score", score) or score)
                        if done:
                            success = bool(getattr(obs, "tests_passed", False)) and not bool(
                                getattr(obs, "syntax_error", False)
                            )

                    score = max(0.0001, min(0.9999, score))
                    log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                    history.append(f"Step {step}: {action_str} -> reward {reward:.2f}")

                score = max(0.0001, min(0.9999, float(score)))

            except Exception as exc:
                print(f"[SYSTEM ERROR] {exc}", file=sys.stderr, flush=True)
                success = False
            finally:
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

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
