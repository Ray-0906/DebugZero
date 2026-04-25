import asyncio
import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]

from pydantic import BaseModel, Field

try:
    from debugZero.client import DebugzeroEnv
    from debugZero.models import DebugzeroAction
except ImportError:
    from client import DebugzeroEnv
    from models import DebugzeroAction


load_dotenv()


class LLMAction(BaseModel):
    role: str = Field(..., description="Either proposer or solver")
    code: str = Field(..., description="Complete Python function code")


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
DEBUGZERO_API_URL = os.getenv("DEBUGZERO_API_URL", "https://YOUR-USERNAME-debugzero.hf.space")

NUM_EPISODES = int(os.getenv("NUM_EPISODES", "3"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "2"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))


def build_openai_client() -> Any:
    if not API_KEY:
        return None
    if OpenAI is None:
        print("[SYSTEM ERROR] API key is set but the openai package is not installed.", flush=True)
        return None
    if not API_BASE_URL.startswith(("http://", "https://")):
        print(
            "[SYSTEM ERROR] API_BASE_URL must be an HTTP URL. "
            "It looks like a key/token was placed there, so deterministic mode will run instead.",
            flush=True,
        )
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def model_dump(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return {
        key: getattr(obj, key)
        for key in dir(obj)
        if not key.startswith("_") and not callable(getattr(obj, key))
    }


def unwrap_observation(result: Any) -> Any:
    return getattr(result, "observation", result)


def unwrap_done(result: Any, obs: Any) -> bool:
    return bool(getattr(result, "done", getattr(obs, "done", False)))


def unwrap_reward(result: Any, obs: Any) -> float:
    value = getattr(result, "reward", getattr(obs, "reward", 0.0))
    return float(value or 0.0)


def sanitize_code(text: str) -> str:
    text = (text or "").strip()
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].strip()
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0].strip()
    return text.strip().strip("`")


def compact_action_string(action: DebugzeroAction) -> str:
    return json.dumps(
        {"role": action.role, "code": action.code},
        separators=(",", ":"),
        ensure_ascii=False,
    )


def log_start(episode: int, env_url: str, model: str, mode: str) -> None:
    print(
        f"[START] episode={episode} env=debugZero url={env_url} model={model} mode={mode}",
        flush=True,
    )


def log_step(
    step: int,
    role: str,
    action: str,
    reward: float,
    done: bool,
    tests_passed: Optional[bool],
    syntax_error: Optional[bool],
    execution_result: str,
) -> None:
    action_preview = action.replace("\n", "\\n")[:800]
    result_preview = (execution_result or "").replace("\n", "\\n")[:500]
    print(
        "[STEP] "
        f"step={step} role={role} action={action_preview} reward={reward:.2f} "
        f"done={str(done).lower()} tests_passed={tests_passed} "
        f"syntax_error={syntax_error} execution_result={result_preview}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


def score_transition(role: str, obs: Any, original_code: str, submitted_code: str) -> float:
    tests_passed = bool(getattr(obs, "tests_passed", False))
    syntax_error = bool(getattr(obs, "syntax_error", False))

    if role == "proposer":
        if syntax_error:
            return -1.0
        if submitted_code.strip() == original_code.strip():
            return -0.5
        return 1.0 if not tests_passed else 0.0

    if role == "solver":
        if syntax_error:
            return -1.0
        return 1.0 if tests_passed else 0.0

    return 0.0


def success_from_episode(last_obs: Any, rewards: List[float], proposer_broke_tests: bool) -> bool:
    return (
        proposer_broke_tests
        and bool(getattr(last_obs, "tests_passed", False))
        and not bool(getattr(last_obs, "syntax_error", True))
        and bool(rewards)
        and rewards[-1] > 0
    )


def build_prompt(obs_dict: Dict[str, Any], original_code: str, history: List[str]) -> str:
    role_next = obs_dict.get("role_next", "proposer")
    current_code = obs_dict.get("current_code", "")
    execution_result = obs_dict.get("execution_result", "")
    history_block = "\n".join(history[-4:]) if history else "None"

    if role_next == "proposer":
        objective = textwrap.dedent(
            """
            You are the Proposer in DebugZero.
            Create one realistic bug in the clean function.
            The code must remain syntactically valid Python and should fail tests.
            Prefer small human-like edits such as boundary, comparison, operator, or return-condition mistakes.
            Do not use imports, file IO, eval, exec, open, os, sys, subprocess, shutil, or pathlib.
            """
        ).strip()
    else:
        objective = textwrap.dedent(
            """
            You are the Solver in DebugZero.
            Repair the current buggy function so it passes the hidden environment tests.
            Return the complete corrected function.
            Do not use imports, file IO, eval, exec, open, os, sys, subprocess, shutil, or pathlib.
            """
        ).strip()

    return textwrap.dedent(
        f"""
        You are controlling a live OpenEnv environment named DebugZero.

        Current role: {role_next}

        Objective:
        {objective}

        Original clean code:
        ```python
        {original_code}
        ```

        Current environment observation:
        {json.dumps(obs_dict, indent=2, ensure_ascii=False)}

        Last execution result:
        {execution_result or "None"}

        Previous actions:
        {history_block}

        Return exactly one valid JSON object with this schema:
        {{
          "role": "{role_next}",
          "code": "<complete Python function, no markdown fences>"
        }}
        """
    ).strip()


def parse_model_content(content: str, fallback_role: str, fallback_code: str) -> LLMAction:
    content = (content or "{}").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
    if content.endswith("```"):
        content = content.rsplit("\n", 1)[0]
    content = content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"role": fallback_role, "code": content}

    if not isinstance(parsed, dict):
        parsed = {"role": fallback_role, "code": fallback_code}

    parsed["role"] = str(parsed.get("role") or fallback_role).strip().lower()
    parsed["code"] = sanitize_code(str(parsed.get("code") or fallback_code))
    return LLMAction(**parsed)


def deterministic_bug(clean_code: str) -> str:
    replacements = [
        ("idx != idx2", "idx == idx2"),
        ("distance < threshold", "distance <= threshold"),
        ("return True", "return False"),
        ("return False", "return True"),
    ]
    for old, new in replacements:
        if old in clean_code:
            return clean_code.replace(old, new, 1)
    return clean_code + "\n# no-op fallback mutation\n"


def deterministic_action(obs_dict: Dict[str, Any], original_code: str) -> DebugzeroAction:
    role = str(obs_dict.get("role_next") or "proposer").lower()
    if role == "solver":
        return DebugzeroAction(role="solver", code=original_code)
    return DebugzeroAction(role="proposer", code=deterministic_bug(original_code))


def get_model_action(
    client: Any,
    obs_dict: Dict[str, Any],
    original_code: str,
    history: List[str],
) -> DebugzeroAction:
    role = str(obs_dict.get("role_next") or "proposer").lower()
    fallback = deterministic_action(obs_dict, original_code)

    if client is None:
        return fallback

    prompt = build_prompt(obs_dict, original_code, history)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a careful code-debugging agent. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    llm_action = parse_model_content(content, role, fallback.code)

    if llm_action.role not in {"proposer", "solver"}:
        llm_action.role = role
    if llm_action.role != role and role in {"proposer", "solver"}:
        llm_action.role = role

    return DebugzeroAction(role=llm_action.role, code=sanitize_code(llm_action.code))


async def maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


async def make_env() -> Any:
    if LOCAL_IMAGE_NAME:
        max_retries = 30
        for attempt in range(max_retries):
            try:
                return await maybe_await(DebugzeroEnv.from_docker_image(LOCAL_IMAGE_NAME))
            except Exception as exc:
                print(
                    f"[SYSTEM ERROR] Docker env start failed attempt={attempt + 1}/{max_retries}: {exc}",
                    flush=True,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(5.0)
                else:
                    raise

    max_retries = 30
    for attempt in range(max_retries):
        try:
            env = DebugzeroEnv(base_url=DEBUGZERO_API_URL)
            if hasattr(env, "sync"):
                return env.sync()
            return env
        except Exception as exc:
            print(
                f"[SYSTEM ERROR] Env connection to {DEBUGZERO_API_URL} failed "
                f"attempt={attempt + 1}/{max_retries}: {exc}",
                flush=True,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(5.0)
            else:
                raise


async def reset_env(env: Any) -> Any:
    if hasattr(env, "reset_async"):
        return await env.reset_async()
    return await maybe_await(env.reset())


async def step_env(env: Any, action: DebugzeroAction) -> Any:
    if hasattr(env, "step_async"):
        return await env.step_async(action)
    return await maybe_await(env.step(action))


async def main() -> None:
    client = build_openai_client()
    mode = "llm" if client is not None else "deterministic"

    env = None
    try:
        env = await make_env()

        for episode in range(1, NUM_EPISODES + 1):
            history: List[str] = []
            rewards: List[float] = []
            steps_taken = 0
            last_obs = None
            original_code = ""
            proposer_broke_tests = False

            log_start(episode=episode, env_url=DEBUGZERO_API_URL, model=MODEL_NAME, mode=mode)

            try:
                reset_result = await reset_env(env)
                obs = unwrap_observation(reset_result)
                last_obs = obs
                original_code = str(getattr(obs, "current_code", "") or "")
                done = unwrap_done(reset_result, obs)

                for step in range(1, MAX_STEPS + 1):
                    if done:
                        break

                    obs_dict = model_dump(obs)
                    action = await asyncio.to_thread(get_model_action, client, obs_dict, original_code, history)
                    action_str = compact_action_string(action)

                    result = await step_env(env, action)
                    obs = unwrap_observation(result)
                    last_obs = obs
                    done = unwrap_done(result, obs)

                    env_reward = unwrap_reward(result, obs)
                    shaped_reward = score_transition(action.role, obs, original_code, action.code)
                    reward = shaped_reward if env_reward == 0.0 else env_reward
                    rewards.append(reward)
                    steps_taken = step

                    if action.role == "proposer" and shaped_reward > 0:
                        proposer_broke_tests = True

                    tests_passed = getattr(obs, "tests_passed", None)
                    syntax_error = getattr(obs, "syntax_error", None)
                    execution_result = str(getattr(obs, "execution_result", "") or "")

                    log_step(
                        step=step,
                        role=action.role,
                        action=action_str,
                        reward=reward,
                        done=done,
                        tests_passed=tests_passed,
                        syntax_error=syntax_error,
                        execution_result=execution_result,
                    )

                    history.append(
                        f"Step {step}: role={action.role} reward={reward:.2f} "
                        f"tests_passed={tests_passed} syntax_error={syntax_error}"
                    )

                    if done:
                        break

                success = success_from_episode(last_obs, rewards, proposer_broke_tests)
                score = 1.0 if success else max(0.0, sum(rewards) / max(1, len(rewards)))
                score = max(0.0001, min(0.9999, float(score)))

            except Exception as exc:
                print(f"[SYSTEM ERROR] {exc}", flush=True)
                success = False
                score = 0.0001
            finally:
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    except Exception as exc:
        print(f"[SYSTEM ERROR] {exc}", flush=True)
    finally:
        try:
            if env is not None and hasattr(env, "close"):
                await maybe_await(env.close())
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"[CRITICAL VALIDATION ERROR] {exc}", flush=True)
