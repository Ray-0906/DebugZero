import asyncio
import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

from openai import OpenAI
from pydantic import BaseModel

from client import DebugzeroEnv
from models import DebugzeroAction

load_dotenv()


class LLMAction(BaseModel):
    role: str
    code: str


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_URL = os.getenv("DEBUGZERO_ENV_URL", "http://localhost:8000")
TASK_NAME = os.getenv("DEBUGZERO_TASK", "debugzero")
BENCHMARK = os.getenv("DEBUGZERO_BENCHMARK", "debugZero")

NUM_EPISODES = int(os.getenv("NUM_EPISODES", 3))
MAX_STEPS = int(os.getenv("MAX_STEPS", 2))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))

MIN_SCORE = 0.0001
MAX_SCORE = 0.9999


def clamp_score(score: float) -> float:
    return max(MIN_SCORE, min(MAX_SCORE, score))


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0]
    return text.strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    text = strip_code_fences(text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


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


def model_dump(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return {}


def extract_env_error(result: Any) -> Optional[str]:
    for attr in ("last_action_error", "error", "message"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            if val:
                return str(val)

    obs = getattr(result, "observation", None)
    if obs is not None:
        execution_result = getattr(obs, "execution_result", None)
        syntax_error = bool(getattr(obs, "syntax_error", False))
        if isinstance(execution_result, str) and execution_result:
            if syntax_error or execution_result.startswith("Unsafe import detected."):
                return execution_result.replace("\n", "\\n")

    return None


def build_prompt(obs_dict: Dict[str, Any], history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    role_next = obs_dict.get("role_next", "proposer")
    current_code = obs_dict.get("current_code", "")
    execution_result = obs_dict.get("execution_result", "")
    metadata = obs_dict.get("metadata", {})
    original_code = metadata.get("original_code", current_code)

    return textwrap.dedent(
        f"""
        You are playing the DebugZero adversarial debugging environment.

        Current role: {role_next}

        Current observation:
        {json.dumps(obs_dict, indent=2, ensure_ascii=False)}

        Previous actions:
        {history_block}

        Task rules:
        - If role is "proposer", return a full Python function that preserves the original signature but introduces exactly one small realistic logical bug.
        - The proposer bug should keep syntax valid and should make the provided tests fail.
        - If role is "solver", return a full Python function that repairs the current buggy code.
        - The solver fix should pass the tests and preserve the original function signature.
        - Return code only inside the JSON field. Do not include markdown fences.

        Original clean code:
        {original_code}

        Current code:
        {current_code}

        Latest execution result:
        {execution_result}

        Return exactly one valid JSON object with this schema:
        {{
          "role": "{role_next}",
          "code": "<full Python function code>"
        }}
        """
    ).strip()


def get_model_action(client: OpenAI, obs_dict: Dict[str, Any], history: List[str]) -> DebugzeroAction:
    prompt = build_prompt(obs_dict, history)
    role_next = str(obs_dict.get("role_next", "proposer"))
    current_code = str(obs_dict.get("current_code", ""))

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert Python debugging agent."},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    parsed = extract_json_object(content)

    try:
        llm_action = LLMAction(**parsed)
        role = llm_action.role.strip() or role_next
        code = strip_code_fences(llm_action.code)
    except Exception:
        role = role_next
        code = current_code

    if role not in {"proposer", "solver"}:
        role = role_next if role_next in {"proposer", "solver"} else "proposer"

    return DebugzeroAction(role=role, code=code)


async def maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


async def make_env() -> DebugzeroEnv:
    env = DebugzeroEnv(base_url=ENV_URL)
    if hasattr(env, "sync"):
        synced = env.sync()
        return synced if synced is not None else env
    return env


def episode_score(obs: Any, success: bool) -> float:
    if obs is not None and hasattr(obs, "score"):
        try:
            return clamp_score(float(getattr(obs, "score")))
        except (TypeError, ValueError):
            pass
    return MAX_SCORE if success else MIN_SCORE


async def main() -> None:
    if not HF_TOKEN:
        print("[SYSTEM ERROR] HF_TOKEN is not set", file=sys.stderr)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = None

    try:
        env = await make_env()

        for _episode in range(1, NUM_EPISODES + 1):
            history: List[str] = []
            rewards: List[float] = []
            steps_taken = 0
            score = MIN_SCORE
            success = False
            obs = None

            log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

            try:
                reset_result = await maybe_await(env.reset())
                obs = getattr(reset_result, "observation", reset_result)
                done = bool(getattr(reset_result, "done", False))

                for step in range(1, MAX_STEPS + 1):
                    if done:
                        break

                    obs_dict = model_dump(obs)

                    try:
                        action = await asyncio.to_thread(get_model_action, client, obs_dict, history)
                    except Exception as exc:
                        print(f"[SYSTEM ERROR] model_action_failed={exc}", file=sys.stderr)
                        role_next = str(obs_dict.get("role_next", "proposer"))
                        action = DebugzeroAction(role=role_next, code=str(obs_dict.get("current_code", "")))

                    action_str = compact_action_string(action.role, action.code)
                    step_result = await maybe_await(env.step(action))
                    obs = getattr(step_result, "observation", None)
                    done = bool(getattr(step_result, "done", False))
                    reward = float(getattr(step_result, "reward", 0.0) or 0.0)
                    rewards.append(reward)
                    steps_taken = step

                    error = extract_env_error(step_result)
                    log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                    obs_tests_passed = bool(getattr(obs, "tests_passed", False)) if obs is not None else False
                    obs_syntax_error = bool(getattr(obs, "syntax_error", False)) if obs is not None else False
                    history.append(
                        f"Step {step}: {action_str} -> reward {reward:.2f}, "
                        f"tests_passed={str(obs_tests_passed).lower()}, "
                        f"syntax_error={str(obs_syntax_error).lower()}"
                    )

                    if done:
                        success = obs_tests_passed and not obs_syntax_error
                        break

                score = episode_score(obs, success)

            except Exception as exc:
                print(f"[SYSTEM ERROR] {exc}", file=sys.stderr)
                success = False
                score = MIN_SCORE
            finally:
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    except Exception as exc:
        print(f"[SYSTEM ERROR] {exc}", file=sys.stderr)
    finally:
        try:
            if env is not None and hasattr(env, "close"):
                await maybe_await(env.close())
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
        sys.exit(0)
    except Exception as exc:
        print(f"[CRITICAL VALIDATION ERROR] {exc}", file=sys.stderr)
        sys.exit(0)
    except BaseException as base_exc:
        print(f"[BASE EXCEPTION] {base_exc}", file=sys.stderr)
        sys.exit(0)
