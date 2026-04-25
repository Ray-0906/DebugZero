---
title: DebugZero
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# DebugZero Environment

DebugZero is an OpenEnv environment for training code models through adversarial debugging self-play.

One model plays two roles:

1. **Proposer**: receives a clean Python function and submits a realistic buggy version.
2. **Solver**: receives the buggy function and submits a repaired version.

The environment executes the submitted code against tests in a constrained Python sandbox and returns structured OpenEnv observations. The training pipeline turns those observations into scalar rewards for GRPO/Unsloth training.

The goal is to teach an LLM a debugging skill that static supervised examples do not capture well: generating plausible failures, diagnosing them, and repairing code based on executable feedback.

## Submission Links

- **Hugging Face Space**: add the final submitted Space URL here before the deadline.
- **Training notebook**: [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb)
- **OpenEnv manifest**: [`openenv.yaml`](openenv.yaml)

## Why This Environment Matters

Most code training data shows finished solutions. DebugZero instead creates a loop where the model has to reason about failure:

- What kind of bug would a real programmer accidentally introduce?
- Does the mutated program still parse and run?
- Does it fail tests for a meaningful reason?
- Can the solver recover the original intended behavior?

That makes the environment useful for training debugging, program repair, adversarial test thinking, and execution-grounded code reasoning.

## OpenEnv Integration

DebugZero uses the standard OpenEnv client/server pattern.

The manifest is:

```yaml
spec_version: 1
name: debugZero
type: space
runtime: fastapi
app: server.app:app
port: 8000
workers: 4
max_concurrent_envs: 100
```

The FastAPI app is created with OpenEnv's server helper in [`server/app.py`](server/app.py):

```python
app = create_app(
    DebugzeroEnvironment,
    DebugzeroAction,
    DebugzeroObservation,
    env_name="debugZero",
    max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", "100")),
)
```

Clients should interact with the environment through [`DebugzeroEnv`](client.py), not by importing server internals. The client serializes `DebugzeroAction` objects, parses OpenEnv `StepResult` payloads, and exposes the normal `reset`, `step`, and `state` flow.

## Episode Flow

Each episode is a two-turn game over one seed function.

### 1. Reset

`reset()` creates a fresh `DebugzeroState`:

- `episode_id`: new UUID
- `step_count`: `0`
- `seed_id`: currently `HumanEval/0`
- `original_code`: clean HumanEval seed implementation
- `current_code`: initially the same clean code
- `role_turn`: `proposer`

The reset observation tells the agent that the proposer acts first and provides the clean function.

### 2. Proposer Step

The proposer sends:

```json
{
  "role": "proposer",
  "code": "<complete mutated Python function>"
}
```

The environment:

1. Stores the submitted code as `current_code`.
2. Runs it with the seed tests using `execute_code`.
3. Returns an observation with:
   - `role_next = "solver"`
   - `tests_passed`
   - `syntax_error`
   - truncated `execution_result`
   - `done = false`

A good proposer submission is syntax-valid, safe to execute, close to the original code, and causes tests to fail.

### 3. Solver Step

The solver sends:

```json
{
  "role": "solver",
  "code": "<complete repaired Python function>"
}
```

The environment:

1. Stores the submitted repair as `current_code`.
2. Runs it against the same tests.
3. Returns an observation with:
   - `role_next = "end"`
   - `tests_passed`
   - `syntax_error`
   - truncated `execution_result`
   - `done = true`

A good solver submission passes tests without syntax errors.

## Action, Observation, and State Schemas

The OpenEnv models live in [`models.py`](models.py).

### Action

`DebugzeroAction` extends OpenEnv `Action`:

| Field | Type | Meaning |
| --- | --- | --- |
| `role` | `str` | Either `proposer` or `solver`. |
| `code` | `str` | The complete buggy or repaired Python function. |

### Observation

`DebugzeroObservation` extends OpenEnv `Observation`:

| Field | Type | Meaning |
| --- | --- | --- |
| `role_next` | `str` | Which role should act next. |
| `current_code` | `str` | Current code after reset or step. |
| `execution_result` | `str` | Captured stdout/stderr summary from sandbox execution. |
| `tests_passed` | `bool` | Whether the submitted code passed the environment tests. |
| `syntax_error` | `bool` | Whether parsing or execution produced a syntax error. |
| `done` | `bool` | OpenEnv completion flag. |
| `reward` | `float` | Server currently returns `0.0`; training code computes shaped rewards externally. |

### State

`DebugzeroState` extends OpenEnv `State`:

| Field | Type | Meaning |
| --- | --- | --- |
| `seed_id` | `str` | Identifier for the seed task. |
| `original_code` | `str` | Clean reference code. |
| `current_code` | `str` | Latest proposer or solver code. |
| `role_turn` | `str` | Internal turn marker: `proposer`, `solver`, or `end`. |

## Reward and Grading Logic

DebugZero separates **verification** from **reward shaping**.

- The OpenEnv server is the verifier. It runs submitted code and returns observations.
- The training layer is the grader. It reads `tests_passed`, `syntax_error`, plausibility, and solve history, then computes scalar rewards.

This is intentional: the same environment can support different reward rubrics without changing the OpenEnv API.

### Server Verifier

[`DebugzeroEnvironment.step`](server/debugZero_environment.py) always executes code and reports the result, but currently returns `reward=0.0` in the observation. The meaningful reward is computed by the training code from the observation fields.

For proposer actions:

- Syntax error: bad mutation.
- Tests still pass: mutation did not create a useful bug.
- Tests fail without syntax error: likely useful bug.

For solver actions:

- Tests pass without syntax error: solved.
- Tests fail or syntax error: not solved.

### Proposer Reward

Implemented in [`training/rewards.py`](training/rewards.py):

```python
reward = validity + plausibility + learnability
```

Components:

| Component | Logic | Reason |
| --- | --- | --- |
| `validity` | `-1.0` if syntax error, `+1.0` if tests fail, `0.0` if tests still pass | Rewards executable bugs, rejects broken syntax. |
| `plausibility` | AST similarity score from `compute_ast_distance` | Rewards small realistic edits over random corruption. |
| `learnability` | `+1.0` when recent solver success rate is between `0.1` and `0.9` | Rewards bugs that are neither trivial nor impossible. |

The proposer is therefore rewarded for bugs that are:

- valid Python,
- test-breaking,
- close to the original AST,
- useful training examples for the solver.

### Solver Reward

Implemented in [`training/rewards.py`](training/rewards.py):

```python
solved = tests_passed and not syntax_error
reward = 1.0 if solved else 0.0
```

Every solver result is recorded in a per-seed rolling deque of length `20`. The proposer uses this history through `get_solve_rate(seed_id)` to estimate whether a bug is learnable.

### Plausibility Grader

Implemented in [`server/plausibility.py`](server/plausibility.py).

The plausibility score compares AST dumps of the clean and mutated code using a Levenshtein-style fuzz ratio:

| AST similarity ratio | Score | Interpretation |
| --- | --- | --- |
| `100` | `0.0` | No edit, not a useful bug. |
| `85` to `99` | `1.0` | Small realistic mutation. |
| `50` to `84` | Linear decay down to `0.1` | Medium-sized change. |
| `< 50` | `0.0` | Too different, likely unrealistic. |

This discourages the proposer from replacing the whole function with nonsense.

### Notebook Reward

The Colab notebook at [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) uses the live OpenEnv server inside the reward function.

For each model completion, it:

1. Extracts Python code from the model output.
2. Calls `env.reset()`.
3. Calls `env.step(DebugzeroAction(...))`.
4. Computes reward from the returned observation.

That means training is connected to the real environment, not a static dataset. The notebook also evaluates baseline and trained policies and saves:

- `results/reward_curve.png`
- `results/loss_curve.png`
- `results/baseline_vs_trained_reward.png`
- `results/training_log.csv`

## Bug Injection Logic

The AST mutation engine lives in [`server/bug_injector.py`](server/bug_injector.py).

`inject_bug(original_code, proposed_operator)` parses the clean code, applies one AST mutation, unparses the result, and accepts it only if all safety checks pass.

Supported mutation operators:

| Operator | Example behavior |
| --- | --- |
| `off_by_one` | Integer constants are shifted by `+1` or `-1`. |
| `wrong_operator` | Comparisons and arithmetic operators are swapped, such as `<` to `>=` or `+` to `-`. |
| `wrong_builtin` | Built-ins are swapped, such as `min`/`max`, `any`/`all`, or `sum`/`len`. |
| `loop_boundary_shift` | `range(n)` becomes `range(n + 1)`, or a two-argument range shifts the start. |
| `condition_negation` | `if condition` becomes `if not condition`. |
| `missing_base_case` | A return inside an `if` body is replaced with `pass`. |
| `slice_boundary_corruption` | Slice lower or upper bounds are shifted. |
| `variable_swap` | Tuple assignment targets are swapped. |

Accepted mutations must satisfy four checks:

1. Original code parses.
2. Mutated code is actually different.
3. Mutated code does not include blocked imports.
4. Mutated code parses after mutation.

## Sandbox and Safety

Execution is handled by [`server/executor.py`](server/executor.py).

The executor builds:

```python
full_code = submitted_code + "\n\n" + tests
```

Then it validates and executes the code in a temporary file with a timeout.

Safety checks include:

- blocked imports: `os`, `sys`, `subprocess`, `shutil`, `pathlib`
- blocked built-ins: `__import__`, `eval`, `exec`, `open`
- AST parsing before execution
- AST walk to catch direct `Import`, `ImportFrom`, and blocked function calls
- subprocess timeout, currently `5` seconds
- temporary directory isolation for each execution

If code is unsafe but parses, the executor returns:

```text
Unsafe import detected.
```

If code does not parse, the executor returns a syntax-error observation.

## Training Pipeline

There are two training paths.

### Recommended: Colab Notebook

Use [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) for the hackathon submission.

It:

1. Installs DebugZero from GitHub.
2. Starts the packaged OpenEnv FastAPI server, or connects to a remote HF Space URL.
3. Runs an OpenEnv smoke test through `DebugzeroEnv`.
4. Builds prompts from live environment resets.
5. Uses TRL `GRPOTrainer`.
6. Uses Unsloth when available, with native TRL fallback.
7. Computes rewards through live `reset` and `step` calls.
8. Saves plots for the README and final presentation.

### Experimental Script

[`training/grpo_train.py`](training/grpo_train.py) contains an experimental GRPO trainer configuration and the richer reward functions from [`training/rewards.py`](training/rewards.py). It is useful as implementation reference, but the notebook is the clearer end-to-end artifact for judges because it connects directly to the environment and saves visible training evidence.

## Prompt Templates

[`training/dual_role_sampler.py`](training/dual_role_sampler.py) defines two role prompts.

The proposer prompt asks the model to:

- inject an adversarial but plausible bug,
- keep code syntax-valid,
- make the function fail tests,
- return only modified code.

The solver prompt asks the model to:

- inspect buggy code,
- repair it,
- return only corrected code.

## Evaluation

Tests live under [`eval/`](eval/).

Current checks cover:

- AST mutation behavior:
  - missing base case,
  - off-by-one mutation,
  - loop boundary shift,
  - wrong built-in,
  - condition negation,
  - safety checks.
- Executor behavior:
  - safe code passes,
  - blocked imports are rejected,
  - syntax errors are rejected,
  - correct code passes tests,
  - buggy code fails tests.

There is also a plausibility evaluation scaffold in [`eval/plausibility_eval.py`](eval/plausibility_eval.py) for comparing generated bugs with human-like bugs from the navidadkhah dataset.

Run local checks with:

```bash
pytest eval
```

## Running Locally

Install dependencies:

```bash
uv sync
```

Start the OpenEnv server:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Smoke-test with the client:

```python
from debugZero.client import DebugzeroEnv
from debugZero.models import DebugzeroAction

with DebugzeroEnv(base_url="http://localhost:8000") as env:
    obs = env.reset().observation
    print(obs.role_next)
    print(obs.current_code)

    buggy = obs.current_code.replace("distance < threshold", "distance <= threshold")
    result = env.step(DebugzeroAction(role="proposer", code=buggy))
    print(result.observation.tests_passed)
```

## Inference Checker

[`inference.py`](inference.py) is a standalone environment checker for the submitted Space. It runs full DebugZero episodes through the packaged OpenEnv client and logs every step in a compact format.

Run against the Hugging Face Space:

```bash
set DEBUGZERO_API_URL=https://YOUR-USERNAME-debugzero.hf.space
set NUM_EPISODES=3
python inference.py
```

By default, it uses a deterministic sanity policy that:

1. resets the environment,
2. submits a known failing proposer mutation,
3. submits the original clean solution as the solver repair,
4. verifies that proposer failure and solver success are both detected.

To use an LLM through the Hugging Face router or another OpenAI-compatible endpoint:

```bash
set DEBUGZERO_API_URL=https://YOUR-USERNAME-debugzero.hf.space
set API_BASE_URL=https://router.huggingface.co/v1
set HF_TOKEN=your_token_here
set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

Useful environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DEBUGZERO_API_URL` | `https://YOUR-USERNAME-debugzero.hf.space` | Remote OpenEnv Space URL. |
| `LOCAL_IMAGE_NAME` | unset | Docker image name for local OpenEnv image testing. |
| `NUM_EPISODES` | `3` | Number of episodes to run. |
| `MAX_STEPS` | `2` | Max steps per episode. DebugZero is normally proposer then solver. |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | OpenAI-compatible model endpoint. |
| `API_KEY` / `HF_TOKEN` | unset | Enables LLM mode when present. |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Chat model used for action generation. |

## Repository Structure

```text
debugZero/
|-- openenv.yaml                  # OpenEnv manifest
|-- README.md                     # Project and submission documentation
|-- models.py                     # Action, observation, and state schemas
|-- client.py                     # OpenEnv client
|-- server/
|   |-- app.py                    # FastAPI OpenEnv app
|   |-- debugZero_environment.py  # Environment state machine
|   |-- executor.py               # Code execution and safety checks
|   |-- bug_injector.py           # AST mutation engine
|   |-- plausibility.py           # AST similarity grader
|   `-- requirements.txt          # HF Space server dependencies
|-- training/
|   |-- rewards.py                # Proposer and solver reward functions
|   |-- dual_role_sampler.py      # Prompt templates
|   `-- grpo_train.py             # Experimental GRPO trainer script
|-- notebooks/
|   `-- train_colab.ipynb         # Recommended rerunnable training notebook
`-- eval/
    |-- test_bug_injector.py      # Mutation tests
    |-- test_executor.py          # Executor tests
    `-- plausibility_eval.py      # Plausibility evaluation scaffold
```

## Deployment Notes

The HF Space runs `server.app:app`, so imports are written to support both:

- top-level Space import mode: `server.app`
- installed package mode: `debugZero.server.app`

Server dependencies for the Space are in [`server/requirements.txt`](server/requirements.txt). The server requires `thefuzz` because `server/plausibility.py` imports it during app startup.

Because the Docker Space serves Uvicorn on port `8000`, the Hugging Face README metadata must include:

```yaml
sdk: docker
app_port: 8000
```

After pushing to Hugging Face, confirm:

- the Space builds successfully,
- `/schema` returns a valid OpenEnv schema,
- `reset` returns the HumanEval seed code,
- `step` returns `tests_passed` and `syntax_error`,
- the README links to the final Space URL and training evidence.

## Current Limitations and Next Steps

Current implementation details to be aware of:

- The server seed is currently a single HumanEval-style function, `HumanEval/0`.
- The server verifies behavior but does not emit shaped scalar rewards yet. Training computes those externally from observations.
- Tests are currently bundled in the environment seed. For a stronger benchmark, split public and hidden tests.
- The AST bug injector exists as a utility, while proposer actions currently submit full mutated code.
- The training notebook is the preferred proof artifact because it uses the live OpenEnv path and produces plots.

High-impact next steps:

- Add more HumanEval or curated seed tasks.
- Move shaped reward metadata into observations for easier external analysis.
- Add hidden tests and baseline-vs-trained examples to the README.
- Use the AST injector to generate proposer warm-start examples.
- Record qualitative before/after solver repairs for the final presentation.
