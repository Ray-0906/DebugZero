# DebugZero

DebugZero is an OpenEnv self-play debugging environment where one language model plays two roles:

1. The Proposer injects a small, realistic bug into clean Python code.
2. The Solver repairs the bug using the sandbox feedback.

The project is built for the OpenEnv hackathon themes most closely aligned with:

- Theme #1: Multi-Agent Interactions
- Theme #4: Self-Improvement

The current codebase is not a toy demo. It has a deterministic seed bank, a verified bug bank, role-aware rewards, a GRPO training loop, and a live API smoke test that exercises the same environment path used for training.

## What Lives Where

| File | Role |
| --- | --- |
| [server/seed_bank.py](server/seed_bank.py) | Curated 6-task seed bank with canonical solutions and tests |
| [server/bug_bank.py](server/bug_bank.py) | Deterministic verified bug generation and train/eval holdout split |
| [server/debugZero_environment.py](server/debugZero_environment.py) | OpenEnv environment state machine for proposer/solver turns |
| [server/executor.py](server/executor.py) | Sandboxed subprocess execution for code plus tests |
| [server/bug_injector.py](server/bug_injector.py) | AST mutation engine for realistic bug injection |
| [server/plausibility.py](server/plausibility.py) | AST-distance plausibility scoring |
| [server/rewards.py](server/rewards.py) | Role-aware reward shaping and rolling solve-rate history |
| [training/dual_role_sampler.py](training/dual_role_sampler.py) | Proposer and solver prompt templates |
| [training/grpo_train.py](training/grpo_train.py) | Mixed-role GRPO dataset build, eval, and training workflow |
| [eval/api_baseline.py](eval/api_baseline.py) | Deterministic controls plus live API promise-check harness |
| [client.py](client.py) | OpenEnv client wrapper |
| [models.py](models.py) | Shared action/observation/state models |
| [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) | Notebook-first training workflow |

## How The Environment Works

Each episode starts from a seed function drawn from the curated bank. The Proposer mutates the clean function. The environment executes the candidate code in a sandbox, runs the seed-specific tests, and returns:

- `tests_passed`
- `syntax_error`
- `execution_result`
- `role_next`
- `metadata` including `seed_id`, `original_code`, and bug context when present

If the Proposer creates a real failing bug, the Solver gets that buggy code plus the failure summary and attempts to fix it. The current environment cycles deterministically through the seed bank so repeated runs are reproducible.

## Dataset

The current task bank is intentionally small and reproducible:

- 6 curated seed tasks
- 18 verified solver training bugs
- 6 eval holdout bugs
- 27 mixed-role rows per dataset build

The six seeds are:

- `has_close_elements`
- `sum_to_n`
- `middle_slice`
- `is_non_decreasing`
- `count_nonempty`
- `running_max`

The bug bank is not random text. It is built at runtime by applying AST mutations and keeping only verified bugs that:

- change the code
- still parse
- pass safety checks
- fail the seed tests

The default bug operators are:

- `wrong_operator`
- `wrong_builtin`
- `condition_negation`
- `off_by_one`
- `loop_boundary_shift`
- `slice_boundary_corruption`

The noisier mutators `variable_swap` and `missing_base_case` are kept out of the default bank so the training signal stays clean. Train/eval splitting is deterministic, and the eval side keeps one harder holdout bug per seed.

## Rewards

The reward design is role-aware and intentionally simple:

| Role | State | Reward |
| --- | --- | --- |
| Proposer | syntax error or unsafe code | `-0.5` |
| Proposer | unchanged or effectively no-op code | `0.0` |
| Proposer | changed code that still passes tests | `0.0` |
| Proposer | valid failing bug | `1.0 + plausibility_bonus + learnability_bonus` |
| Solver | syntax error or unsafe code | `-0.5` |
| Solver | tests pass | `1.0` |
| Solver | tests fail | `0.0` |

The proposer gets an AST-based plausibility bonus when the edit is small and realistic. The learnability bonus is driven by a rolling solve-rate history with a 20-episode window per seed; the bonus is only active when the current solve rate is in the middle band, roughly `0.2` to `0.8`.

That reward shape is why the environment is useful for GRPO: it is not just pass/fail, but it still keeps the signal clean enough to train on.

## Training

The main training path is the notebook-first workflow in [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb). It:

1. installs dependencies
2. builds the seed bank and verified bug bank
3. runs the deterministic API controls
4. runs the live API promise-check probe
5. runs a pre-training fixed evaluation
6. trains with TRL GRPO
7. runs the same fixed evaluation again
8. saves a before/after plot to `debugzero_model/debugzero_results.png`

For a quick local smoke test, use:

```bash
python -X utf8 training/grpo_train.py --dry_run
```

For a real training run, drop `--dry_run` and use the notebook or the same script on a GPU machine.

Model guidance:

- Best default for this repo: `unsloth/Qwen2.5-Coder-3B-Instruct`
- Fast smaller-model experiments: a 1B to 3B coder model
- If you have more time and memory: a 7B to 8B coder model

The solver prompt has two modes:

- `concise` mode is the default for smaller models
- `full` mode remains available for larger models later

If `bitsandbytes` is available, training uses `adamw_8bit`; otherwise it falls back to `adamw_torch`. The actual GRPO path calls `trainer.train()`, so this is a real training loop rather than a placeholder.

## Evaluation

The live API smoke test in [eval/api_baseline.py](eval/api_baseline.py) has two layers:

1. deterministic controls
2. live API probing across all 6 seeds

The deterministic controls verify that:

- canonical seed code passes
- verified bugs fail
- syntax errors are detected

The live API probe then reports:

- proposer success rate
- solver success rate
- proposer syntax-error rate
- solver syntax-error rate
- average proposer reward
- average solver reward
- one representative success
- one representative failure

It also prints which step succeeded for proposer and solver attempts, so you can tell whether the model solved an episode on the first attempt or needed multiple turns.

To run the probe, set the environment variables and launch the server first:

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_MODEL="meta-llama/llama-3.1-8b-instruct"
$env:DEBUGZERO_ENV_URL="http://localhost:8000"
python -X utf8 eval/api_baseline.py
```

The `OPENAI_MODEL` value can be any strong coding model. A capable 7B to 8B class model gives a clearer smoke test than a weak model.

## Setup And Run

Install dependencies:

```bash
uv sync
```

Start the OpenEnv server from the repo root:

```bash
uv run --project . server
```

You can also run the FastAPI app directly:

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Run the API baseline:

```bash
python -X utf8 eval/api_baseline.py
```

Run the GRPO smoke test:

```bash
python -X utf8 training/grpo_train.py --dry_run
```

The notebook path is the recommended place to do the full training run, especially if you are moving between Colab and a local validation pass.

## Results And Evidence

The training workflow writes a summary plot to `debugzero_model/debugzero_results.png` and prints before/after fixed-eval metrics in the terminal. That gives you a quick way to show whether the solver pass rate and reward moved after training.

## Safety

DebugZero does not execute model-generated code directly in the host process. The executor writes code and tests to a temporary file, runs them in a subprocess, blocks unsafe imports and builtins, and returns a structured result. The OpenEnv server then wraps that environment behind the normal client/server interface.

## Notes

- The current task bank is deliberately compact so you can see signal quickly.
- If you want broader training later, the easiest upgrade is to add more `SeedSpec` entries to [server/seed_bank.py](server/seed_bank.py).
- Docker and deployment assets are present, but the current workflow is centered on local validation, API probing, and notebook training.
