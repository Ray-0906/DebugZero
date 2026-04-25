# DebugZero Implementation Guide

## What This Repository Is

`DebugZero` is an `OpenEnv`-style debugging environment built around a two-role loop:

- a `proposer` takes correct Python code and tries to inject a realistic logical bug
- a `solver` receives buggy code and tries to repair it

The repository is designed to support three things at once:

1. a runnable environment server
2. a GRPO-style training pipeline
3. an API-based evaluation harness for smoke-testing the environment before training

The current setup is intentionally small and deterministic so it is easy to verify, train on, and explain in a hackathon setting.

## High-Level Environment Workflow

At a high level, one episode works like this:

1. The environment resets onto one curated seed task from the seed bank.
2. The `proposer` sees the clean reference implementation and tries to make one small logical mistake.
3. The environment executes the proposed code against the task tests.
4. If the code now fails tests in a valid way, the proposer gets rewarded and the environment hands the buggy code to the `solver`.
5. The `solver` tries to repair the code.
6. The environment executes the solver output against the same tests.
7. If the fix passes, the solver gets rewarded and the episode ends.

This gives a clean adversarial/self-improvement loop:

- the proposer learns to create realistic, test-breaking bugs
- the solver learns to reverse those bugs
- the reward functions are role-specific

## Project Architecture

The project is split into a few clear layers.

### 1. Task and Bug Data

- `seed_bank.py` defines the curated clean tasks
- `bug_bank.py` builds verified buggy variants from those tasks

### 2. Environment Runtime

- `server/debugZero_environment.py` is the main environment state machine
- `server/executor.py` runs generated code safely against tests
- `server/bug_injector.py` provides AST-based bug mutations
- `server/plausibility.py` scores whether a proposer bug looks realistic
- `server/app.py` exposes the environment through a FastAPI/OpenEnv server

### 3. Shared Interface

- `models.py` defines the request/response data models shared across client and server
- `client.py` provides a small client wrapper for interacting with the environment

### 4. Training

- `training/dual_role_sampler.py` builds role-specific prompts
- `training/rewards.py` computes proposer and solver rewards
- `training/grpo_train.py` builds the dataset, runs evaluation, and launches GRPO training

### 5. Evaluation

- `eval/api_baseline.py` is the main smoke-test harness

It first runs deterministic controls, then optionally runs a live API probe using a model served through an OpenAI-compatible endpoint.

## Data Design

The current data setup is not Faker-style random generation. It is a curated-and-verified pipeline.

### Seed Bank

The environment uses a fixed curated set of six Python tasks:

- `HumanEval/0` -> `has_close_elements`
- `DebugZero/1` -> `sum_to_n`
- `DebugZero/2` -> `middle_slice`
- `DebugZero/3` -> `is_non_decreasing`
- `DebugZero/4` -> `count_nonempty`
- `DebugZero/5` -> `running_max`

Each seed contains:

- `seed_id`
- `entrypoint`
- `prompt`
- `canonical_solution`
- `test`

### Bug Bank

The bug bank is generated programmatically from the seed bank using AST mutation operators. Only bugs that meet all of the following conditions are kept:

- the code actually changed
- the mutated code still parses
- the code is considered safe enough to execute
- the mutated code fails the seed's tests

The current bug bank split is:

- `18` training bug samples
- `6` evaluation holdout bug samples

This gives the training loop a deterministic but nontrivial set of bug/fix tasks.

## Reward Design

The reward functions are intentionally role-specific.

### Proposer Reward

The proposer is rewarded for creating realistic failing bugs, not just for changing code.

Current behavior:

- syntax error or unsafe output -> `-0.5`
- unchanged code -> `0.0`
- changed code that still passes tests -> `0.0`
- valid failing bug -> `1.0 + plausibility_bonus + learnability_bonus`

Where:

- `plausibility_bonus` rewards bugs that look more like realistic programmer mistakes
- `learnability_bonus` favors bugs that are neither trivial nor impossible for the solver

### Solver Reward

The solver reward is intentionally simpler:

- syntax error -> `-0.5`
- failed fix -> `0.0`
- passing fix -> `1.0`

This makes the solver side easier to optimize and easier to explain.

## Training Workflow

The training path is centered on GRPO-style optimization.

### What Gets Trained

The training dataset is mixed-role:

- solver-heavy by design
- still includes proposer rows so both roles are represented

The current mixed-role build uses:

- `18` solver rows
- `9` proposer rows
- `27` rows total

### Training Loop Shape

`training/grpo_train.py` does the following:

1. builds the verified bug bank
2. constructs the mixed-role dataset
3. formats prompts through the dual-role sampler
4. evaluates pre-training behavior on a fixed holdout set
5. runs GRPO training
6. evaluates post-training behavior
7. saves a small results plot

There is also a `--dry_run` path for quick local smoke testing.

### Recommended Models

Best default for this environment:

- `unsloth/Qwen2.5-Coder-3B-Instruct`

Reasonable alternatives:

- `1B` to `2B` coder models for faster cheap runs
- `7B` to `8B` coder/instruct models for stronger evaluation if compute allows

In practice:

- `1B` to `3B` is the most sensible training range for this repo's size and task complexity
- `7B` to `8B` is useful as a stronger API smoke-test model or a higher-end final experiment

## Evaluation Workflow

The main evaluation script is `eval/api_baseline.py`.

It has two phases.

### 1. Deterministic Controls

Before any live model call, it verifies that:

- canonical code passes
- verified buggy code fails
- obvious syntax errors are detected

This is the fast check that the environment has real signal.

### 2. Live API Probe

If `OPENAI_API_KEY` and `OPENAI_MODEL` are present, it then runs a multi-episode proposer/solver loop over the seed bank.

It reports:

- proposer success rate
- solver success rate
- proposer valid bug rate
- proposer unchanged rate
- proposer changed-but-passing rate
- proposer syntax rate
- solver syntax rate
- average proposer reward
- average solver reward
- one representative success
- one representative failure

This is the main pre-training sanity check.

## End-to-End Runtime Flow

If we trace one full path through the system, it looks like this:

1. `seed_bank.py` provides a clean seed task.
2. `server/debugZero_environment.py` resets onto that seed.
3. A proposer model generates code from a proposer prompt built by `training/dual_role_sampler.py` or `eval/api_baseline.py`.
4. `server/executor.py` runs the candidate code against the seed tests.
5. `training/rewards.py` computes proposer reward.
6. If the proposer created a valid failing bug, the solver gets a repair prompt.
7. The solver generates repaired code.
8. `server/executor.py` runs the repair candidate.
9. `training/rewards.py` computes solver reward.
10. `training/grpo_train.py` uses these rewards during GRPO training or fixed evaluation.

## Tracked Python Files

Below is what each currently tracked `.py` file is doing.

### Live Runtime And Training Files

#### [__init__.py](./__init__.py)

Marks the repository root package so imports can work cleanly in package-style execution.

#### [seed_bank.py](./seed_bank.py)

Defines the curated seed task bank. Each seed includes the prompt, canonical solution, test harness, and function entrypoint. This is the base dataset for the whole environment.

#### [bug_bank.py](./bug_bank.py)

Builds and stores verified buggy samples from the seed bank. It filters mutations down to samples that are syntactically valid, meaningfully changed, safe to run, and test-failing. It also splits them into training and evaluation holdouts.

#### [models.py](./models.py)

Defines the shared Pydantic models used across client/server communication. This includes the environment action and observation structures.

#### [client.py](./client.py)

Provides a small client interface for talking to the environment server using the shared models. It is the clean consumer-side entrypoint for external interaction.

#### [eval/api_baseline.py](./eval/api_baseline.py)

Runs the main smoke-test and evaluation workflow. It verifies deterministic controls first, then runs a live proposer/solver API probe across the seed bank when API credentials are available.

#### [server/__init__.py](./server/__init__.py)

Marks the `server` package.

#### [server/app.py](./server/app.py)

Creates the FastAPI/OpenEnv application and wires the environment into the server layer so external tools can call it.

#### [server/bug_injector.py](./server/bug_injector.py)

Contains the AST mutation logic used to generate plausible logical bugs from clean code. This is the mutation engine behind the verified bug bank.

#### [server/debugZero_environment.py](./server/debugZero_environment.py)

Implements the main environment state machine. It handles reset, proposer steps, solver steps, execution feedback, seed progression, and observation construction.

This is the heart of the repo.

#### [server/executor.py](./server/executor.py)

Runs model-generated Python code in a constrained execution path with safety checks and test execution. This is the file that turns raw code into pass/fail execution signals.

#### [server/plausibility.py](./server/plausibility.py)

Scores how realistic a proposer bug looks. That plausibility signal is used as part of proposer reward shaping.

#### [training/dual_role_sampler.py](./training/dual_role_sampler.py)

Builds the role-specific prompts used during training. It formats proposer prompts for bug injection and solver prompts for bug repair, including the concise solver mode used for smaller models.

#### [training/grpo_train.py](./training/grpo_train.py)

The main training entrypoint. It builds datasets, prepares the trainer, evaluates before and after training, and saves training artifacts such as the results plot.

#### [training/rewards.py](./training/rewards.py)

Defines the reward logic for proposer and solver outputs. It is the main source of learning signal for GRPO.

### Tracked Template / Scaffolding Files

These files are tracked in git, but they are not part of the live DebugZero runtime. They are template assets checked into the repo under `.claude/...` and appear to be scaffolding for generating OpenEnv environments.

#### [`.claude/skills/generate-openenv-env/assets/openenv_env_template/__init__.py`](./.claude/skills/generate-openenv-env/assets/openenv_env_template/__init__.py)

Template package marker for generated environments.

#### [`.claude/skills/generate-openenv-env/assets/openenv_env_template/client.py`](./.claude/skills/generate-openenv-env/assets/openenv_env_template/client.py)

Template client file used when scaffolding a new OpenEnv environment.

#### [`.claude/skills/generate-openenv-env/assets/openenv_env_template/models.py`](./.claude/skills/generate-openenv-env/assets/openenv_env_template/models.py)

Template shared models file used by the environment generator.

#### [`.claude/skills/generate-openenv-env/assets/openenv_env_template/server/__ENV_NAME___environment.py`](./.claude/skills/generate-openenv-env/assets/openenv_env_template/server/__ENV_NAME___environment.py)

Template environment implementation stub for newly generated environments.

#### [`.claude/skills/generate-openenv-env/assets/openenv_env_template/server/__init__.py`](./.claude/skills/generate-openenv-env/assets/openenv_env_template/server/__init__.py)

Template server package marker for generated environments.

#### [`.claude/skills/generate-openenv-env/assets/openenv_env_template/server/app.py`](./.claude/skills/generate-openenv-env/assets/openenv_env_template/server/app.py)

Template FastAPI/OpenEnv app file for generated environments.

## What This Repo Is Optimized For

This repository is optimized for:

- explaining the environment clearly
- proving that the environment has real reward signal
- running a small but defensible GRPO training loop
- showing before/after improvement, especially on the solver side

It is not optimized for:

- huge-scale dataset diversity
- deployment polish
- extremely large benchmark coverage

That tradeoff is deliberate. The current design favors clarity, determinism, and hackathon-speed iteration.

## In One Sentence

`DebugZero` is a compact self-play debugging environment where one model learns to inject realistic bugs, another learns to fix them, and the repo is structured so that this loop can be tested, trained, and explained cleanly.
