---
title: DebugZero Environment Server
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - debugging
  - self-play
---

# DebugZero

Most coding agents look better at greenfield generation than they do at the thing developers actually need every day: taking almost-correct code, finding the one subtle mistake, and repairing it without breaking everything else.

DebugZero is a self-play debugging environment for that exact gap. Instead of giving a model a static benchmark and asking it to patch code after the fact, DebugZero turns debugging into a game between two roles:

1. The `Proposer` takes correct Python code and injects one small, realistic bug.
2. The `Solver` sees the broken code plus the sandbox feedback and tries to repair it.

The result is an environment where the agent is not rewarded for generic code generation, but for a much narrower and more useful capability: making and fixing the kind of small, plausible mistakes that dominate real debugging work.

If the long-term goal is a code agent that can recover from failure instead of only autocomplete its way forward, this is the muscle we want to train.

## Hugging Face Space

- Environment Space: [The-Fool-09/debugZero](https://huggingface.co/spaces/The-Fool-09/debugZero)

## 1. Problem

There is a real capability gap between "can write code" and "can debug code."

Most code models are trained to continue text or produce a final answer. Real debugging is different. In the wild, the code is usually not blank; it is already there, mostly right, and failing for one annoying reason. A good debugger has to:

- read an implementation and preserve the intent
- notice a small local behavioral bug, not just a syntax problem
- use test failures as evidence
- repair the bug with the smallest correct change

That gap matters because many developer-facing agents will spend more time fixing near-correct code than writing fresh files from scratch. Static repair benchmarks are useful, but they do not create an adversarial loop where one model learns to generate realistic failures and another learns to resolve them.

DebugZero targets exactly that loop: one role learns to produce believable breakages, the other learns to recover. That makes the environment useful both as an evaluator and as a training ground.

## 2. Environment

Each episode begins from a curated seed function in [server/tasks.py](server/tasks.py). The current bank is intentionally compact and reproducible:

- 6 curated seed tasks
- 18 verified training bugs
- 6 eval holdout bugs
- 27 mixed-role dataset rows per build

The six seed functions are:

- `has_close_elements`
- `sum_to_n`
- `middle_slice`
- `is_non_decreasing`
- `count_nonempty`
- `running_max`

### What happens in one episode

An episode is short and concrete:

1. The environment starts from a known-correct seed function.
2. The `Proposer` submits a version with one realistic bug.
3. The sandbox executes the code and runs tests.
4. The `Solver` uses the broken code plus execution feedback to repair it.

That loop is simple enough to be reproducible, but still rich enough to capture the part of coding work where agents usually wobble: reading intent, using evidence, and making a minimal correction.

### What the agent sees

After every step, the environment returns:

- `current_code`
- `execution_result`
- `tests_passed`
- `syntax_error`
- `role_next`
- `metadata`, including `seed_id` and `original_code`

This makes the environment grounded in program behavior rather than pure text imitation. The model is always acting against executable feedback.

### What the agent does

The action space is simple on purpose:

- The `Proposer` submits a full Python function containing exactly one small logical bug.
- The `Solver` submits a full repaired Python function.

The environment in [server/debugZero_environment.py](server/debugZero_environment.py) executes candidate code in the sandbox from [server/executor.py](server/executor.py), runs the task tests, and advances the role turn.

### What gets rewarded

The reward is role-aware:

| Role | Good behavior | Bad behavior |
| --- | --- | --- |
| Proposer | Create a small, plausible bug that fails tests | Syntax errors, unsafe code, or edits that still pass |
| Solver | Repair the bug and pass tests | Syntax errors, unsafe code, or failed fixes |

The proposer reward also includes a plausibility bonus from [server/graders.py](server/graders.py). That matters because we do not want noisy or destructive corruption. We want bugs that look like mistakes a human might actually make.

In other words, the environment is not asking "can the model produce code-shaped text?" It is asking "can the model create and repair realistic failures under execution pressure?"

## 3. Results

### Environment validation

Before training, the repo includes a deterministic validation pass in [eval/api_baseline.py](eval/api_baseline.py). Running it locally on April 26, 2026 produced:

- Canonical pass count: `6/6`
- Verified bug fail count: `6/6`
- Syntax detection count: `6/6`

Those three checks matter because they show the environment has real signal:

- clean reference code succeeds
- generated holdout bugs actually break behavior
- obviously bad code is rejected cleanly

So before any RL story starts, we already know the environment is behaving sensibly.

### Training smoke-test result

I also ran the local GRPO smoke test:

```bash
python -X utf8 training/grpo_train.py --dry_run
```

That dry run uses the tiny fallback local model and only `2` training steps, so it is not meant to be a competitive final result. It is meant to answer a more basic question: does the full loop run end to end and emit measurable before/after artifacts?

It did. The run produced:

- [debugzero_model/debugzero_results.png](debugzero_model/debugzero_results.png)
- [debugzero_model/proposer_metrics.json](debugzero_model/proposer_metrics.json)

The actual dry-run metrics were:

| Metric | Pre | Post |
| --- | --- | --- |
| Solver pass rate | `0.00` | `0.00` |
| Solver syntax error rate | `1.00` | `1.00` |
| Solver mean reward | `-0.50` | `-0.50` |
| Proposer valid bug rate | `0.00` | `0.00` |
| Proposer syntax error rate | `1.00` | `1.00` |
| Proposer mean reward | `-0.50` | `-0.50` |

![Dry-run training results](debugzero_model/debugzero_results.png)

That is not a "look how good the model is" result. It is almost the opposite, and that is useful. A tiny local model does not magically solve the environment. The debugging tasks are hard enough to expose failure modes immediately, and the pipeline still records those failures in a way we can improve on with stronger models and longer training.

In other words: the smoke test shows that DebugZero is not a toy environment that collapses under trivial policies. It produces a measurable training target, and it is honest when the model is not yet good enough.

### What changes after real training

The full training workflow in [training/grpo_train.py](training/grpo_train.py) evaluates the model before and after training and saves a comparison plot. The headline metrics are:

- solver pass rate
- solver mean reward
- proposer break rate
- proposer mean reward

Those are the numbers that matter for this project. If training is helping, we should see the solver repair more holdout bugs, the proposer produce more valid failures, and the mean rewards move in the right direction. The dry run establishes the instrumentation; larger real runs are where the improvement story should become visible.

## 4. Why It Matters

DebugZero matters to anyone building agents that interact with code under uncertainty:

- For coding-agent researchers: it turns debugging into a measurable environment with executable feedback.
- For RL-for-code work: it gives a reward signal that is richer than simple pass/fail while still staying grounded in tests.
- For developer tools: it targets the everyday regime where code is almost correct and small repairs matter more than full rewrites.
- For education and evaluation: it cleanly separates "can propose a realistic bug" from "can repair one."

The deeper reason this matters is that self-improvement for code agents should not only mean "generate more code." It should also mean "generate the right failures, learn from them, and recover."

That is the audience for this environment: people who care about trustworthy coding agents, better debugging behavior, and measurable progress on the messy middle between passing and failing.

## Repository Guide

If you want to navigate the code quickly:

| File | Role |
| --- | --- |
| [server/tasks.py](server/tasks.py) | Curated task bank used by the environment |
| [bug_bank.py](bug_bank.py) | Verified bug generation and train/eval split |
| [server/debugZero_environment.py](server/debugZero_environment.py) | Main environment state machine |
| [server/executor.py](server/executor.py) | Sandboxed execution against tests |
| [server/bug_injector.py](server/bug_injector.py) | AST mutation engine for realistic bug injection |
| [server/graders.py](server/graders.py) | Reward shaping, solve-rate history, and plausibility scoring |
| [training/dual_role_sampler.py](training/dual_role_sampler.py) | Proposer and solver prompt templates |
| [training/grpo_train.py](training/grpo_train.py) | Dataset build, fixed eval, and GRPO training workflow |
| [eval/api_baseline.py](eval/api_baseline.py) | Deterministic controls and live API probe |
| [inference.py](inference.py) | Multi-episode inference runner with flat logs |

## How To Run

Install dependencies:

```bash
uv sync
```

Start the server:

```bash
uv run --project . server
```

Run deterministic controls and the optional live API probe:

```bash
python -X utf8 eval/api_baseline.py
```

Run the inference loop with flat `[START]`, `[STEP]`, and `[END]` logs:

```bash
python -X utf8 inference.py
```

Run the GRPO smoke test:

```bash
python -X utf8 training/grpo_train.py --dry_run
```

## Additional References

- Hugging Face Space: [The-Fool-09/debugZero](https://huggingface.co/spaces/The-Fool-09/debugZero)
- Implementation guide: [implementation.md](implementation.md)
- Notebook workflow: [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb)
- API baseline harness: [eval/api_baseline.py](eval/api_baseline.py)
- Inference runner: [inference.py](inference.py)

External materials such as slides, blog posts, or demo videos are not published in this repo yet. When they exist, this section is where they should be linked.
