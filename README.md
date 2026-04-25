---
title: DebugZero
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# DebugZero Environment

*Our environment extends the Absolute Zero paradigm (Zhao et al., NeurIPS 2025) to adversarial bug-fixing self-play. The Proposer's mutation operators are implemented from scratch using Python AST manipulation across 8 operator types. The verifier adapts Mutahunter's execution pipeline.*

DebugZero is a self-play environment where one LLM plays two roles:
1. **Proposer**: Injects realistic bugs into clean Python functions using AST-level edits.
2. **Solver**: Fixes the bugs, practicing on the generated adversarial examples.

This project enables fully self-improving debugging capabilities without human-curated datasets, verified by a strict Python execution sandbox.

## Setup

First, install dependencies:
```bash
uv sync
```

## Running the Training Loop

We provide a direct implementation of the Hugging Face TRL GRPOTrainer, modified to support a dual-role `reward_fn` dynamic routing system.

```bash
python training/grpo_train.py
```

For the hackathon submission, use the rerunnable Colab notebook at
[`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb). It installs DebugZero
from GitHub, connects through the packaged OpenEnv client, trains with TRL/Unsloth
against live `reset`/`step` environment rollouts, and saves reward/loss plots for
the README.

## Repository Structure

```
debugZero/
├── server/
│   ├── app.py                   # OpenEnv FastAPI app
│   ├── debugZero_environment.py # Environment state machine tracking Proposer/Solver turns
│   ├── executor.py              # Isolated subprocess execution sandbox (timeout=5s)
│   ├── bug_injector.py          # AST mutation engine with 8 operators and strict safety checks
│   └── plausibility.py          # Levenshtein AST distance calculator 
├── training/
│   ├── rewards.py               # Calculation for Proposer and Solver rewards + learnability deque
│   ├── dual_role_sampler.py     # Prompt templates for Qwen2.5-Coder-3B-Instruct
│   └── grpo_train.py            # TRL trainer bridging output metadata to reward forms
├── eval/
│   └── plausibility_eval.py     # Offline benchmark tool against navidadkhah 25k bug dataset
├── models.py                    # Data types for the dual-role protocol
└── client.py                    # OpenEnv DebugZero client class
```

## Safety

DebugZero wraps the execution in a standard `openenv` container via FastAPI, but specifically filters AST payloads using an `is_safe` string filter to prevent `os`, `sys`, `subprocess`, `shutil`, and `pathlib` evaluations.
