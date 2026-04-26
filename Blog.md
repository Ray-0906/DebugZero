# DebugZero: Teaching a Coding Agent to Create and Fix Bugs

Most code benchmarks ask a model to write a fresh solution from scratch. That is useful, but it skips a big part of real programming work: debugging code that is almost correct.

That is the problem we built **DebugZero** to explore.

DebugZero is an OpenEnv environment where a coding agent learns through a two-role game:

- a **Proposer** takes a correct function and introduces a small but meaningful bug
- a **Solver** takes that buggy function and tries to repair it

The environment runs the submitted code in a sandbox, executes tests, and returns structured observations and rewards. In other words, the model does not just generate code and hope for the best. It acts inside an environment that can tell it whether a bug is real, whether a fix works, and whether the behavior is improving over time.

## Why we built it

We wanted an environment that treats debugging as a first-class skill.

In practice, strong programmers do more than write correct code. They also:

- recognize how correct-looking code can fail
- make small, targeted edits instead of rewriting everything
- use test failures as evidence
- recover from mistakes efficiently

Static benchmarks usually measure the end result. DebugZero is meant to train the process.

## How an episode works

Each episode starts from a clean seed task: a short Python function plus a hidden test harness.

On the first turn, the proposer submits a modified version of the function. The goal is not to destroy the program randomly. The goal is to create a bug that is realistic, small, and detectable by tests.

The environment then:

1. parses the submitted code
2. executes it in a sandboxed subprocess
3. runs the task tests
4. returns the current code, execution result, test status, reward, and next role

If the proposer successfully creates a valid bug, the solver gets the next turn. The solver then submits a repaired function, and the environment checks whether the original behavior has been restored.

This makes the whole loop executable and grounded. The agent is not rewarded for sounding plausible. It is rewarded for actually changing program behavior in the intended way.

## What makes the reward signal useful

DebugZero uses role-aware rewards instead of a single generic success metric.

For the proposer, reward is higher when the bug is:

- syntactically valid
- actually test-breaking
- close to the original implementation rather than random corruption

For the solver, reward is higher when the fix cleanly restores the expected behavior.

That design matters because it pushes both roles toward realistic debugging behavior. The proposer learns to create useful failures. The solver learns to make precise repairs.

## What we trained

We trained a policy for this environment using **GRPO** and role-conditioned prompting. One important design choice was to train against the **deployed environment itself**, not against notebook-local copies of the environment logic.

That means the training loop interacts with the same OpenEnv interface that serves the environment in deployment:

- reset the environment
- observe the current task state
- submit a proposer or solver action
- receive reward and updated observation

This kept training aligned with the real environment instead of drifting into a separate offline approximation.

## Why the two-role setup is interesting

The most fun part of DebugZero is that it creates its own pressure to improve.

If the solver becomes stronger, the proposer has to invent better bugs. If the proposer becomes better at making subtle failures, the solver has to become more precise at repair. That gives us a natural self-play curriculum for debugging.

Instead of hand-authoring every training example, we get an environment where challenge and skill can rise together.

## What DebugZero is really trying to test

At a deeper level, this project is about whether coding agents can become better debuggers through interaction rather than static supervision alone.

We care about questions like:

- Can an agent learn to create realistic failure modes?
- Can it repair bugs without over-editing the program?
- Can self-play produce a useful curriculum for code reasoning?
- Can reward grounded in execution and tests teach something that static datasets miss?

DebugZero is our attempt at turning those questions into something concrete and measurable.

## Links

- Hugging Face Space: https://the-fool-09-debugzero.hf.space
- Hugging Face project page: https://huggingface.co/spaces/The-Fool-09/debugZero
- Training notebook: `notebooks/train_colab_updated_1.ipynb`

In short, DebugZero is not just a benchmark where a model writes code. It is an environment where the model learns from failure, creates new failure cases, and improves through the loop of breaking and repairing programs. That is the behavior we wanted to surface, and that is what we trained for.
