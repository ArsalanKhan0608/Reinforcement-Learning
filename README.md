# Reinforcement learning exercises

A collection of notebooks and maze experiments. The `Dec 4 Task` directory contains a Gymnasium maze environment and Stable-Baselines3 DQN/A2C examples.

## Run the maze examples

Use Python 3.10 or newer. From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r "Dec 4 Task/requirements.txt"
python "Dec 4 Task/RLwithDQN.py"
python "Dec 4 Task/loadDQN.py"
# Or train the A2C example:
python "Dec 4 Task/RLwithA2C.py"
```

Training runs for 20,000 timesteps. DQN saves `my_model.zip` beside its script; the loader reads that same file, regardless of your working directory. Run DQN training before the loader. Model quality and convergence are not guaranteed.

`MazeEnv` accepts a nonempty square integer grid: `-10` is a wall, `-1` is a walkable cell, `0` is the unique start, and `10` is the unique goal. A route to the goal must exist. Actions follow the existing implementation: **0 left, 1 right, 2 up, 3 down**. `reset(seed=..., options=...)` returns `(observation, info)`; `step(action)` returns `(observation, reward, terminated, truncated, info)`.

## Environment regression tests

```sh
python -m pip install pytest
python -m pytest -q tests/test_maze_env.py
```

These tests exercise the real Gymnasium API checker, reset behavior, goal and wall transitions, and input validation. They require Gymnasium and NumPy but do not train an agent. The other notebooks and `Learning RL` examples have separate experimental dependencies and are not covered by this setup.
