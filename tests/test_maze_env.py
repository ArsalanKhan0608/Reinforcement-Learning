import importlib.util
from pathlib import Path
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

module_path = Path(__file__).resolve().parents[1] / "Dec 4 Task" / "CustomMazeEnv.py"
spec = importlib.util.spec_from_file_location("custom_maze", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MazeEnv = module.MazeEnv
MAZE = [[0, -1], [-10, 10]]


def test_gymnasium_contract():
    check_env(MazeEnv(MAZE), skip_render_check=True)


def test_reset_restores_maze_and_episode_counters():
    env = MazeEnv(MAZE)
    initial, _ = env.reset(seed=123, options={})
    env.step(0)  # Invalid left move.
    env.step(0)
    assert env.invalid_step_count == 2
    env.time_step = 9
    env.valid_step = 4
    env.step(1)
    restored, info = env.reset(seed=123)
    np.testing.assert_array_equal(restored, initial)
    assert env.current_pos == env.start_pos
    assert (env.time_step, env.valid_step, env.invalid_step_count) == (0, 0, 0)
    assert info == {}


def test_wall_goal_and_observation():
    env = MazeEnv(MAZE)
    env.reset()
    obs, reward, terminated, truncated, _ = env.step(3)  # Down into wall.
    assert reward == -10 and not terminated and not truncated
    assert env.current_pos == (0, 0)
    env.step(1)  # Right.
    obs, reward, terminated, truncated, _ = env.step(3)  # Down to goal.
    assert reward == 10 and terminated and not truncated
    assert env.observation_space.contains(obs)
    assert obs.dtype == np.int32


@pytest.mark.parametrize("maze", [[], [0, 10], [[0, 10, -1]], [[0.0, -1.0], [-10.0, 10.0]], [[0, 11], [-1, 10]], [[0, -10], [-10, 10]], [[0, 0], [-1, 10]]])
def test_invalid_mazes_report_value_error(maze):
    with pytest.raises(ValueError):
        MazeEnv(maze)


@pytest.mark.parametrize("action", [-1, 4, 1.5])
def test_invalid_action_is_rejected(action):
    env = MazeEnv(MAZE)
    with pytest.raises(ValueError):
        env.step(action)
