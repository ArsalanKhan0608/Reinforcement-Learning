import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

rat_mark = 0.5  # Define the rat_mark value, change as needed
LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3  # Define action constants

class Qmaze(gym.Env):
    def __init__(self, maze, rat=(0, 0)):
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        self.target = (nrows - 1, ncols - 1)
        self.free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == 1.0]
        self.free_cells.remove(self.target)
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if rat not in self.free_cells:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        self.rat = rat
        self.state = rat + ('start',)
        self.min_reward = -0.5 * self._maze.size
        self.total_reward = 0
        self.visited = set()

        self.action_space = spaces.Discrete(4)  # Define action space (left, up, right, down)
        self.observation_space = spaces.Box(low=0, high=1, shape=self._maze.shape + (1,), dtype=np.float32)

    def reset(self):
        self.total_reward = 0
        self.visited = set()
        self.state = self.rat + ('start',)
        return np.expand_dims(self.observe(), axis=-1)

    def step(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        obs = np.expand_dims(self.observe(), axis=-1)
        done = True if status != 'not_over' else False
        return obs, reward, done, {}

    def update_state(self, action):
        nrow, ncol, nmode = self.state

        if self._maze[nrow, ncol] > 0.0:
            self.visited.add((nrow, ncol))

        valid_actions = self.valid_actions()

        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            elif action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:
            nmode = 'invalid'

        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        nrow, ncol, _ = self.state
        nrows, ncols = self._maze.shape

        if nrow == nrows - 1 and ncol == ncols - 1:
            return 1.0
        if self.state[2] == 'blocked':
            return self.min_reward - 1
        if (nrow, ncol) in self.visited:
            return -0.25
        if self.state[2] == 'invalid':
            return -0.75
        if self.state[2] == 'valid':
            return -0.04

    def observe(self):
        canvas = self.draw_env()
        return canvas

    def draw_env(self):
        canvas = np.copy(self._maze)
        nrow, ncol, _ = self.state
        canvas[nrow, ncol] = rat_mark
        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        nrow, ncol, _ = self.state
        nrows, ncols = self._maze.shape

        if nrow == nrows - 1 and ncol == ncols - 1:
            return 'win'

        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, _ = self.state
        else:
            row, col = cell

        actions = [0, 1, 2, 3]
        nrows, ncols = self._maze.shape

        if row == 0:
            actions.remove(1)
        elif row == nrows - 1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == ncols - 1:
            actions.remove(2)

        if row > 0 and self._maze[row - 1, col] == 0.0:
            actions.remove(1)
        if row < nrows - 1 and self._maze[row + 1, col] == 0.0:
            actions.remove(3)

        if col > 0 and self._maze[row, col - 1] == 0.0:
            actions.remove(0)
        if col < ncols - 1 and self._maze[row, col + 1] == 0.0:
            actions.remove(2)

        return actions



# Define your maze here
maze = np.array([[1.0, 1.0, 1.0],
                 [1.0, 0.0, 1.0],
                 [1.0, 1.0, 1.0]])  # Example maze

# Create Qmaze environment
env = Qmaze(maze)

# Wrap the environment using VecEnv for compatibility with SB3
env = make_vec_env(lambda: env, n_envs=1)

# Create and train DQN agent
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=int(1e4))  # Train for 10,000 timesteps (you can adjust this)

# Evaluate the trained agent
obs = env.reset()
for _ in range(1000):  # Evaluation loop for 1000 steps
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()  # Render the environment during evaluation
    if dones:
        obs = env.reset()

# Save the trained model if needed
model.save("dqn_maze_model")