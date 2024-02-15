import gymnasium as gym
import numpy as np
from gymnasium import spaces
class ModifiedMazeGameEnv(gym.Env):
    def __init__(self, maze):
        self.maze = np.array(maze)  # maze represented as a 2D numpy array
        self.start_pos = self._find_position('S')
        self.goal_pos = self._find_position('G')
        self.current_pos = self.start_pos
        self.num_rows, self.num_cols = self.maze.shape

        # 4 possible actions, 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        
        # observation_space is the entire maze in 1D form
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_rows * self.num_cols,), dtype=np.int32)

        self.total_reward = 0

    def reset(self, **kwargs):
        self.total_reward = 0
        self.current_pos = self.start_pos
        return self._get_observation()

    def step(self, action):
        new_pos = np.array(self.current_pos)

        # Define action effects
        if action == 0:  # Up
            new_pos[0] -= 1
        elif action == 1:  # Down
            new_pos[0] += 1
        elif action == 2:  # Left
            new_pos[1] -= 1
        elif action == 3:  # Right
            new_pos[1] += 1

        # Check if the new position is valid
        if self.is_valid_position(new_pos):
            self.current_pos = new_pos

        # Update rewards and check if goal is reached
        reward, done = self._get_reward_and_done()
        self.total_reward += reward
        return self._get_observation(), reward, done, {}

    def is_valid_position(self, pos):
        row, col = pos
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False
        if self.maze[row, col] == '#':
            return False
        return True

    def _get_reward_and_done(self):
        if np.array_equal(self.current_pos, self.goal_pos):
            return 10, True  # High positive reward for reaching the goal
        elif self.maze[tuple(self.current_pos)] == '#':
            return -10, False  # Negative reward for hitting a wall
        else:
            return -1, False  # Small negative reward for other moves

    def _find_position(self, char):
        return np.argwhere(self.maze == char)[0]

    def _get_observation(self):
        # Create a 1D representation of the maze with the current position marked
        observation = np.zeros(self.num_rows * self.num_cols)
        observation[self.current_pos[0] * self.num_cols + self.current_pos[1]] = 1
        return observation
    
