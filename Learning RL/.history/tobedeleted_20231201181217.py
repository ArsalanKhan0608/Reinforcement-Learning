import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ModifiedMazeGameEnv(gym.Env):
    def __init__(self, maze):
        self.maze = np.array(maze)
        self.original_maze = np.array(maze)  # Store the original maze

        print(f"Maze: {self.maze}")

        self.start_pos = np.argwhere(self.maze == 0)[0]
        print(f"Starting Position: {self.start_pos}")

        self.goal_pos = np.argwhere(self.maze == 10)[0]
        print(f"Goal Position: {self.goal_pos}")

        self.current_pos = self.start_pos
        print(f"Current Position: {self.current_pos}")

        self.num_rows, self.num_cols = self.maze.shape
        print(f"Maze Dimensions: {self.num_rows} rows, {self.num_cols} cols")

        self.last_action = None
        print(f"Last Action: {self.last_action}")

        self.repeated_action_count = 0
        print(f"Repeated Action Count: {self.repeated_action_count}")

        self.action_space = spaces.Discrete(4)
        print(f"Action Space: {self.action_space}")

        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.num_rows * self.num_cols,), dtype=np.int32)
        print(f"Observation Space: {self.observation_space}")

        self.total_reward = 0
        print(f"Total Reward: {self.total_reward}")
        
    def reset(self, seed=None, options=None, **kwargs):
        self.total_reward = 0
        self.current_pos = self.start_pos
        self.last_action = None
        self.repeated_action_count = 0

        # Reset the maze to the original state
        self.maze = np.array(self.original_maze)
        self.maze[tuple(self.current_pos)] = 0  # Place the agent at the start position

        return self._get_observation(), {}  # Return observation and an empty info dictionary


    
    
    def step(self, action):
        new_pos = np.array(self.current_pos)
        print(f"New Position Before Action: {new_pos}")

        if action == 0:
            new_pos[0] -= 1
        elif action == 1:
            new_pos[0] += 1
        elif action == 2:
            new_pos[1] -= 1
        elif action == 3:
            new_pos[1] += 1
        print(f"New Position After Action {action}: {new_pos}")

        valid_move = self.is_valid_position(new_pos)
        print(f"Is Move Valid: {valid_move}")

        if valid_move:


            # Reset the previous position to its original value
            self.maze[tuple(self.current_pos)] = self.original_maze[tuple(self.current_pos)]

            # Update current position
            self.current_pos = new_pos

            # Mark the new position with 0
            self.maze[tuple(self.current_pos)] = 0
            print(f"Updated Current Position: {self.current_pos}")

            self.last_action = action
            print(f"Updated Last Action: {self.last_action}")

            self.repeated_action_count = 0
            print(f"Reset Repeated Action Count: {self.repeated_action_count}")
        else:
            reward = -0.1
            print(f"Invalid Move Reward: {reward}")

            if action == self.last_action:
                self.repeated_action_count += 1
            else:
                self.repeated_action_count = 0
            print(f"Updated Repeated Action Count: {self.repeated_action_count}")

        reward, done = self._get_reward_and_done()
        print(f"Reward: {reward}, Done: {done}")

        if self.repeated_action_count > 3 and not valid_move:
            reward -= 1
            print(f"Penalty for Repeated Invalid Actions: {reward}")

        self.total_reward += reward
        print(f"Total Reward: {self.total_reward}")

        observation = self._get_observation()
        print(f"Observation: {observation}")
        return observation, reward, done, False, {}



    def is_valid_position(self, pos):
        row, col = pos
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            print(f"Invalid Position: {pos}")
            return False
        if self.maze[row, col] == -10:
            print(f"Position is an Obstacle: {pos}")
            return False
        print(f"Valid Position: {pos}")
        return True

    def _get_reward_and_done(self):
        if np.array_equal(self.current_pos, self.goal_pos):
            print(f"Goal Reached at Position: {self.current_pos}")
            return 10, True
        else:
            reward = self.maze[tuple(self.current_pos)]
            print(f"Reward for Position {self.current_pos}: {reward}")
            return reward, False

    def _get_observation(self):
        observation = self.maze.flatten().astype(np.int32)
        print(f"Observation: {observation}")
        return observation

