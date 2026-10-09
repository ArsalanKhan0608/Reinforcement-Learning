from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
# Running the provided MazeEnv code with a test maze
class MazeEnv(gym.Env):
    def __init__(self, input_maze):
        super(MazeEnv, self).__init__()

        input_maze = np.array(input_maze)
        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # left, right, up, down
        self.time_step=0
        self.initial_maze = np.array(input_maze)  # Store the initial state of the maze
        self.maze = self._validate_maze(input_maze)
        self.start_pos = tuple(np.argwhere(self.maze == 0)[0])
        self.goal_pos = tuple(np.argwhere(self.maze == 10)[0])
        self.current_pos = self.start_pos
        self.valid_step=0
        self.invalid_step_count=0

        # Flatten the maze for the observation space
        maze_size = np.prod(self.maze.shape)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(maze_size,), dtype=np.int32)

    def _validate_maze(self, maze):
        maze = np.array(maze)

        if maze.ndim != 2 or not maze.size or maze.shape[0] != maze.shape[1]:
            raise ValueError("Maze must be square-shaped.")

        if not issubclass(maze.dtype.type, np.integer):
            raise ValueError("Maze values must be integers.")
        
        if not np.isin(maze, [-10, -1, 0, 10]).all():
            raise ValueError("Maze values must be -10 (wall), -1 (path), 0 (start), or 10 (goal).")

        if np.sum(maze == 0) != 1 or np.sum(maze == 10) != 1:
            raise ValueError("Maze must contain exactly one start position (0) and one goal position (10).")

        if not self.is_path_available(maze):
            raise ValueError("No valid path from start to goal.")
        
        return maze
    
    def is_path_available(self, maze):
        path_maze = np.where(maze == -10, 0, 1)
        start = tuple(np.argwhere(maze == 0)[0])
        goal = tuple(np.argwhere(maze == 10)[0])
        visited = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current == goal:
                return True

            for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # Right, Left, Down, Up
                next_pos = (current[0] + direction[0], current[1] + direction[1])
                if 0 <= next_pos[0] < maze.shape[0] and 0 <= next_pos[1] < maze.shape[1]:
                    if path_maze[next_pos] == 1 and next_pos not in visited:
                        visited.add(next_pos)
                        queue.append(next_pos)

        return False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.time_step = 0
        self.valid_step = 0
        self.invalid_step_count = 0
        self.current_pos = self.start_pos
        self.maze = np.copy(self.initial_maze)  # Reset the maze to its initial state
        return self._get_observation(), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Action must be an integer from 0 to 3.")

        # Save the previous position
        previous_pos = self.current_pos

        # Calculate the new position based on the action
        direction = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_pos = (self.current_pos[0] + direction[0], self.current_pos[1] + direction[1])

        # Initialize reward
        reward = 0
        valid_step = True

        # Check if the new position is within the bounds of the maze
        if 0 <= new_pos[0] < self.maze.shape[0] and 0 <= new_pos[1] < self.maze.shape[1]:
            # Check for blockage
            if self.maze[new_pos] == -10:
                reward = -10
                valid_step = False
            else:
                # Check for goal
                if new_pos == self.goal_pos:
                    reward = 10
                elif self.maze[new_pos] == 0:
                    reward = -2  # Starting position
                else:
                    reward = -1  # Legal step

                # Update the agent's position
                self.current_pos = new_pos
                self.maze[previous_pos] = -1  # Mark the previous position as visited
                self.maze[new_pos] = 0  # Mark the new position as current
        else:
            # Penalize if out of bounds or into a blockage
            reward = -10
            valid_step = False

        # Penalize for repeated illegal actions
        if not valid_step:
            self.invalid_step_count += 1
            if self.invalid_step_count >= 3:
                reward = -3  # Additional penalty for 3 consecutive invalid steps
                self.invalid_step_count = 0  # Reset the counter
        else:
            self.invalid_step_count = 0

        # Check for completion of the maze
        done = self.current_pos == self.goal_pos

        return self._get_observation(), reward, done, False, {}




    def _get_observation(self):
        # Flatten the maze to create a 1D observation space
        return self.maze.flatten().astype(np.int32)

    def render(self, mode='human'):
        render_maze = np.copy(self.maze)
        render_maze[self.current_pos] = 2  # Mark the agent's current position

        for row in render_maze:
            row_str = ''.join(['X' if x == -10 else 'O' if x == -1 else 'G' if x == 10 else 'S' for x in row])
            print(row_str)
        print('')
