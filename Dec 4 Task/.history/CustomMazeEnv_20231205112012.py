from collections import deque
import numpy as np

# Running the provided MazeEnv code with a test maze
class MazeEnv:
    def __init__(self, input_maze):
        super(MazeEnv, self).__init__()

        self.maze = self._validate_maze(input_maze)
        self.start_pos = tuple(np.argwhere(self.maze == 0)[0])
        self.goal_pos = tuple(np.argwhere(self.maze == 10)[0])
        self.current_pos = self.start_pos

    def _validate_maze(self, maze):
        maze = np.array(maze)

        if maze.shape[0] != maze.shape[1]:
            raise ValueError("Maze must be square-shaped.")

        if not issubclass(maze.dtype.type, np.integer):
            raise ValueError("Maze values must be integers.")
        
        if np.sum(maze == 0) != 1 or np.sum(maze == 10) != 1:
            raise ValueError("Maze must contain exactly one start position (0) and one goal position (10).")

        if not self.is_path_available(maze):
            raise ValueError("No valid path from start to goal.")
        
        return maze
    
    def is_path_available(self, maze):
        path_maze = np.where(maze == -10, 0, 1)
        start = tuple(np.argwhere(maze == 0)[0])
        goal = tuple(np.argwhere(maze == 10)[0])
        visited = set()
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current == goal:
                return True
            visited.add(current)

            for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # Right, Left, Down, Up
                next_pos = (current[0] + direction[0], current[1] + direction[1])
                if 0 <= next_pos[0] < maze.shape[0] and 0 <= next_pos[1] < maze.shape[1]:
                    if path_maze[next_pos] == 1 and next_pos not in visited:
                        queue.append(next_pos)

        return False

    def reset(self):
        self.current_pos = self.start_pos
        self.maze[self.start_pos] = 0  # Reset start position value
        return self._get_observation()

    def step(self, action):
        # Save the previous position
        previous_pos = self.current_pos

        # Calculate the new position based on the action
        direction = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        print(f'Direction: {direction}')
        new_pos = (self.current_pos[0] + direction[0], self.current_pos[1] + direction[1])

        # Check if the new position is valid and update current position
        if 0 <= new_pos[0] < self.maze.shape[0] and 0 <= new_pos[1] < self.maze.shape[1]:
            if self.maze[new_pos] != -10:  
                self.current_pos = new_pos
                # Reset the previous position to path (-1)
                self.maze[previous_pos] = -1
                # Update the reward based on the new position
                reward = self.maze[new_pos]
                # Mark the new position as current
                self.maze[new_pos] = 0
            else:
                # Penalize if trying to move into a blockage
                reward = -10
        else:
            # Penalize if out of bounds
            reward = -10

        done = self.current_pos == self.goal_pos
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        # Flatten the maze to create a 1D observation space
        print(f"Observation: {self.maze}")
        return self.maze.flatten()

    def render(self, mode='human'):
        render_maze = np.copy(self.maze)
        render_maze[self.current_pos] = 2  # Mark the agent's current position

        for row in render_maze:
            row_str = ''.join(['X' if x == -10 else 'O' if x == -1 else 'G' if x == 10 else 'S' for x in row])
            print(row_str)
        print('')

# Test maze
input_maze = [
    [ 0, -1, -1, -1],
    [-10, -10, -1, -10],
    [-1, -1, -1, -10],
    [-1, -10, 10, -10]
]

# Creating and testing the MazeEnv
try:
    env = MazeEnv(input_maze)
    env.reset()
    env.render()
    for _ in range(10):
        obs, reward, done, _ = env.step(np.random.randint(0, 4))  # Taking random actions
        env.render()
        if done:
            print("Goal Reached!")
            break
except ValueError as e:
    print(f"Error: {e}")

