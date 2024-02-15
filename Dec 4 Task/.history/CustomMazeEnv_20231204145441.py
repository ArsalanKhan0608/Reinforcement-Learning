from collections import deque
import numpy as np

class MazeEnv:
    def __init__(self,input_maze):
        #validate the input maze
        self.maze=self._validate_maze(input_maze)
        self.start_pos=tuple(np.argwhere(self.maze==0)[0])
        self.goal_pos=tuple(np.argwhere(self.maze==10)[0])

    def _validate_maze(self, maze):
        maze=np.array(maze)

        #check if the maze is square
        if maze.shape[0]!=maze.shape[1]:
            raise ValueError("Maze must be square-shaped.")

        #check for integer values
        if not issubclass(maze.dtype.type,np.integer):
            raise ValueError("Maze values must be integers.")
        
        #check for one start and one goal
        if np.sum(maze==0)!=1 or np.sum(maze==10)!=1:
            print("Maze must contain exactly one start position (0) and one goal position (10).")

        #check for a valid path
        if not self.is_path_available(maze):
            raise ValueError("No Valid path from start to goal.")
        
        return maze
    
    def is_path_available(self, maze):
        # Marking obstacles and paths in the maze
        path_maze=np.where(maze==-10,0,1)

        start=tuple(np.argwhere(path_maze==1)[0])
        goal=tuple(np.argwhere(maze==10)[0])
        visited=set()
        queue=deque([start])
        while queue:
            current=queue.popleft()
            if current==goal:
                return True
