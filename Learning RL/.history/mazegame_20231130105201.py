import gymnasium as gym
import numpy as np
from gymnasium import spaces
class MazeGameEnv(gym.Env):
    def __init__(self,maze):
        self.maze=np.array(maze) #maze represented as a 2D numpy array
        print("Maze",self.maze)
        self.start_pos=np.where(self.maze=='S')
        print("Start Position",self.start_pos)
        print("Start Position Type",type(self.start_pos))
        self.goal_pos=np.where(self.maze=='G') # Goal position
        print("Goal Position:",self.goal_pos)
        print("Goal Position Type",type(self.goal_pos))
        self.current_pos=self.start_pos # Starting position is the current position initially
        print("Current Position:",self.current_pos)
        print("Current Position Type",type(self.current_pos))
        self.num_rows,self.num_cols=self.maze.shape
        print(f"num rows: {self.num_rows} and num columns: {self.num_cols}")
        #4 possible actions, 0=up, 1=down,2=left,3=right
        self.action_space=spaces.Discrete(4)
        print(f"Action Space: {self.action_space}")
        
        # observation_space is a grid of size: rowsxcolumns
        low = np.array([0, 0])  # lower-bound values
        high = np.array([self.num_rows - 1, self.num_cols - 1])  # upper-bound values
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)
        print(f"Observation Space: {self.observation_space}")
        self.total_reward = 0

        self.cell_size=125
        print("Current position in array form: ",np.array(self.current_pos, dtype=np.int32).flatten())
        print("Current position in array form Shape: ",np.array(self.current_pos, dtype=np.int32).flatten().shape)
        

    
    
    def reset(self, **kwargs):
        self.total_reward = 0
        self.current_pos = self.start_pos
        return np.array(self.current_pos, dtype=np.int32).flatten(), {}
    
    def step(self,action):
        #Move the agent based on the selected action
        new_pos=np.array(self.current_pos)
        print("New Position in Step:", new_pos)
        print("New Position in Step Shape:", new_pos.shape)
        if action==0: #Up
            new_pos[0] -=1

        elif action==1:
            new_pos[0] +=1

        elif action==2: # Left
            new_pos[1]-=1

        elif action==3: #Right
            new_pos[1]+=1

        #Check if the new position is valid
        if self.is_valid_position(new_pos):
            self.current_pos=new_pos
            
        #Reward Function
        reward = 0.0  # Initialize reward
        if np.array_equal(self.current_pos, self.goal_pos):
            reward = 5.0
            done = True
        else:
            cell_value = self.maze[tuple(self.current_pos)]
            print("Cell Value: ",type(self.maze[tuple(self.current_pos)]))
            if cell_value == '-1':
                reward -= 1.0
            elif cell_value == '-5':
                reward -= 5.0
            else:
                reward = 0.0
            done = False
        self.total_reward += reward
        
        if done:
            print(self.total_reward)
        return np.array(self.current_pos, dtype=np.int32).flatten(),reward,done,False, {}


    
    def is_valid_position(self, pos):
        # sourcery skip: assign-if-exp, boolean-if-exp-identity, reintroduce-else, remove-unnecessary-cast
        row,col=pos
        # if agent goes out of the grid
        if row<0 or col<0 or row>=self.num_rows or col>=self.num_cols:
            return False
        #if the agent hits an obstacle
        if self.maze[row,col]=='#':
            return False
        
        return True
    
