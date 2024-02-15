import gymnasium as gym
import pygame
import numpy as np
from gymnasium import spaces
class MazeGameEnv(gym.Env):
    def __init__(self,maze):
        self.maze=np.array(maze) #maze represented as a 2D numpy array
        print(self.maze)
        self.start_pos=np.where(self.maze=='S')
        print(self.start_pos)
        self.goal_pos=np.where(self.maze=='G') # Goal position
        print(self.goal_pos)
        self.current_pos=self.start_pos # Starting position is the current position initially
        print(self.current_pos)
        self.num_rows,self.num_cols=self.maze.shape
        print(f"num rows: {self.num_rows} and num columns: {self.num_cols}")
        #4 possible actions, 0=up, 1=down,2=left,3=right
        self.action_space=spaces.Discrete(4)
        print(f"Current Action: {self.action_space}")
        
        # observation_space is a grid of size: rowsxcolumns
        self.observation_space=spaces.Tuple((spaces.Discrete(self.num_rows),spaces.Discrete(self.num_cols)))
        print(f"Observation Space: {self.observation_space}")
        
        #initialize the pygame
        pygame.init()
        self.cell_size=125
        
        #setting Display Size
        self.screen=pygame.display.set_mode((self.num_cols*self.cell_size, self.num_rows*self.cell_size))
    
    
    def reset(self):
        self.current_pos=self.start_pos
        return self.current_pos
    
    def step(self,action):
        #Move the agent based on the selected action
        new_pos=np.array(self.current_pos)
        
        if action==0: #Up
            new_pos[0] -=1
            print(f"new position after UP: {new_pos}")
        elif action==1:
            new_pos[0] +=1
            print(f"new position after down: {new_pos}")
            
        
        elif action==2: # Left
            new_pos[1]-=1
            print(f"new position after Left: {new_pos}")
            
        
        elif action==3: #Right
            new_pos[1]+=1
            print(f"new position after Right: {new_pos[0]}")
            
        
        #Check if the new position is valid
        if self.is_valid_position(new_pos):
            self.current_pos=new_pos
            
        #Reward Function
        if np.array_equal(self.current_pos,self.goal_pos):
            reward=1.0
            done=True
        else:
            reward=0.0
            done=False
        return self.current_pos,reward,done, {}
    
    def is_valid_position(self, pos):
        row,col=pos
        # if agent goes out of the grid
        if row<0 or col<0 or row>=self.num_rows or col>=self.num_cols:
            return False
        
        #if the agent hits an obstacle
        if self.maze[row,col]=='#':
            return False
        
        return True
    
    def render(self):
        #clear the screen
        self.screen.fill((255,255,255))
        
        # Draw env elements one cell at a time
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left=col*self.cell_size
                cell_top=row*self.cell_size
                
                try:
                    print(np.array(self.current_pos)==np.array([row,col]).reshape(-1,1))
                except Exception as e:
                    print('Initial State')
                    
                if self.maze[row,col]=='#': #Obstacle
                    pygame.draw.rect(self.screen,(0,0,0),(cell_left,cell_top,self.cell_size,self.cell_size))
                
                elif self.maze[row,col]=='S': # Starting postion
                    pygame.draw.rect(self.screen,(0,255,0),(cell_left,cell_top,self.cell_size,self.cell_size))
                
                elif self.maze[row,col]=='G': # Goal postion
                    pygame.draw.rect(self.screen,(0,0,255),(cell_left,cell_top,self.cell_size,self.cell_size))
        pygame.display.update() #update the display