# Import necessary libraries
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from CustomMazeEnv import MazeEnv

# Create the Maze Environment
input_maze = [
    [0, -1, -1, -1],
    [-10, -10, -1, -10],
    [-1, -1, -1, -10],
    [-1, -10, 10, -10]
]
env = MazeEnv(input_maze)

# Check if the environment follows the Gym interface
check_env(env, warn=True)

# Create a DQN agent
model = DQN("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Test the trained agent
obs,_ = env.reset()
steps=0
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done,terminated, info = env.step(action)
    env.render()
    steps+=1
    if done:
      print(f"Steps: {steps}")
      print(f"Reward: {reward}")
      print(f"Terminated: {terminated}")
      print(f"Info: {info}")
      obs,_ = env.reset()
      break
