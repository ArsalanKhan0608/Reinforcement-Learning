# Import necessary libraries
import gymnasium as gym
from stable_baselines3 import DQN
from CustomMazeEnv import MazeEnv

# Create a new instance of the Maze environment
new_env = MazeEnv(input_maze)  # Assuming 'input_maze' is defined or imported

# Load the trained model
model = DQN.load("path/to/save/model", env=new_env)

# Test the loaded model
obs = new_env.reset()
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, terminated, info = new_env.step(action)
    new_env.render()
    if done:
        print(f"Reward: {reward}, Terminated: {terminated}, Info: {info}")
        break
