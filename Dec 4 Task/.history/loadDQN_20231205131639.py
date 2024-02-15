# Import necessary libraries
import gymnasium as gym
from stable_baselines3 import DQN
from CustomMazeEnv import MazeEnv


# Create the Maze Environment
input_maze = [
    [0, -1, -1, -1,-10],
    [-10, -10, -1, -10,-10],
    [-1, -1, -1, -10,-10],
    [-1, -10, 10, -10,-10],
    [-1,-10,-1,-1,-10]
]
# Create a new instance of the Maze environment
new_env = MazeEnv(input_maze)  # Assuming 'input_maze' is defined or imported

# Load the trained model
model = DQN.load("/home/arsalan/Desktop/Reinforcement Learning/Dec 4 Task/my_model")


# Test the loaded model
obs,_ = new_env.reset()
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, terminated, info = new_env.step(action)
    new_env.render()
    if done:
        print(f"Reward: {reward}, Terminated: {terminated}, Info: {info}")
        break
