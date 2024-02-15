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

# Define and Initialize the DQN model
model = DQN('MlpPolicy', env, verbose=1,
            buffer_size=5000,  # Increase the size of the replay buffer
            learning_rate=0.005,  # Slightly increase learning rate
            gamma=0.95,  # Adjust discount factor
            exploration_fraction=0.2,  # Increase exploration fraction
            exploration_initial_eps=1.0,  # Initial exploration rate
            exploration_final_eps=0.1,  # Increase final exploration rate
            train_freq=4,  # Update the model more frequently
            gradient_steps=1,  # Number of gradient steps after each update
            target_update_interval=500,  # Update the target network more frequently
            learning_starts=1000,  # Start learning after more steps
            tau=1.0,  # Soft update coefficient
            batch_size=128,  # Increase batch size
            policy_kwargs=dict(net_arch=[128, 128]))  # Larger neural network

# Train the agent
model.learn(total_timesteps=20000)

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
