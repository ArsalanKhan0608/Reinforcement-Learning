# Import necessary libraries
import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from CustomMazeEnv import MazeEnv

# Create the Maze Environment
input_maze = [
    [0, -1, -1, -1,-10],
    [-10, -10, -1, -10,-10],
    [-1, -1, -1, -10,-10],
    [-1, -10, 10, -10,-10],
    [-1,-10,-1,-1,-10]
]
env = MazeEnv(input_maze)

# Check if the environment follows the Gym interface
check_env(env, warn=True)

# Define and Initialize the DQN model
model = A2C('MlpPolicy', env, verbose=1,
            learning_rate=0.005,  # Slightly increase learning rate
            gamma=0.95,  # Adjust discount factor
            n_steps=5,  # Number of steps to run for each environment per update
            vf_coef=0.5,  # Value function coefficient in the loss calculation
            ent_coef=0.01,  # Entropy coefficient for exploration
            max_grad_norm=0.5,  # Maximum gradient norm
            use_rms_prop=True,  # Use RMSProp optimizer
            use_tf32=False,  # Use mixed precision training
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
