import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
# Register the custom environment
gym.register(id='MazeGame-v0',entry_point='mazegame:ModifiedMazeGameEnv',kwargs={'maze':None})
# Define the maze layout


maze= [[ 0, -1, -1, -1 ],
    [-1, -10, -1, -10],
    [-1, -1, -1, -1 ],
    [-10, -1, -10, 10]
]


# Create the environment
env = gym.make('MazeGame-v0', maze=maze)
env = Monitor(env)
print(f"Environment created: {env}")

# Define and Initialize the DQN model
model = A2C(
    policy='MlpPolicy',
    env=env,
    learning_rate=0.001,
    # n_steps=5,
    # gamma=0.99,
    # gae_lambda=1,
    # ent_coef=0.1,  # Encourage exploration
    # vf_coef=0.5,
    # max_grad_norm=0.5,
    # rms_prop_eps=0.00001,
    # use_rms_prop=True,
    # use_sde=False,
    # sde_sample_freq=-1,
    # normalize_advantage=False,
    # stats_window_size=100,
    verbose=0,  # For debugging
    # seed=None,
    # device="auto",
    # _init_setup_model=True
)




print(f"Model defined: {model}")

model.learn(total_timesteps=20000)  # Train for a specific number of timesteps
print("Model training completed.")

# Evaluate the trained model
# mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1)  # Evaluate the model
# print(f"Mean reward: {mean_reward}")

# # Use the trained model to play the game or make predictions
obs = env.reset()
print(f"Initial observation: {obs}")
print("Obs shape before reshaping",obs[0].shape)
print("Obs shape Type",type(obs[0]))
print("Value in obs:",obs[0])
# obs = np.array(obs[0]).reshape(2,).astype(np.float32)  # Convert the data type of the ndarray
obs=obs[0]
step=0
while True:
    action, _ = model.predict(obs, deterministic=True)
    print(f"Selected action: {action}")
    obs, reward, done, terminated,info = env.step(action)
    print(f"New observation: {obs}, Reward: {reward}, Done: {done}")
    step+=1
    if done:
        obs = env.reset()
        print(f"Total Steps: {step}")
        print("Resetting environment.")
        break
