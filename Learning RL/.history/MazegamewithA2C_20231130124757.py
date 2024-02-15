import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
# Register the custom environment
gym.register(id='MazeGame-v0',entry_point='mazegame:MazeGameEnv',kwargs={'maze':None})
# Define the maze layout
maze= [
    ['S','.','.','.'],
    ['.','#','.','#'],
    ['.','.','.','.'],
    ['#',".","#",'G']
]


# Create the environment
env = gym.make('MazeGame-v0', maze=maze)
env = Monitor(env)
print(f"Environment created: {env}")

# Define and Initialize the DQN model
model = A2C('MlpPolicy', env, verbose=1,
            learning_rate=0.0007,  # learning rate
            gamma=0.99,  # discount factor
            gae_lambda=0.95,  # factor for trade-off of bias vs variance for Generalized Advantage Estimator
            ent_coef=0.0,  # entropy coefficient for exploration
            vf_coef=0.5,  # value function coefficient
            max_grad_norm=0.5,  # the maximum value for the gradient clipping
            use_rms_prop=False,  # using RMSprop optimizer (if False, uses Adam optimizer)
            use_sde=False,  # whether to use State Dependent Exploration
            sde_sample_freq=-1,  # sample a new noise matrix every n steps when using gSDE
            normalize_advantage=False,  # whether to normalize the advantage
            policy_kwargs=dict(net_arch=[64, 64]))  # size of the neural network for the policy
  
print(f"Model defined: {model}")

model.learn(total_timesteps=10000)  # Train for a specific number of timesteps
print("Model training completed.")

# Evaluate the trained model
# mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1)  # Evaluate the model
# print(f"Mean reward: {mean_reward}")

# Use the trained model to play the game or make predictions
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
