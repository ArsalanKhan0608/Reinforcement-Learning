import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
# Register the custom environment
gym.register(id='MazeGame-v0',entry_point='mazegame:MazeGameEnv',kwargs={'maze':None})
# Define the maze layout
maze= [
    ['S','#','.','G'],
    ['.','#','.','#'],
    ['.','.','.','.'],
    ['#',".","#",'.']
]


# Create the environment
env = gym.make('MazeGame-v0', maze=maze)
env = Monitor(env)
print(f"Environment created: {env}")

# Create callback
# eval_callback = EvalCallback(env, best_model_save_path='./logs/',
#                              log_path='./logs/', eval_freq=500,
#                              deterministic=True, render=False)

model = DQN('MlpPolicy', env, verbose=0,
            buffer_size=1000,  # size of the replay buffer
            learning_rate=0.005,  # learning rate
            gamma=0.9,  # discount factor
            exploration_fraction=0.1,  # exploration vs exploitation trade-off
            exploration_initial_eps=1.0,  # initial exploration rate
            exploration_final_eps=0.02,  # final exploration rate
            train_freq=50,  # update the model every 4 steps
            gradient_steps=1,  # how many gradient steps after each update
            target_update_interval=100,  # update the target network every 1000 steps
            learning_starts=500,  # start learning after 2000 steps
            tau=1.0,  # the soft update coefficient
            batch_size=64,  # size of a batched sampled from replay buffer for training
            policy_kwargs=dict(net_arch=[64, 64]))  # size of the neural network for the policy


print(f"Model defined: {model}")

model.learn(total_timesteps=30000)  # Train for a specific number of timesteps
print("Model training completed.")

# # Evaluate the trained model
# # mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1)  # Evaluate the model
# # print(f"Mean reward: {mean_reward}")

# # Use the trained model to play the game or make predictions
# obs = env.reset()
# print(f"Initial observation: {obs}")
# print("Obs shape before reshaping",obs[0].shape)
# print("Obs shape Type",type(obs[0]))
# print("Value in obs:",obs[0])
# # obs = np.array(obs[0]).reshape(2,).astype(np.float32)  # Convert the data type of the ndarray
# obs=obs[0]
# step=0
# while True:
#     action, _ = model.predict(obs, deterministic=True)
#     print(f"Selected action: {action}")
#     obs, reward, done, terminated,info = env.step(action)
#     print(f"New observation: {obs}, Reward: {reward}, Done: {done}")
#     step+=1
#     if done:
#         obs = env.reset()
#         print(f"Total Steps: {step}")
#         print("Resetting environment.")
#         break
