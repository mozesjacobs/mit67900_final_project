import gymnasium as gym
import torch
import numpy as np

from sb3_contrib import TRPO
# from algorithms.trpo import PolicyNetwork
from gymnasium.envs.registration import register
from custom_env import *
from utils import compute_average_reward, CustomCallback

register(
    id='CustomEnv',
    entry_point='custom_env:EnvironmentWrapper',
    max_episode_steps=300,
)


# Code adapted from:
# https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html

def main():
    # Setup
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Make environment
    # env = gym.make('LunarLander-v2')
    env = gym.make("CustomEnv",board_kwargs={'Enemy_number':8})
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n

    # Parameters

    lr = 0.001
    gamma = 0.9
    n_steps = 2048
    total_timesteps = 200000
    line_search_max_iter = 10
    gae_lambda = 0.95
    batch_size = 128
    policy_kwargs = {'net_arch':[30]}

    agent = TRPO(
        'MlpPolicy',
        env,
        learning_rate=lr,
        n_steps=n_steps,
        gamma=gamma,
        line_search_max_iter=line_search_max_iter,
        gae_lambda=gae_lambda,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log="../logs/"
    )
    # Build agent
    avg_r = compute_average_reward(agent, env)
    print(f'Average reward before training: {avg_r}')
    # Train
    callback = CustomCallback(agent)
    agent.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)
    callback_info = callback.get_info()
    rewards = callback_info['episode_rewards']
    pos_rewards = callback_info['positive_rewards_count']
    avg_r = compute_average_reward(agent, env)
    print(f'Average reward after training: {avg_r}')
    return agent, rewards, pos_rewards

if __name__ == "__main__":
    main()