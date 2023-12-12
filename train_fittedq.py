import torch
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

from algorithms.fittedq import Agent
from trainer import train
from custom_env import EnvironmentWrapper
from utils import compute_average_reward

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Make environment
    #env = gym.make('LunarLander-v2')
    env = gym.make('CustomEnv')
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n

    # Parameters
    lr = 1e-2
    buffer_size = int(1e5)
    batch_size = 64
    gamma = 0.99            
    tau = 1e-3             
    update_interval = 4 
    total_time = 1000000
    run_length = 1000
    window_length = 100
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995

    # Build agent
    agent = Agent(state_dim, action_dim, seed, buffer_size, batch_size, 
                  update_interval, gamma, device, lr, tau)
    
    # Train
    scores, t = train(agent, env, 'trained_models/fittedq', total_time, run_length, window_length, eps_start, eps_end, eps_decay)

if __name__ == "__main__":
    main()