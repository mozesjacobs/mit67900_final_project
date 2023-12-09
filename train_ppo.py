import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import base64, io
import numpy as np
from collections import deque, namedtuple
from IPython.display import HTML
from IPython import display 
import glob

from algorithms.ppo import episode, Agent
from trainer import train_ppo as train

# Code adapted from:
# https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html

def main():
    # Setup
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Make environment
    env = gym.make('LunarLander-v2')
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n

    # Parameters
    lr = 1e-3
    gamma = 0.99
    hidden_dim = 64
    max_episodes = 2000
    max_t = 500
    epsilon = 0.2
    beta = 1.0
    delta = 0.01
    c1 = 0.5
    c2 = 0.01
    k_epoch = 40
    batch_size = 1600

    # Build agent
    agent = Agent(gamma, epsilon, beta, delta, c1, c2, k_epoch, 
                  state_dim, action_dim, lr, lr, hidden_dim, batch_size)
    
    # Train
    scores = train(agent, env, episode, 'ppo', max_episodes, max_t)


if __name__ == "__main__":
    main()