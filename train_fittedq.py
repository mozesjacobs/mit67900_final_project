import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import base64, io
import numpy as np
from IPython.display import HTML
from IPython import display 
import glob

from algorithms.fittedq import episode, Agent
from trainer import train

# Code adapted from:
# https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html

def main():
    # Setup
    seed = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Make environment
    env = gym.make('LunarLander-v2')
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
    max_episodes = 2000
    max_t = 1000
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995

    # Build agent
    agent = Agent(state_dim, action_dim, seed, buffer_size, batch_size, 
                  update_interval, gamma, device, lr, tau)
    
    # Train
    scores = train(agent, env, episode, 'fittedq', max_episodes, max_t, eps_start, eps_end, eps_decay)

if __name__ == "__main__":
    main()