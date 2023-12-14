import torch
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import yaml
import random

from ddqn import Agent
from trainer import train
from utils import compute_average_reward, make_env


# Code adapted from:
# https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html

CONFIG_PATH = "config_ddqn.yaml"

def main():
    # Load config settings
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f) 

    # Setup
    set_seed(config['seed'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Make environment
    env = make_env(config)
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n

    # Build agent
    if 'load_model' in config and config['load_model']:
        agent = load_agent(config, state_dim, action_dim, device)
    else:
        agent = build_agent(config, state_dim, action_dim, device)

    # Train
    scores, t = train(agent, env, config['modelfile'], config['logfile'], config['total_time'],
        config['run_length'], config['window_length'], config['eps_start'],
        config['eps_end'], config['eps_decay'], config['reward_threshold'])
    
    np.save(config['scorefile'], scores)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def build_agent(config, state_dim, action_dim, device):
    return Agent(state_dim, action_dim, int(config['buffer_size']),
                 config['batch_size'], config['update_interval'],
                 config['gamma'], device, config['lr'], config['tau'])

def load_agent(config, state_dim, action_dim, device):
    agent = build_agent(config, state_dim, action_dim, device)
    return agent.load(config['loadfile'])

if __name__ == "__main__":
    main()