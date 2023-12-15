import sys
sys.path.append("../")

import torch
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import yaml
import random
from sb3_contrib import TRPO
from stable_baselines3 import PPO

from ddqn import Agent as DDQNAgent
from fittedq import Agent as FittedQAgent
from trainer import train
from utils import compute_average_reward, make_env


# Code adapted from:
# https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html

CONFIG_DDQN = "config_ddqn.yaml"
CONFIG_FITTEDQ = "config_fittedq.yaml"
CONFIG_TRPO = "config_trpo.yaml"
CONFIG_PPO = "config_ppo.yaml"

# exp1 - gravity = 10
# exp2 - gravity = 5
# exp6 - pretrained gravity = 5, train gravity = 10
# exp7 - pretrained gravity = 10, train gravity = 5

CONFIGS = [CONFIG_DDQN, CONFIG_FITTEDQ, CONFIG_TRPO, CONFIG_PPO]
EXPS = ["../exp1/", "../exp2/", "../exp6/", "../exp7/"]

def main():
    # Load config settings
    configs = []
    for exp in EXPS:
        curr_configs = []
        for config in CONFIGS:
            with open(exp + config, 'r') as f:
                curr_configs.append(yaml.safe_load(f))
        configs.append(curr_configs)
    
    # Setup (choose seed randomly)
    set_seed(4235)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Results
    results = []
    for i in range(len(configs)):
        print(i)
        exp = configs[i]
        for j in range(len(exp)):
            config = exp[j]
            # Make environment
            env = make_env(config)
            state_dim = np.prod(env.observation_space.shape)
            action_dim = env.action_space.n
            agent = load_agent(config, env, EXPS[i], state_dim, action_dim, device)
            avg_reward = compute_average_reward(agent, env, num_runs=config['window_length'], nb_pos_r=False)
            results.append(avg_reward)
    results = np.save("results/eval1.npy", np.array(results))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def load_agent(config, env, exp_name, state_dim, action_dim, device):
    if 'ddqn' in config['logfile']:
        agent = DDQNAgent(state_dim, action_dim, int(config['buffer_size']),
                    config['batch_size'], config['update_interval'],
                    config['gamma'], device, config['lr'], config['tau'])
        return agent.load(exp_name + config['modelfile'])
    elif 'fittedq' in config['logfile']:
        agent = FittedQAgent(state_dim, action_dim, int(config['buffer_size']),
                    config['batch_size'], config['update_interval'],
                    config['gamma'], device, config['lr'], config['tau'])
        return agent.load(exp_name + config['modelfile'])
    elif 'trpo' in config['logfile']:
        agent = TRPO(
            config['policy'],
            env,
            learning_rate=config['lr'],
            n_steps=config['run_length'],
            gamma=config['gamma'],
            line_search_max_iter=config['line_search_max_iter'],
            gae_lambda=config['gae_lambda'],
            batch_size=config['batch_size'],
            policy_kwargs={'net_arch':config['net_arch']},
            stats_window_size=config['window_length'],
            seed=config['seed'],
            verbose=0
        )
    else:
        agent = PPO(
            config['policy'],
            env,
            learning_rate=config['lr'],
            n_steps=config['run_length'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_range=config['clip_range'],
            batch_size=config['batch_size'],
            policy_kwargs={'net_arch':config['net_arch']},
            stats_window_size=config['window_length'],
            seed=config['seed'],
            verbose=0
        )
    agent.set_parameters(exp_name + config['modelfile'])
    return agent

if __name__ == "__main__":
    main()