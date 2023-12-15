# ChatGPT Code

import sys
sys.path.append("../")

import torch
import numpy as np
import yaml
import random
import gymnasium as gym
import imageio

from sb3_contrib import TRPO
from stable_baselines3 import PPO

from ddqn import Agent as DDQNAgent
from fittedq import Agent as FittedQAgent

CONFIG_DDQN = "config_ddqn.yaml"
CONFIG_FITTEDQ = "config_fittedq.yaml"
CONFIG_TRPO = "config_trpo.yaml"
CONFIG_PPO = "config_ppo.yaml"

CONFIGS = [CONFIG_DDQN, CONFIG_FITTEDQ, CONFIG_TRPO, CONFIG_PPO]
EXPS = ["../exp2/", "../exp62/", "../exp63/", "../exp64/"]
#EXPS = ["../exp63/", "../exp64/"]

num_episodes = 3  # Number of episodes to run

# Function to run an episode and capture frames
def run_episode(env, model):
    frames = []
    state, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(state)
        state, _, done, _, _ = env.step(action)
        frame = env.render() #env.render(mode='rgb_array')
        frames.append(frame)
    return frames

# Function to create a GIF from frames
def create_gif(frames, filename='output.gif', fps=30):
    imageio.mimsave(filename, frames, fps=fps)

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

def make_env(config):
    if "LunarLander" in config['env']:
        return gym.make(
            config['env'],
            gravity=config['gravity'],
            enable_wind=config['enable_wind'],
            wind_power=config['wind_power'],
            turbulence_power=config['turbulence_power'],
            render_mode='rgb_array'
        )
    return gym.maked(config['env'], render_mode='rgb_array')

# Load config settings
configs = []
for exp in EXPS:
    curr_configs = []
    for config in CONFIGS:
        with open(exp + config, 'r') as f:
            curr_configs.append(yaml.safe_load(f))
    configs.append(curr_configs)

# Setup (choose seed randomly)
set_seed(4335)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Results
results = []
for i in range(len(configs)):
#for i in range(3, len(configs)):
    print(i)
    exp = configs[i]
    for j in range(len(exp)):
        print(j)
        config = exp[j]
        # Make environment
        env = make_env(config)
        state_dim = np.prod(env.observation_space.shape)
        action_dim = env.action_space.n
        agent = load_agent(config, env, EXPS[i], state_dim, action_dim, device)
        frames = run_episode(env, model=agent)  # Replace with your actual model
        fname = "results/gifs/gravity_" + str(i) + "_" + str(j) + ".gif"
        create_gif(frames, filename=fname, fps=30)
        env.close()

# Run episodes and capture frames
#all_frames = []
#for _ in range(num_episodes):
#    
#    all_frames.extend(frames)

# Create and save the GIF
##create_gif(all_frames, filename='output.gif', fps=30)

# Close the environment
#env.close()