import torch
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import yaml
import random

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from utils import compute_average_reward, make_env


# Code adapted from:
# https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html

CONFIG_PATH = "config_ppo.yaml"

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
    agent = build_agent(config, env)
    if 'load_model' in config and config['load_model']:
        agent.set_parameters(config['loadfile'])

    # Stop training when the model reaches the reward threshold
    eval_env = make_env(config)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=config['reward_threshold'], verbose=1)
    end_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1)

    with open(config['logfile'], "w") as f:
        avg_r = compute_average_reward(agent, env)
        print(f'Average reward before training: {avg_r}', file=f)

        # Train
        #agent.learn(total_timesteps=config['total_time'], progress_bar=True, callback=end_callback)
        agent.learn(total_timesteps=config['total_time'], progress_bar=True)

        avg_r = compute_average_reward(agent, env)
        print(f'Average reward after training: {avg_r}', file=f)

    agent.save(config['modelfile'])
    #np.save(config['scorefile'], scores)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def build_agent(config, env):
    return PPO(
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


if __name__ == "__main__":
    main()