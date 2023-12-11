import gymnasium as gym
import torch
import numpy as np

from sb3_contrib import TRPO
# from algorithms.trpo import PolicyNetwork


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
    env = gym.make("MountainCar-v0")
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n

    # Parameters

    lr = 0.01
    gamma = 0.99
    # hidden_dim = 64
    n_steps = 2048
    total_timesteps = 100000
    line_search_max_iter = 10
    gae_lambda = 0.95
    batch_size = 128
    policy_kwargs = {'net_arch':[64, 64]}

    # policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
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
        verbose=1
    )
    # Build agent
    avg_r = compute_average_reward(agent, env)
    print(f'Average reward before training: {avg_r}')
    # Train
    scores = agent.learn(total_timesteps=total_timesteps, progress_bar=True)
    avg_r = compute_average_reward(agent, env)
    print(f'Average reward after training: {avg_r}')
    return agent

def compute_average_reward(model, env, num_runs=100):
    all_rewards = []

    for _ in range(num_runs):
        obs, _ = env.reset()
        total_reward = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        all_rewards.append(total_reward)

    average_reward = np.mean(all_rewards)
    return average_reward

if __name__ == "__main__":
    main()