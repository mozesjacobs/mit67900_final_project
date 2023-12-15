import torch

from stable_baselines3 import PPO
from gymnasium.envs.registration import register
from custom_env import *
from utils import compute_average_reward, CustomCallback

# Code adapted from:
# https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html

register(
    id='CustomEnv',
    entry_point='custom_env:EnvironmentWrapper',
    max_episode_steps=300,
)

def main():
    # Setup
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Make environment
    env = gym.make('CustomEnv', board_kwargs={'Enemy_number':8})
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n

    # Parameters
    lr = 0.001
    gamma = 0.9
    n_steps = 2048
    total_timesteps = 200000
    gae_lambda = 0.95
    batch_size = 128
    policy_kwargs = {'net_arch': [30]}

    # Build agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        gamma=gamma,
        n_steps=n_steps,
        gae_lambda=gae_lambda,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs
    )

    avg_r = compute_average_reward(model, env)
    print(f'Average reward, avg. number of positive rewards before training: {avg_r}')
    callback = CustomCallback(model)
    # Train
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    callback_info = callback.get_info()
    rewards = callback_info['episode_rewards']
    pos_rewards = callback_info['positive_rewards_count']

    avg_r = compute_average_reward(model, env)
    print(f'Average reward, avg. number of positive rewards after training: {avg_r}')

    return model, rewards, pos_rewards

if __name__ == "__main__":
    main()