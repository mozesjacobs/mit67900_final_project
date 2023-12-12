from stable_baselines3 import DQN

from custom_env import *

from utils import compute_average_reward
from gymnasium.envs.registration import register

register(
    id='CustomEnv',
    entry_point='custom_env:EnvironmentWrapper',
    max_episode_steps=300,
)

def main():
    env = gym.make("CustomEnv")

    model = DQN("MlpPolicy", env, verbose=0)

    avg_r = compute_average_reward(model, env)
    print(f'Average reward before training: {avg_r}')

    model.learn(total_timesteps=100000, log_interval=4, progress_bar=True)

    avg_r = compute_average_reward(model, env)
    print(f'Average reward after training: {avg_r}')
    return model

if __name__ == '__main__':
    main()
