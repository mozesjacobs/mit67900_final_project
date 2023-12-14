import sys
import torch.nn as nn
import numpy as np
from collections import deque, namedtuple
from utils import compute_average_reward

def train(agent, env, model_path, log_path, total_time=1000000, run_length=1000, window_length=100, eps=1.0, eps_end=0.01, eps_decay=0.995, reward_threshold=200):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    stdout_before = sys.stdout
    sys.stdout = open(log_path, 'w')

    scores = []                        # list containing scores for each run
    scores_window = deque(maxlen=window_length)
    state, _ = env.reset()
    score = 0
    t = 0
    run_t = 0
    num_runs = 0
    run_ended = False
    while t < total_time:
        # Step in current trajectory
        action = agent.act(state, eps)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.step(state, action, reward, next_state, terminated or truncated)
        state = next_state
        score += reward

        # Finished current run
        t += 1
        run_t += 1
        if terminated or truncated or (run_t == run_length):
            scores.append(score)
            scores_window.append(score)
            eps = max(eps_end, eps_decay*eps)
            state, _ = env.reset()
            score = 0
            run_t = 0
            num_runs += 1
            run_ended = True
        
        # Progress
        if run_ended and (num_runs % 100 == 0):
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(num_runs, np.mean(scores_window)))
            run_ended = False
        if np.mean(scores_window)>=reward_threshold:
            avg_reward = compute_average_reward(agent, env, num_runs=window_length, nb_pos_r=False)
            if avg_reward >= reward_threshold:
                print('\nEnvironment solved in {:d} timesteps!\tAverage Score: {:.2f}'.format(t, avg_reward))
                #print('\nEnvironment solved in {:d} timesteps!\tAverage Score: {:.2f}'.format(t, np.mean(scores_window)))
                #agent.save(model_path)
                break
        sys.stdout.flush()

    agent.save(model_path)
    print(f"Total Timesteps: {t}")
    sys.stdout.close()
    sys.stdout = stdout_before
    return scores, t