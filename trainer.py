import torch
import torch.nn as nn
import numpy as np
from collections import deque, namedtuple

def train(agent, env, fpath, total_time=1000000, run_length=1000, window_length=100, eps=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
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
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward

        # Finished current run
        t += 1
        run_t += 1
        if done or (run_t == run_length):
            scores.append(score)
            scores_window.append(score)
            eps = max(eps_end, eps_decay*eps)
            state, _ = env.reset()
            score = 0
            run_t = 0
            num_runs += 1
            run_ended = True
        
        # Progress
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(num_runs, np.mean(scores_window)), end="")
        if run_ended and (num_runs % 100 == 0):
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(num_runs, np.mean(scores_window)))
            run_ended = False
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} timesteps!\tAverage Score: {:.2f}'.format(total_time, np.mean(scores_window)))
            agent.save(fpath)
            break
    return scores, t