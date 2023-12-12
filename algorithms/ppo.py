import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
from collections import deque, namedtuple

def episode(agent, env, max_t, count):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    state, _ = env.reset()
    score = 0
    for t in range(max_t):
        count += 1
        action, lp = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.step([state, action, reward, lp, done])
        agent.learn(count)
        state = next_state
        score += reward
        if done:
            break
    return score, count


# Proximal Policy Optimization
class Agent():
    
    def __init__(self, γ, ϵ, β, δ, c1, c2, k_epoch, obs_space, action_space, lr1, lr2, hidden_dim, batch_size):
        '''
        Args:
        - γ (float): discount factor
        - ϵ (float): soft surrogate objective constraint
        - β (float): KL (Kullback–Leibler) penalty 
        - δ (float): KL divergence adaptive target
        - c1 (float): value loss weight
        - c2 (float): entropy weight
        - k_epoch (int): number of epochs to optimize
        - obs_space (int): observation space
        - action_space (int): action space
        - α_θ (float): actor learning rate
        - αv (float): critic learning rate
        
        '''
        self.γ = γ
        self.ϵ = ϵ
        self.β = β
        self.δ = δ
        self.c1 = c1
        self.c2 = c2
        self.k_epoch = k_epoch
        self.batch_size = batch_size
        self.actor_critic = ActorCriticNetwork(obs_space, hidden_dim, action_space)
        self.optimizer = torch.optim.Adam([
            {'params': self.actor_critic.actor.parameters(), 'lr': lr1},
            {'params': self.actor_critic.critic.parameters(), 'lr': lr2}
        ])
        
        #buffer to store current batch
        self.batch = []

        self.loss_func = nn.MSELoss()

    def predict(self, state, deterministic=True):
        """
        Wrapper for `act` function.
        """
        return self.act(state), 0

    def act(self, state):
        return self.actor_critic.act(state)
    
    def step(self, experience):
        self.batch.append(experience)

    def learn(self, count):
        if count % self.batch_size == 0:
            self.clipped_update()
    
    def process_rewards(self, rewards, terminals):
        ''' Converts our rewards history into cumulative discounted rewards
        Args:
        - rewards (Array): array of rewards 

        Returns:
        - G (Array): array of cumulative discounted rewards
        '''
        #Calculate Gt (cumulative discounted rewards)
        G = []

        #track cumulative reward
        total_r = 0

        #iterate rewards from Gt to G0
        for r, done in zip(reversed(rewards), reversed(terminals)):

            #Base case: G(T) = r(T)
            #Recursive: G(t) = r(t) + G(t+1)^DISCOUNT
            total_r = r + total_r * self.γ

            #no future rewards if current step is terminal
            if done:
                total_r = r

            #add to front of G
            G.insert(0, total_r)

        #whitening rewards
        G = torch.tensor(G)
        G = (G - G.mean())/G.std()

        return G
    
    def kl_divergence(self, old_lps, new_lps):
        ''' Calculate distance between two distributions with KL divergence
        Args:
        - old_lps (Array): array of old policy log probabilities
        - new_lps (Array): array of new policy log probabilities
        '''
        
        #track kl divergence
        total = 0
        
        #sum up divergence for all actions
        for old_lp, new_lp in zip(old_lps, new_lps):
            
            #same as old_lp * log(old_prob/new_prob) cuz of log rules
            total += old_lp * (old_lp - new_lp)

        return total
    
    
    def penalty_update(self):
        ''' Update policy using surrogate objective with adaptive KL penalty
        '''
        
        #get items from current batch
        states = [sample[0] for sample in self.batch]
        actions = [sample[1] for sample in self.batch]
        rewards = [sample[2] for sample in self.batch]
        old_lps = [sample[3] for sample in self.batch]
        terminals = [sample[4] for sample in self.batch]

        #calculate cumulative discounted rewards
        Gt = self.process_rewards(rewards, terminals)

        #track divergence
        divergence = 0

        #perform k-epoch update
        for epoch in range(self.k_epoch):

            #get ratio
            new_lps, entropies = self.actor_critic.evaluate_action(states, actions)
            #same as new_prob / old_prob
            ratios = torch.exp(new_lps - torch.Tensor(old_lps))

            #compute advantages
            states_tensor = torch.stack([torch.from_numpy(state).float().unsqueeze(0) for state in states]).squeeze(1)
            vals = self.actor_critic.critic(states_tensor).squeeze(1).detach()
            advantages = Gt - vals

            #get loss with adaptive kl penalty
            divergence = self.kl_divergence(old_lps, new_lps).detach()
            loss = -ratios * advantages + self.β * divergence

            #SGD via Adam
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        #update adaptive penalty
        if divergence >= 1.5 * self.δ:
            self.β *= 2
        elif divergence <= self.δ / 1.5:
            self.β /= 2
        
        #clear batch buffer
        self.batch = []
            
    def clipped_update(self):
        ''' Update policy using clipped surrogate objective
        '''
        #get items from trajectory
        states = [sample[0] for sample in self.batch]
        actions = [sample[1] for sample in self.batch]
        rewards = [sample[2] for sample in self.batch]
        old_lps = [sample[3] for sample in self.batch]
        terminals = [sample[4] for sample in self.batch]

        #calculate cumulative discounted rewards
        Gt = self.process_rewards(rewards, terminals)

        #perform k-epoch update
        for epoch in range(self.k_epoch):

            #get ratio
            new_lps, entropies = self.actor_critic.evaluate_action(states, actions)

            ratios = torch.exp(new_lps - torch.Tensor(old_lps))

            #compute advantages
            states_tensor = torch.stack([torch.from_numpy(state).float().unsqueeze(0) for state in states]).squeeze(1)
            vals = self.actor_critic.critic(states_tensor).squeeze(1).detach()
            advantages = Gt - vals

            #clip surrogate objective
            surrogate1 = torch.clamp(ratios, min=1 - self.ϵ, max=1 + self.ϵ) * advantages
            surrogate2 = ratios * advantages

            #loss, flip signs since this is gradient descent
            loss =  -torch.min(surrogate1, surrogate2) + self.c1 * self.loss_func(Gt, vals) - self.c2 * entropies

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        #clear batch buffer
        self.batch = []

class ActorCriticNetwork(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, out_dim):
        '''
        Args:
        - obs_space (int): observation space
        - action_space (int): action space
        
        '''
        super(ActorCriticNetwork, self).__init__()

        self.actor = nn.Sequential(
                            nn.Linear(in_dim, hidden_dim),
                            nn.Tanh(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.Tanh(),
                            nn.Linear(hidden_dim, out_dim),
                            nn.Softmax(dim=1))


        self.critic = nn.Sequential(
                        nn.Linear(in_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, 1))
        
    def forward(self):
        ''' Not implemented since we call the individual actor and critc networks for forward pass
        '''
        raise NotImplementedError
        
    def act(self, state):
        ''' Selects an action given current state
        Args:
        - network (Torch NN): network to process state
        - state (Array): Array of action space in an environment

        Return:
        - (int): action that is selected
        - (float): log probability of selecting that action given state and network
        '''

        # Setup state
        state = torch.from_numpy(state).float().unsqueeze(0)

        # Action probabilities
        action_probs = self.actor(state)

        # Sample an action using the probability distribution
        m = Categorical(action_probs)
        action = m.sample()

        # Return action
        return action.item(), m.log_prob(action)
    
    def evaluate_action(self, states, actions):
        ''' Get log probability and entropy of an action taken in given state
        Args:
        - states (Array): array of states to be evaluated
        - actions (Array): array of actions to be evaluated
        
        '''
        
        # Convert state to float tensor, add 1 dimension, allocate tensor on device
        states = torch.stack([torch.from_numpy(state).float().unsqueeze(0) for state in states]).squeeze(1)

        # Use network to predict action probabilities
        action_probs = self.actor(states)

        # Get probability distribution
        m = Categorical(action_probs)

        #return log_prob and entropy
        return m.log_prob(torch.Tensor(actions)), m.entropy()