import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Code adapted from:
# https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html
# The above is a DQN algorithm. I modified the loss function to be DDQN instead.
#
# DDQN loss function from:
# https://github.com/fschur/DDQN-with-PyTorch-for-OpenAI-Gym/blob/master/DDQN_discrete.py


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return self.fc3(x)


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, buffer_size, batch_size, update_interval, gamma, device, lr, tau):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.device = device
        self.tau = tau

        self.qnetwork = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=lr)

        self.loss_func = nn.MSELoss()

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state)
        self.qnetwork.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def step(self, state, action, reward, next_state, done):
        # Setup
        state = torch.from_numpy(np.array(state)).float().to(self.device)
        action = torch.from_numpy(np.array(action)).long().to(self.device)
        reward = torch.from_numpy(np.array(reward)).float().to(self.device)
        next_state = torch.from_numpy(np.array(next_state)).float().to(self.device)
        done = torch.from_numpy(np.array(done).astype(np.uint8)).float().to(self.device)

        # LOSS FROM
        # https://github.com/jcborges/FCQ/tree/master
        
        # Compute
        q_next = self.qnetwork(next_state).detach()
        q_next_max = torch.max(q_next)#[0].view(-1) 
        q_target = reward.view(-1) + q_next_max * (1 - done.view(-1))
        q_local = self.qnetwork(state)[action.view(-1)].view(-1)

        # Loss
        loss = self.loss_func(q_local, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.qnetwork.state_dict(), path + ".pt")