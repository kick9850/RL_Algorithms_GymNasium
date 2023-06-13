import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from collections import namedtuple
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    def __init__(self, state_space, action_space, hidden_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_space, hidden_size)

        self.action_head = nn.Linear(hidden_size, action_space)
        self.value_head = nn.Linear(hidden_size, 1) # Scalar Value

        self.save_actions = []
        self.rewards = []
        os.makedirs('./save_model/AC_CartPole-v1', exist_ok=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        state_value = self.value_head(x)

        return F.softmax(action_score, dim=-1), state_value


class Agent():
    def __init__(self,
                 state_space,
                 action_space,
                 hidden_size,
                 lr,
                 gamma):

        self.state_space = state_space
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.learning_rate = lr
        self.gamma = gamma
        self.model = Policy(self.state_space, self.action_space,self.hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def plot(self, steps):
        ax = plt.subplot(111)
        ax.cla()
        ax.grid()
        ax.set_title('Training')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Run Time')
        ax.plot(steps)
        RunTime = len(steps)

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self.model(state)
        m = Categorical(probs)
        action = m.sample()
        self.model.save_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def finish_episode(self):
        R = 0
        save_actions = self.model.save_actions
        policy_loss = []
        value_loss = []
        rewards = []

        for r in self.model.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

        for (log_prob , value), r in zip(save_actions, rewards):
            reward = r - value.item()
            policy_loss.append(-log_prob * reward)
            value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        loss.backward()
        self.optimizer.step()

        del self.model.rewards[:]
        del self.model.save_actions[:]