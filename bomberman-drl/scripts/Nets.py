import torch.nn as nn
import torch
from torch import distributions

class BaseActor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.gate = nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(state_size, int(state_size/2)), self.gate,
            nn.Linear(int(state_size/2), int(state_size/4)), self.gate,
            nn.Linear(int(state_size/4), action_size),
          
        )

      
    def forward(self, state):
        return self.net(state)

    def forward_with_softmax(self, state):
        return torch.softmax(self.net(state), dim=-1)
    
    def get_action(self, prob):
        dist = distributions.Categorical(prob)
        action = dist.sample()
        return action
    
    def get_best_action(self, prob):
        _, indices = prob.max(dim=1)
        return indices

    def eval_action(self, prob, action):
        dist = distributions.Categorical(prob)
        return {
            "log_prob": dist.log_prob(action),
            "entropy": dist.entropy()
        }

class BaseCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.gate = nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(state_size, int(state_size/2)), self.gate,
            nn.Linear(int(state_size/2), int(state_size/4)), self.gate,
            nn.Linear(int(state_size/4), 1)
        )
    
    def forward(self, state):
        return self.net(state)