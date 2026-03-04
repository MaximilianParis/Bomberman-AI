import torch.nn as nn
import torch
from torch import distributions

class BaseActor(nn.Module):
    def __init__(self,conv_output_dim,rest_dim,action_size):
        super().__init__()

        # ----- CNN für Grid ----- 
           
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1,stride=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1,stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1,stride=1),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        
      
        
        # ----- MLP Head -----
        self.mlp = nn.Sequential(
            nn.Linear(conv_output_dim + rest_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, grid, rest):
        
            x = self.conv(grid)
            x = self.flatten(x)
            x = torch.cat((x, rest), dim=1)
            return self.mlp(x)


    def forward_with_softmax(self, grid, rest):
      
        return torch.softmax(self.forward(grid,rest), dim=-1)
    
    def get_action(self, prob):
        dist = distributions.Categorical(prob)
        action = dist.sample()
        return action
    
    def get_best_action(self,  grid, rest):
        prob= self.forward_with_softmax( grid, rest)
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