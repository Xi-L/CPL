"""
A simple FC Pareto Set model.
"""

import torch
import torch.nn as nn

class ParetoSetModel(torch.nn.Module):
    def __init__(self, n_dim):
        super(ParetoSetModel, self).__init__()
        self.n_dim = n_dim
       
        self.fc1 = nn.Linear(1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.n_dim)
       
    def forward(self, t):

        x = torch.relu(self.fc1(t))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x.to(torch.float64)