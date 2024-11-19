import torch
import torch.nn as nn

class RegressionModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super(RegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.model(x) 