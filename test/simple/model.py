import torch
import torch.nn as nn
import math

# Neural Network Model
class SineNet(nn.Module):
    def __init__(self):
        super(SineNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# Convert degrees to radians
def degrees_to_radians(degrees):
    return degrees * math.pi / 180 