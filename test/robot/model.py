import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=2):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x):
        return torch.tanh(self.network(x)) * 0.1