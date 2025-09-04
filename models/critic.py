import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()

        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """
        state: [B, state_dim]
        return: [B, 1]
        """
        return self.value_net(state)
