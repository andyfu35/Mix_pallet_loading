import torch.nn as nn
from models.actor import Actor
from models.critic import Critic

class ActorCriticPolicy(nn.Module):
    def __init__(self, state_dim, candidate_dim, hidden_dim=128):
        super().__init__()
        self.actor = Actor(state_dim, candidate_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)

    def forward(self, state, candidates, mask=None):
        """
        state: [B, state_dim]
        candidates: [B, N, candidate_dim]
        mask: [B, N] (0/1)
        """
        action_probs = self.actor(state, candidates, mask)  # [B, N]
        value = self.critic(state)  # [B, 1]
        return action_probs, value
