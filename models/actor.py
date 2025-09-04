import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, candidate_dim, hidden_dim=128):
        super().__init__()

        # 狀態編碼器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # 候選點編碼器
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head
        self.actor_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, candidates, mask=None):
        """
        state: [B, state_dim]
        candidates: [B, N, candidate_dim]
        mask: [B, N] (0 = 無效候選點, 1 = 有效候選點)
        """
        B, N, _ = candidates.size()

        # 狀態特徵
        state_feat = self.state_encoder(state)  # [B, H]
        state_feat = state_feat.unsqueeze(1).expand(-1, N, -1)  # [B, N, H]

        # 候選特徵
        cand_feat = self.candidate_encoder(candidates)  # [B, N, H]

        # 融合
        joint_feat = state_feat + cand_feat  # [B, N, H]

        # logits
        logits = self.actor_head(joint_feat).squeeze(-1)  # [B, N]

        # masking
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)

        action_probs = F.softmax(logits, dim=-1)  # [B, N]

        return action_probs
