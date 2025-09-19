# models/policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    觀測 obs → 全局 embedding
    候選點 cand_feats: [B, A_max, C]；用 obs_embed 與 cand_embed 做互動，產生每個候選的 logit
    使用 mask 做無效動作遮罩
    """
    def __init__(self, obs_dim: int, cand_feat_dim: int, hidden: int = 256):
        super().__init__()
        # 全局
        self.obs_mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # 候選
        self.cand_mlp = nn.Sequential(
            nn.Linear(cand_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # 將 obs/cand 交互後得到每個候選的 logit
        self.score = nn.Linear(hidden * 2, 1)

        # 值函數
        self.v_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, cand_feats, mask):
        """
        obs: [B, obs_dim]
        cand_feats: [B, A, C]
        mask: [B, A]  0/1
        """
        B, A, C = cand_feats.shape
        obs_embed = self.obs_mlp(obs)                       # [B, H]
        cand_embed = self.cand_mlp(cand_feats.reshape(B*A, C)).reshape(B, A, -1)  # [B, A, H]

        # 與 obs 融合
        obs_expand = obs_embed.unsqueeze(1).expand(-1, A, -1)   # [B, A, H]
        x = torch.cat([obs_expand, cand_embed], dim=-1)         # [B, A, 2H]
        logits = self.score(x).squeeze(-1)                      # [B, A]

        # 遮罩（把非法動作 logit 設為 -inf）
        very_neg = torch.finfo(logits.dtype).min
        masked_logits = torch.where(mask > 0.5, logits, torch.full_like(logits, very_neg))

        # Policy 發布與 Value
        dist = Categorical(logits=masked_logits)
        value = self.v_mlp(obs_embed)                           # [B, 1]
        return dist, value

    @torch.no_grad()
    def act(self, obs, cand_feats, mask):
        dist, value = self.forward(obs, cand_feats, mask)
        action = dist.sample()                           # [B]
        logprob = dist.log_prob(action)                  # [B]
        return action, logprob, value
