# agents/ppo.py
import torch
import torch.nn.functional as F
from agents.buffer import RolloutBuffer

class PPOAgent:
    def __init__(self, policy, optimizer,
                 clip_eps=0.2, gamma=0.99, gae_lambda=0.95,
                 value_coef=0.5, entropy_coef=0.01,
                 batch_size=64, n_epochs=10, device=None):
        """
        policy: ActorCriticPolicy
        optimizer: torch.optim.Adam(policy.parameters(), lr=...)
        """
        self.policy = policy
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device or next(policy.parameters()).device

        self.buffer = RolloutBuffer()

    @torch.no_grad()
    def select_action(self, state, candidates, mask):
        """
        state:      [state_dim]            (float32)
        candidates: [N_fixed, cand_dim]    (float32)  ← 需事先 pad 成固定長度
        mask:       [N_fixed]              (float32: 1 有效, 0 無效)
        """
        # to device + add batch dim
        state = state.to(self.device).unsqueeze(0)             # [1, S]
        candidates = candidates.to(self.device).unsqueeze(0)   # [1, N, C]
        mask = mask.to(self.device).unsqueeze(0) if mask is not None else None  # [1, N]

        action_probs, value = self.policy(state, candidates, mask)  # probs [1, N], value [1, 1]

        # 保險：避免非常小的概率導致數值問題
        action_probs = action_probs.clamp(min=1e-8)
        dist = torch.distributions.Categorical(action_probs)

        action = dist.sample()                    # [1]
        log_prob = dist.log_prob(action)          # [1]

        # squeeze 成標準形狀
        action = action.squeeze(0)                # []
        log_prob = log_prob.squeeze(0)            # []
        value = value.squeeze(0)                  # [1]

        return int(action.item()), log_prob.detach(), value.detach()

    def update(self):
        """用 buffer 的資料更新 PPO（假設 train 端已把 candidates/mask 固定長度）"""
        self.buffer.compute_returns_and_advantages(self.gamma, self.gae_lambda)

        # 確保資料在正確裝置與 dtype
        states = self.buffer.states.to(self.device)                # [T, S]
        actions = self.buffer.actions.to(self.device).long()       # [T]
        old_log_probs = self.buffer.log_probs.to(self.device).float()  # [T]
        advantages = self.buffer.advantages.to(self.device).float()    # [T]
        returns = self.buffer.returns.to(self.device).float()          # [T]
        candidates = self.buffer.candidates.to(self.device).float()    # [T, N, C]
        masks = self.buffer.masks.to(self.device).float()              # [T, N]

        T = states.size(0)
        idx = torch.randperm(T, device=self.device)

        for _ in range(self.n_epochs):
            for start in range(0, T, self.batch_size):
                end = start + self.batch_size
                b = idx[start:end]

                b_states = states[b]          # [B, S]
                b_actions = actions[b]        # [B]
                b_old_logp = old_log_probs[b] # [B]
                b_adv = advantages[b]         # [B]
                b_ret = returns[b]            # [B]
                b_cands = candidates[b]       # [B, N, C]
                b_masks = masks[b]            # [B, N]

                # 前向
                action_probs, values = self.policy(b_states, b_cands, b_masks)  # [B,N], [B,1]
                action_probs = action_probs.clamp(min=1e-8)
                dist = torch.distributions.Categorical(action_probs)

                new_logp = dist.log_prob(b_actions)       # [B]
                entropy = dist.entropy().mean()

                # 比率
                ratios = torch.exp(new_logp - b_old_logp) # [B]

                # 演員損失（clip）
                surr1 = ratios * b_adv
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # 評論家損失
                values = values.squeeze(-1)                # [B]
                critic_loss = F.mse_loss(values, b_ret)

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # 可選，穩定一些
                self.optimizer.step()

        self.buffer.clear()
