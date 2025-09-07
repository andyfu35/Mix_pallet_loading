import torch
import torch.nn.functional as F
from agents.buffer import RolloutBuffer


class PPOAgent:
    def __init__(self, policy, optimizer,
                 clip_eps=0.2, gamma=0.99, gae_lambda=0.95,
                 value_coef=0.5, entropy_coef=0.01,
                 batch_size=64, n_epochs=10):
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

        # ✅ 初始化 RolloutBuffer
        self.buffer = RolloutBuffer()

    def select_action(self, state, candidates, mask):
        """
        state: [state_dim]
        candidates: [N, candidate_dim]
        mask: [N]
        """
        state = state.unsqueeze(0)             # [1, state_dim]
        candidates = candidates.unsqueeze(0)   # [1, N, cand_dim]
        mask = mask.unsqueeze(0) if mask is not None else None  # [1, N]

        action_probs, value = self.policy(state, candidates, mask)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.detach(), value.detach()

    def update(self):
        """用 buffer 的資料更新 PPO"""
        self.buffer.compute_returns_and_advantages(self.gamma, self.gae_lambda)

        for _ in range(self.n_epochs):
            for states, actions, old_log_probs, advantages, returns, candidates, masks in self.buffer.get_batches(self.batch_size):

                # Forward with candidates & mask
                action_probs, values = self.policy(states, candidates, masks)
                dist = torch.distributions.Categorical(action_probs)
                log_probs = dist.log_prob(actions)

                # Ratio
                ratios = torch.exp(log_probs - old_log_probs)

                # Actor loss (clip)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                critic_loss = F.mse_loss(values.squeeze(), returns)

                # Entropy bonus
                entropy = dist.entropy().mean()

                # Total loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # ✅ 清空 buffer
        self.buffer.clear()
