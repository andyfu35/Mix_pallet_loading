# agents/buffer.py
import torch

class RolloutBuffer:
    """
    支援：
    - reward normalization：
        mode='rollout'  → 以當前這一批 rollout 的 reward 做標準化
        mode='running'  → 跨批次維持 running mean/std 來標準化
    - advantage normalization：PPO 標配（固定啟用）
    """
    def __init__(self, reward_norm=True, reward_norm_mode="rollout"):
        assert reward_norm_mode in ("rollout", "running")
        self.reward_norm = reward_norm
        self.reward_norm_mode = reward_norm_mode

        # running RMS（僅在 running 模式使用）
        self._rew_mean = 0.0
        self._rew_var = 1.0
        self._rew_count = 1e-4

        self._init_storage()

    def _init_storage(self):
        self.states = []
        self.cand_feats = []
        self.masks = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.returns = []
        self.advs = []

    def clear(self):
        # 保留 running 統計，不清空
        self._init_storage()

    def add(self, state, cand_feats, mask, action, logprob, reward, done, value):
        self.states.append(state)
        self.cand_feats.append(cand_feats)
        self.masks.append(mask)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    # --------- running RMS 工具（僅在 reward_norm_mode='running' 用） ---------
    @torch.no_grad()
    def _update_reward_rms(self, x: torch.Tensor):
        """
        x: [T] 的 rewards（或可用 returns）張量
        使用並更新 running mean/var（Welford 合併公式）
        """
        x = x.float()
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = x.numel()

        delta = batch_mean - self._rew_mean
        tot_count = self._rew_count + batch_count

        new_mean = self._rew_mean + delta * batch_count / tot_count
        m_a = self._rew_var * self._rew_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self._rew_count * batch_count / tot_count
        new_var = M2 / tot_count

        self._rew_mean = new_mean.item()
        self._rew_var = new_var.item()
        self._rew_count = tot_count

    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if not self.reward_norm:
            return rewards
        if self.reward_norm_mode == "rollout":
            # 用當前這批 rewards 做標準化
            mean = rewards.mean()
            std = rewards.std(unbiased=False)
            return (rewards - mean) / (std + 1e-8)
        else:
            # running 模式：先用目前批次更新 RMS，再用更新後的 mean/std 正規化
            self._update_reward_rms(rewards)
            mean = torch.tensor(self._rew_mean, dtype=torch.float32, device=rewards.device)
            std = torch.tensor(self._rew_var, dtype=torch.float32, device=rewards.device).sqrt()
            return (rewards - mean) / (std + 1e-8)

    # ------------------------------------------------------------------------

    def compute_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95):
        rewards = torch.tensor(self.rewards, dtype=torch.float32)          # [T]
        dones   = torch.tensor(self.dones,   dtype=torch.float32)          # [T]
        values  = torch.stack(self.values).view(-1).float()                # [T]

        # ==== Reward normalization ====
        rewards = self._normalize_rewards(rewards)

        # ---- GAE ----
        T = rewards.numel()
        advs = torch.zeros(T, dtype=torch.float32)
        gae = 0.0
        next_value = 0.0  # 遇到 done 時 bootstrap=0（mask 會處理）

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advs[t] = gae
            next_value = values[t]

        returns = advs + values.detach()

        # ==== Advantage normalization（PPO 標配）====
        advs = (advs - advs.mean()) / (advs.std(unbiased=False) + 1e-8)

        self.returns = returns.flatten()
        self.advs = advs.flatten()

    def as_batches(self, batch_size, device):
        n = len(self.states)
        idxs = torch.randperm(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            mb = idxs[start:end]

            yield (
                torch.stack([self.states[i] for i in mb]).to(device),
                torch.stack([self.cand_feats[i] for i in mb]).to(device),
                torch.stack([self.masks[i] for i in mb]).to(device),
                torch.stack([self.actions[i] for i in mb]).to(device),
                torch.stack([self.logprobs[i] for i in mb]).to(device),
                self.returns[mb].to(device).view(-1),
                self.advs[mb].to(device).view(-1),
                torch.stack([self.values[i] for i in mb]).detach().to(device).view(-1),
            )
