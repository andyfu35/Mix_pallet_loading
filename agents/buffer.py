# agents/buffer.py
import torch
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def store(self, state, action, log_prob, reward, done, value, candidates, mask):
        self.states.append(state)               # [S]
        self.actions.append(action)             # int/long
        self.log_probs.append(log_prob)         # []
        self.rewards.append(float(reward))      # scalar
        self.dones.append(float(done))          # 0/1
        self.values.append(value.squeeze().item())  # 存成 float，避免之後 stack 雜型
        self.candidates.append(candidates)      # [MAX_CANDS, C]
        self.masks.append(mask)                 # [MAX_CANDS]

    def compute_returns_and_advantages(self, gamma=0.99, lam=0.95, last_value=0.0, last_done=1.0):
        """
        last_value: rollout 最後一步「下一狀態」的 V(s_{T}) 估計（critic）
        last_done : 1 if episode ended at the last step, else 0（作為 bootstrap mask）
        """
        T = len(self.rewards)
        values = self.values + [float(last_value)]  # 用 critic 的 bootstrap
        dones = self.dones + [float(last_done)]

        advs = [0.0] * T
        gae = 0.0
        for t in reversed(range(T)):
            delta = self.rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advs[t] = gae

        rets = [advs[t] + values[t] for t in range(T)]

        # 轉 tensor
        self.states = torch.stack(self.states)                       # [T, S]
        self.actions = torch.tensor(self.actions, dtype=torch.long)  # [T]
        self.log_probs = torch.stack(self.log_probs).float()         # [T]
        self.advantages = torch.tensor(advs, dtype=torch.float32)    # [T]
        self.returns = torch.tensor(rets, dtype=torch.float32)       # [T]
        self.candidates = torch.stack(self.candidates).float()       # [T, N, C]
        self.masks = torch.stack(self.masks).float()                 # [T, N]

        # normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batches(self, batch_size):
        n = len(self.states)
        idx = np.arange(n)
        np.random.shuffle(idx)
        for s in range(0, n, batch_size):
            b = idx[s:s+batch_size]
            yield (
                self.states[b],
                self.actions[b],
                self.log_probs[b],
                self.advantages[b],
                self.returns[b],
                self.candidates[b],
                self.masks[b],
            )

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.candidates = []
        self.masks = []
        self.advantages = []
        self.returns = []
