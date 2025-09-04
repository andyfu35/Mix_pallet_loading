import torch
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def store(self, state, action, log_prob, reward, done, value):
        """存一個 step 的資料"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(self, gamma=0.99, lam=0.95):
        """
        計算 GAE(λ) Advantage 和 Returns
        """
        values = self.values + [0]  # bootstrap
        T = len(self.rewards)

        self.advantages = [0] * T
        gae = 0
        for t in reversed(range(T)):
            delta = self.rewards[t] + gamma * values[t+1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae

        self.returns = [adv + v for adv, v in zip(self.advantages, self.values)]

        # 轉成 tensor
        self.states = torch.stack(self.states)
        self.actions = torch.tensor(self.actions)
        self.log_probs = torch.stack(self.log_probs)
        self.advantages = torch.tensor(self.advantages, dtype=torch.float32)
        self.returns = torch.tensor(self.returns, dtype=torch.float32)

        # Advantage normalization (常用 trick)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batches(self, batch_size):
        """
        隨機打亂後分成 mini-batches
        """
        n = len(self.states)
        indices = np.arange(n)
        np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield (
                self.states[batch_idx],
                self.actions[batch_idx],
                self.log_probs[batch_idx],
                self.advantages[batch_idx],
                self.returns[batch_idx],
            )

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = []
        self.returns = []
