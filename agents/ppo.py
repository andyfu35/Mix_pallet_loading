import torch
import torch.nn.functional as F

class PPOAgent:
    def __init__(self, policy, lr, clip_eps, entropy_coef, value_coef, device, max_grad_norm=0.5):
        self.policy = policy.to(device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = device
        self.max_grad_norm = max_grad_norm

    def update(self, buffer, n_epochs, batch_size):
        policy_loss_total, value_loss_total, entropy_total, n_batches = 0, 0, 0, 0
        approx_kl_total = 0.0

        for epoch in range(n_epochs):
            for states, cands, masks, actions, old_logp, returns, advs, old_values in \
                    buffer.as_batches(batch_size, self.device):

                dist, values = self.policy(states, cands, masks)
                values = values.squeeze(-1).view(-1)   # [B]
                returns = returns.view(-1)             # [B]
                advs = advs.view(-1)                   # [B]
                old_values = old_values.view(-1)       # [B]

                new_logp = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advs
                policy_loss = -torch.min(surr1, surr2).mean()

                v_pred_clipped = old_values + (values - old_values).clamp(-0.2, 0.2)
                v_loss_unclipped = F.mse_loss(values, returns, reduction="mean")
                v_loss_clipped   = F.mse_loss(v_pred_clipped, returns, reduction="mean")
                value_loss = torch.max(v_loss_unclipped, v_loss_clipped)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = (old_logp - new_logp).mean().clamp_min(0).item()

                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                entropy_total += entropy.item()
                approx_kl_total += approx_kl
                n_batches += 1

        return {
            "policy_loss": policy_loss_total / n_batches,
            "value_loss": value_loss_total / n_batches,
            "entropy": entropy_total / n_batches,
            "approx_kl": approx_kl_total / n_batches
        }
