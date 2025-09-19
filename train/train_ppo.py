import os
import time
import random
import numpy as np
import torch
import pybullet as p
import json

from agents.buffer import RolloutBuffer
from agents.ppo import PPOAgent
from models.policy import ActorCritic
from envs.container_packing_env import ContainerPackingEnv
from train.lookahead import rollout_value   # <<< 新增

# ========= Utils =========
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
    except Exception:
        pass

class RMS:
    def __init__(self, shape):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = 1e-4
    def update(self, x):
        m = x.mean(0); v = x.var(0, unbiased=False); n = x.size(0)
        delta = m - self.mean
        tot = self.count + n
        new_mean = self.mean + delta * n / tot
        m_a = self.var * self.count
        m_b = v * n
        M2 = m_a + m_b + delta.pow(2) * self.count * n / tot
        self.mean, self.var, self.count = new_mean, M2 / tot, tot
    def normalize(self, x, eps=1e-8):
        return (x - self.mean) / torch.sqrt(self.var + eps)

# ========= Hyper-params =========
UPDATE_TIMESTEP = 512      # 原本 2048 → 降低，收集 512 step 就更新一次
N_EPOCHS        = 5        # 保持一樣，讓每批資料用足
MINIBATCH_SIZE  = 128      # 原本 512 → 降低，更多梯度更新次數
LR              = 3e-4     # 稍微提高學習率，加快收斂
CLIP_EPS        = 0.2      # 原本 0.1 → 放寬更新幅度，加快策略調整
ENTROPY_COEF    = 0.005    # 原本 0.02 → 降低探索，趕快 exploit
VALUE_COEF      = 0.5      # 保持
GAMMA           = 0.99     # 視野縮短一點，強調近期裝載率提升
GAE_LAMBDA      = 0.9      # 降低 lambda，減少方差，加快學習
MAX_CANDS       = 16       # 原本 64 → 減少候選點，降低動作空間
CAND_DIM        = 6        # 不變

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def pad_candidates(cands):
    if cands is None or len(cands) == 0:
        feats = np.zeros((MAX_CANDS, CAND_DIM), dtype=np.float32)
        mask = np.zeros((MAX_CANDS,), dtype=np.float32)
        return feats, mask
    cands = np.asarray(cands, dtype=np.float32)
    N = min(len(cands), MAX_CANDS)
    feats = np.zeros((MAX_CANDS, CAND_DIM), dtype=np.float32)
    feats[:N] = cands[:N]
    mask = np.zeros((MAX_CANDS,), dtype=np.float32)
    mask[:N] = 1.0
    return feats, mask

def _extract_candidates(env):
    cand_list = []
    cands = getattr(env, "candidates", [])
    for item in cands:
        if isinstance(item, (list, tuple)) and len(item) >= 4:
            x, y, z, meta = item[0], item[1], item[2], item[3]
            if isinstance(meta, (list, tuple, np.ndarray)) and len(meta) >= 3:
                feat = np.array([x, y, z, meta[0], meta[1], meta[2]], dtype=np.float32)
            else:
                feat = np.array([x, y, z, 0, 0, 0], dtype=np.float32)
        elif isinstance(item, (list, tuple)) and len(item) == 6:
            feat = np.array(item, dtype=np.float32)
        else:
            continue
        cand_list.append(feat)
    return cand_list

def _get_fill_rate_from_env(env, info_dict=None):
    if isinstance(info_dict, dict) and "fill_rate" in info_dict:
        return info_dict["fill_rate"]
    fr_attr = getattr(env, "current_fill_ratio", None)
    try:
        if callable(fr_attr):
            return float(fr_attr())
        elif isinstance(fr_attr, (int, float)):
            return float(fr_attr)
    except Exception:
        pass
    return None

def main(seed=42):
    seed_everything(seed)

    # 1) PyBullet
    with open("data/container_specs/container_20ft.json") as f:
        container_env = json.load(f)
    cfg = container_env["container"]
    use_gui = True
    client_id = p.connect(p.GUI if use_gui else p.DIRECT)

    # 2) 環境
    try:
        env = ContainerPackingEnv(cfg, client_id=client_id)
    except TypeError:
        env = ContainerPackingEnv()

    obs_dim = int(env.observation_space.shape[0])
    policy = ActorCritic(obs_dim=obs_dim, cand_feat_dim=CAND_DIM, hidden=256)
    agent = PPOAgent(policy, lr=LR, clip_eps=CLIP_EPS,
                     entropy_coef=ENTROPY_COEF, value_coef=VALUE_COEF, device=DEVICE)

    obs_rms = RMS(obs_dim)
    buffer = RolloutBuffer()

    timestep = 0
    episode_idx = 0
    ep_return = 0.0

    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32)

    while True:
        buffer.clear()
        states_for_rms = []

        while timestep < UPDATE_TIMESTEP:
            # 候選點處理
            cand_list = _extract_candidates(env)
            cand_feats_np, mask_np = pad_candidates(cand_list)
            cand_feats = torch.tensor(cand_feats_np, dtype=torch.float32)
            mask = torch.tensor(mask_np, dtype=torch.float32)

            states_for_rms.append(state.unsqueeze(0))
            norm_state = obs_rms.normalize(state).unsqueeze(0)
            cand_feats_b = cand_feats.unsqueeze(0)
            mask_b = mask.unsqueeze(0)

            if mask.max().item() < 0.5:
                reward = -1.0
                terminated, truncated = False, True
                next_state = state.clone().numpy()
                with torch.no_grad():
                    _, value = agent.policy(norm_state.to(DEVICE),
                                            cand_feats_b.to(DEVICE),
                                            mask_b.to(DEVICE))
                value = value.squeeze(0).cpu()
                action = torch.tensor(0)
                logprob = torch.tensor(0.0)
                info = {"reason": "no_candidates",
                        "fill_rate": _get_fill_rate_from_env(env)}
                done = True
            else:
                # === 用 lookahead 選動作 ===
                best_a, score = rollout_value(env, depth=2, gamma=0.99, beam_k=5)
                action = torch.tensor(best_a)

                # policy 仍算 logprob/value（讓 PPO 學習）
                with torch.no_grad():
                    _, logprob, value = agent.policy.act(
                        norm_state.to(DEVICE),
                        cand_feats_b.to(DEVICE),
                        mask_b.to(DEVICE)
                    )
                action = action.squeeze(0).cpu()
                logprob = logprob.squeeze(0).cpu()
                value = value.squeeze(0).cpu()

                next_state, reward, terminated, truncated, info = env.step(int(action.item()))
                done = bool(terminated or truncated)

            buffer.add(state, cand_feats, mask, action, logprob, float(reward), done, value)
            ep_return += float(reward)
            timestep += 1
            state = torch.tensor(next_state, dtype=torch.float32)

            if done:
                episode_idx += 1
                if len(states_for_rms) > 0:
                    try:
                        obs_rms.update(torch.cat(states_for_rms, dim=0).cpu())
                    except Exception:
                        pass
                states_for_rms = []
                fill_rate = _get_fill_rate_from_env(env, info_dict=info)
                if fill_rate is not None:
                    print(f"[Episode {episode_idx}] Return={ep_return:.3f}, FillRate={fill_rate:.3f}")
                else:
                    print(f"[Episode {episode_idx}] Return={ep_return:.3f}")
                ep_return = 0.0
                state, info = env.reset()
                state = torch.tensor(state, dtype=torch.float32)

            if timestep >= UPDATE_TIMESTEP:
                break

        buffer.compute_returns_and_advantages(gamma=GAMMA, gae_lambda=GAE_LAMBDA)
        stats = agent.update(buffer, n_epochs=N_EPOCHS, batch_size=MINIBATCH_SIZE)
        print(f"[Update] approx_kl={stats['approx_kl']:.4f}")

        timestep = 0


if __name__ == "__main__":
    main()
