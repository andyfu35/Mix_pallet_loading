# train/train_ppo.py
import json
import torch
import numpy as np
import pybullet as p
import pybullet_data

from envs.container_packing_env import ContainerPackingEnv
from models.policy import ActorCriticPolicy
from agents.ppo import PPOAgent


def train(env, ppo, max_episodes=1000, max_steps=200, update_timestep=2000):
    """
    固定 candidates/mask 為 [MAX_CANDS, CAND_DIM] / [MAX_CANDS] 後再存入 buffer，
    避免 torch.stack 失敗；policy 用 mask 遮掉 padding。
    """
    timestep = 0
    episode_rewards = []
    episode_boxes = []

    MAX_CANDS = getattr(env, "max_candidates", 100)  # 與環境一致
    CAND_DIM = 6  # (x, y, z, coverage, left_support, right_support)

    def pad_to_max(cand_tensor, mask_tensor, max_n, cand_dim):
        """
        cand_tensor: [N, cand_dim]
        mask_tensor: [N]
        return: cand_fixed [max_n, cand_dim], mask_fixed [max_n]
        """
        n = int(cand_tensor.size(0))
        if n >= max_n:
            return cand_tensor[:max_n], mask_tensor[:max_n]
        pad_c = torch.zeros((max_n - n, cand_dim), dtype=cand_tensor.dtype, device=cand_tensor.device)
        pad_m = torch.zeros((max_n - n,), dtype=mask_tensor.dtype, device=mask_tensor.device)
        return torch.cat([cand_tensor, pad_c], dim=0), torch.cat([mask_tensor, pad_m], dim=0)

    for ep in range(max_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        total_reward = 0.0

        for step in range(max_steps):
            timestep += 1

            # --- 取候選點 ---
            N = len(env.candidates)
            if N == 0:
                print("⚠️ 無候選點，提前結束 episode")
                break

            # 抽取候選點特徵 [N, 6]，如超過 MAX_CANDS 先截斷
            raw_cands = env.candidates[:MAX_CANDS]

            cand_feats = []
            for (x, y, z, info_c) in raw_cands:
                cand_feats.append([
                    float(x), float(y), float(z),
                    float(info_c.get("coverage", 0.0)),
                    float(info_c.get("left_support", False)),
                    float(info_c.get("right_support", False)),
                ])
            candidates_tensor = torch.tensor(cand_feats, dtype=torch.float32)          # [N, 6]
            mask_tensor = torch.ones(candidates_tensor.size(0), dtype=torch.float32)   # [N]

            # 固定長度（padding 到 MAX_CANDS）
            candidates_fixed, mask_fixed = pad_to_max(candidates_tensor, mask_tensor, MAX_CANDS, CAND_DIM)
            # --- 用固定長度版本選動作（policy 會用 mask 遮掉 padding） ---
            action, log_prob, value = ppo.select_action(state, candidates_fixed, mask_fixed)

            # --- 與環境互動 ---
            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            # --- 存進 buffer（固定長度版本！） ---
            ppo.buffer.states.append(state)                  # [S]
            ppo.buffer.actions.append(int(action))           # scalar -> int
            ppo.buffer.log_probs.append(log_prob)            # []
            ppo.buffer.values.append(value)                  # [1]
            ppo.buffer.rewards.append(float(reward))         # scalar float
            ppo.buffer.dones.append(float(done))             # 0/1 float
            ppo.buffer.candidates.append(candidates_fixed)   # [MAX_CANDS, 6]
            ppo.buffer.masks.append(mask_fixed)              # [MAX_CANDS]

            # --- 累計 ---
            state = next_state
            total_reward += float(reward)

            # --- 觸發更新 ---
            if timestep % update_timestep == 0:
                ppo.update()

            if done:
                break

        # --- 每輪統計 ---
        num_boxes = len(getattr(env, "placed", []))
        episode_boxes.append(num_boxes)
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: reward={total_reward:.3f}, boxes={num_boxes}")

    return episode_rewards, episode_boxes


if __name__ == "__main__":
    # === 建立 PyBullet client ===
    client_id = p.connect(p.DIRECT)   # 若要看畫面可改 p.GUI
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # === 載入 container 配置 ===
    with open("data/container_specs/container_20ft.json") as f:
        container_env = json.load(f)
    cfg = container_env["container"]

    # === 初始化環境 ===
    env = ContainerPackingEnv(cfg, resolution=0.1, client_id=client_id)
    obs_dim = env.observation_space.shape[0]
    cand_dim = 6   # 候選點特徵 (x, y, z, coverage, left_support, right_support)

    # === 建立 policy & PPO ===
    policy = ActorCriticPolicy(state_dim=obs_dim, candidate_dim=cand_dim, hidden_dim=128)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    ppo = PPOAgent(policy, optimizer)

    # === 開始訓練 ===
    rewards, boxes = train(env, ppo, max_episodes=100, max_steps=10000, update_timestep=2048)

    # === 存模型 ===
    torch.save(policy.state_dict(), "ppo_container.pt")
    print("✅ 訓練完成，模型已儲存為 ppo_container.pt")
    print("📦 每輪放置箱子數：", boxes)

    # === 關閉 PyBullet ===
    p.disconnect(client_id)
