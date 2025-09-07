import json
import torch
import numpy as np
import pybullet as p
import pybullet_data

from envs.container_packing_env import ContainerPackingEnv
from models.policy import ActorCriticPolicy
from agents.ppo import PPOAgent


def train(env, ppo, max_episodes=1000, max_steps=200, update_timestep=2000):
    timestep = 0
    episode_rewards = []

    for ep in range(max_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        total_reward = 0

        for step in range(max_steps):
            timestep += 1

            # --- 建立 candidates + mask ---
            valid_action_count = len(env.candidates)
            if valid_action_count == 0:
                print("⚠️ 無候選點，提前結束 episode")
                break

            # 抽取候選點數值特徵 [N, 6]
            cand_feats = []
            for (x, y, z, info) in env.candidates:
                cand_feats.append([
                    x, y, z,
                    info.get("coverage", 0.0),
                    float(info.get("left_support", False)),
                    float(info.get("right_support", False)),
                ])
            candidates_tensor = torch.tensor(cand_feats, dtype=torch.float32)  # [N, 6]

            # mask → 與候選點長度一致 [N]
            mask_tensor = torch.ones(len(env.candidates), dtype=torch.float32)

            # --- 選 action ---
            action, log_prob, value = ppo.select_action(state, candidates_tensor, mask_tensor)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = torch.tensor(next_state, dtype=torch.float32)

            # --- 存進 buffer ---
            ppo.buffer.states.append(state)
            ppo.buffer.actions.append(action)
            ppo.buffer.log_probs.append(log_prob)
            ppo.buffer.values.append(value)
            ppo.buffer.rewards.append(reward)
            ppo.buffer.dones.append(done)
            ppo.buffer.candidates.append(candidates_tensor)
            ppo.buffer.masks.append(mask_tensor)

            state = next_state
            total_reward += reward

            # --- 更新 PPO ---
            if timestep % update_timestep == 0:
                ppo.update()

            if done:
                break

        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: reward={total_reward:.3f}")

    return episode_rewards


if __name__ == "__main__":
    # === 建立 PyBullet client ===
    client_id = p.connect(p.DIRECT)   # 用 p.GUI 可以看到畫面
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
    rewards = train(env, ppo, max_episodes=100, max_steps=200, update_timestep=2000)

    # === 存模型 ===
    torch.save(policy.state_dict(), "ppo_container.pt")
    print("✅ 訓練完成，模型已儲存為 ppo_container.pt")

    # === 關閉 PyBullet ===
    p.disconnect(client_id)
