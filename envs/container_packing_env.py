import gymnasium as gym
import numpy as np
from envs.container_env import ContainerEnv, Box
from envs.candidate_generator import CandidateGenerator
from envs.reward_functions import RewardCalculator


class ContainerPackingEnv(gym.Env):
    def __init__(self, cfg, resolution=0.1, client_id=None):
        super().__init__()
        if client_id is None:
            raise RuntimeError("PyBullet client_id 未提供，請先在外部 p.connect() 再傳進來")

        self.client_id = client_id
        self.container = ContainerEnv(cfg, client_id=client_id)
        self.container.reset()

        # 工具
        self.candidate_gen = CandidateGenerator(self.container, resolution=resolution)
        self.reward_calc = RewardCalculator(resolution=0.02, wasted_threshold=0.2)

        # 候選點上限（動態 Discrete）
        self.max_candidates = 100
        self.action_space = gym.spaces.Discrete(self.max_candidates)

        # 狀態空間設計
        # (1) 當前箱子尺寸 (3 維)
        # (2) 候選點特徵 (每個候選點 6 維 → x, y, z, coverage, left_support, right_support)
        # (3) 當前已放置數量 (1 維)
        obs_dim = 3 + self.max_candidates * 6 + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 狀態追蹤
        self.placed = []
        self.candidates = []
        self.current_box = None
        self.last_waste = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 清空狀態
        self.placed = []
        self.last_waste = 0.0

        # 建立虛擬箱子 (不 spawn，只拿尺寸)
        self.current_box = Box(0.4, 0.4, 0.4, body_id=None, client_id=self.client_id)  # TODO: 可改隨機

        # 生成候選點
        self.candidates = self.candidate_gen.generate(self.current_box, self.placed)

        return self._get_obs(), {}

    def step(self, action):
        if action >= len(self.candidates):
            return self._get_obs(), -1.0, True, False, {"error": "invalid action"}

        # 選定候選點
        x, y, z, info = self.candidates[action]
        new_box = Box.spawn(
            self.current_box.l, self.current_box.w, self.current_box.h,
            pos=(x, y), client_id=self.client_id, mass=0.0
        )
        new_box.set_position((x, y, z))
        self.placed.append(new_box)

        # --- 計算 flatness ---
        hm = self.candidate_gen.top_view.get_heightmap()
        x_min = int((x - self.current_box.l / 2 + self.container.length / 2) / self.candidate_gen.res)
        x_max = int((x + self.current_box.l / 2 + self.container.length / 2) / self.candidate_gen.res)
        y_min = int((y - self.current_box.w / 2 + self.container.width / 2) / self.candidate_gen.res)
        y_max = int((y + self.current_box.w / 2 + self.container.width / 2) / self.candidate_gen.res)

        flatness = self.reward_calc.compute_flatness(
            self.candidate_gen.top_view, (x_min, x_max, y_min, y_max)
        )

        # --- 計算 wasted space ---
        wasted_after = self.reward_calc.compute_wasted_space(self.container, self.candidate_gen.top_view)

        # --- reward ---
        reward, terminated = self.reward_calc.compute_step_reward(
            success=True,
            flatness=flatness,
            wasted_before=self.last_waste,
            wasted_after=wasted_after,
            done=False
        )
        self.last_waste = wasted_after

        truncated = False  # 這裡暫時不用 max_steps，可以在外部控制

        # 下一個箱子 (虛擬，不 spawn)
        if not terminated:
            self.current_box = Box(0.4, 0.4, 0.4, body_id=None, client_id=self.client_id)  # TODO: 可改隨機
            self.candidates = self.candidate_gen.generate(self.current_box, self.placed)

        return self._get_obs(), reward, terminated, truncated, {"candidates": len(self.candidates)}

    def _get_obs(self):
        # (1) 當前箱子尺寸
        box_feat = np.array([self.current_box.l, self.current_box.w, self.current_box.h], dtype=np.float32)

        # (2) 候選點特徵
        cand_feats = []
        for (x, y, z, info) in self.candidates[:self.max_candidates]:
            cand_feats.append([
                x, y, z,
                info.get("coverage", 0.0),
                float(info.get("left_support", False)),
                float(info.get("right_support", False)),
            ])
        # padding
        if len(cand_feats) < self.max_candidates:
            cand_feats.extend([[0.0] * 6] * (self.max_candidates - len(cand_feats)))
        cand_feats = np.array(cand_feats, dtype=np.float32).flatten()

        # (3) 已放置數量
        placed_feat = np.array([len(self.placed)], dtype=np.float32)

        # 合併成最終 observation
        obs = np.concatenate([box_feat, cand_feats, placed_feat])
        return obs


if __name__ == "__main__":
    import json
    import pybullet as p
    import pybullet_data

    # 連線到 PyBullet (用 GUI 模式方便看)
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # 載入 container 配置
    with open("data/container_specs/container_20ft.json") as f:
        container_env = json.load(f)
    container_cfg = container_env["container"]

    # 初始化環境
    env = ContainerPackingEnv(container_cfg, resolution=0.1, client_id=client_id)
    obs, _ = env.reset()
    print("初始 observation shape:", obs.shape)

    # 連續執行 5 步
    for step in range(100):
        action = np.random.randint(len(env.candidates))  # 隨機挑一個候選點
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step+1}: action={action}, reward={reward:.3f}, terminated={terminated}, obs_shape={obs.shape}")
        if terminated or truncated:
            break

    p.disconnect(client_id)
