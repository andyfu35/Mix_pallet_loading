import numpy as np

class RewardCalculator:
    def __init__(self, beta=50.0, gamma=5.0, resolution=0.02, wasted_threshold=0.2):
        """
        beta: 最終填充率獎勵權重
        gamma: step 懲罰權重，用於浪費面積
        resolution: 高度圖解析度 (m)
        wasted_threshold: 判定浪費的高度差 (m)
        """
        self.beta = beta
        self.gamma = gamma
        self.res = resolution
        self.wasted_threshold = wasted_threshold

    def compute_fill_rate(self, container, boxes):
        """計算裝載率 (0.0~1.0)"""
        total_vol = container.length * container.width * container.height
        if total_vol <= 0:
            return 0.0
        occupied_vol = sum(b.l * b.w * b.h for b in boxes)
        return occupied_vol / total_vol

    def compute_flatness(self, topview_map, box_region):
        """
        平整度指標：該區域內高度標準差，越平整越小
        box_region: (x_min, x_max, y_min, y_max) in heightmap grid indices
        """
        hm = topview_map.get_heightmap()
        x_min, x_max, y_min, y_max = box_region
        region = hm[y_min:y_max, x_min:x_max]
        return float(np.std(region)) if region.size else 0.0

    def compute_wasted_space(self, container, topview_map):
        """
        計算浪費空間比例 (不可利用區域占容量比例)
        以與最高高度的差值超過閾值的區域進行計算
        """
        hm = topview_map.get_heightmap()
        if hm.size == 0:
            return 0.0
        max_h = np.max(hm)
        waste_area = ((max_h - hm) > self.wasted_threshold).astype(float)
        wasted_volume = np.sum((max_h - hm) * waste_area) * (self.res ** 2)
        container_vol = container.length * container.width * container.height
        return wasted_volume / container_vol

    def compute_step_reward(self, success, flatness, wasted_before, wasted_after, done=False, final_fill=0.0):
        """
        測試箱子放置當下的 reward
        - flatness: 區域標準差，越小代表越平整
        - wasted_before/after: 放置前後浪費比例
        - done: 是否爲最後一步
        - final_fill: episode 結束時計算裝載率獎勵
        """
        if not success:
            return 0.0, True

        reward = 0.0
        # 正向：平整度越高（std 越小）越好，轉換為正向獎勵
        reward += (1.0 - flatness) if flatness <= 1 else 0.0

        # 負向：浪費增加則扣分
        delta_waste = wasted_after - wasted_before
        if delta_waste > 0:
            reward -= delta_waste * self.gamma
        elif delta_waste < 0:
            reward += abs(delta_waste) * (self.gamma * 0.5)

        # End-of-episode final fill reward
        if done:
            reward += final_fill * self.beta

        return reward, done
