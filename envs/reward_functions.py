import numpy as np

class RewardCalculator:
    def __init__(self, beta=50.0, gamma=5.0, resolution=0.02):
        """
        beta: 最終填充率獎勵權重
        gamma: 未使用空間懲罰權重
        resolution: 高度圖解析度 (m)
        """
        self.beta = beta
        self.gamma = gamma
        self.res = resolution

    def compute_fill_rate(self, container, boxes):
        """計算裝載率 (0~1)"""
        container_volume = container.length * container.width * container.height
        if container_volume <= 0:
            return 0.0
        total_box_volume = sum([b.l * b.w * b.h for b in boxes])
        return total_box_volume / container_volume

    def compute_wasted_space(self, container, topview_map):
        hm = topview_map.get_heightmap()
        if hm.size == 0:
            return 0.0
        max_height = np.max(hm)
        wasted_volume = np.sum((max_height - hm) * (self.res ** 2))
        container_volume = container.length * container.width * container.height
        return wasted_volume / container_volume

    def compute_step_reward(self, success, wasted_before, wasted_after, done=False, final_fill=0.0):
        """
        success: 是否成功放置
        wasted_before: 放置前浪費比例
        wasted_after: 放置後浪費比例
        done: 是否 episode 結束
        final_fill: episode 結束時填充率
        """
        if not success:
            return -1.0, True

        reward = 0.0

        # 本次放置造成的新增浪費
        delta_waste = wasted_after - wasted_before
        reward -= delta_waste * self.gamma

        # episode 結束時 → 裝載率大獎勵
        if done:
            reward += final_fill * self.beta

        return reward, done

