import numpy as np

class RewardCalculator:
    """
    設計理念：
    - 主目標：裝載率 ↑（最重要）
      * per-step：Δfill_rate > 0 給獎勵（只獎勵正增量，避免倒扣干擾）
      * end-of-episode：final_fill 給額外大獎勵（推高終局裝載率）
    - 次目標：平整度好（std 小），否則扣分
    - 選配：浪費空間變多扣分、變少加少量分
    """

    def __init__(
        self,
        # === 權重 ===
        w_delta_fill=100.0,   # 每步 Δfill_rate 的權重（例：Δ=0.01 → +1.0 分）
        w_final_fill=50.0,    # 結束時 final_fill 的權重
        w_flat_pen=10.0,      # 平整度（std）扣分權重（std * w_flat_pen 直接扣）
        w_waste=2.0,          # 浪費變化權重（變多扣、變少加半）

        # === 平滑/裁剪 ===
        clip_delta_fill=(0.0, 0.05),  # 單步 Δfill 裁剪（避免一口氣過大）
        clip_flatness=(0.0, 0.05),    # 區域 std 裁剪（避免極端地形放大扣分）
        clip_delta_waste=(-0.1, 0.1), # 浪費變化裁剪

        # === 其他 ===
        resolution=0.02,      # 高度圖解析度（m）
        wasted_threshold=0.2  # 判定浪費的高度差（m）
    ):
        self.w_delta_fill = float(w_delta_fill)
        self.w_final_fill = float(w_final_fill)
        self.w_flat_pen   = float(w_flat_pen)
        self.w_waste      = float(w_waste)

        self.clip_delta_fill  = clip_delta_fill
        self.clip_flatness    = clip_flatness
        self.clip_delta_waste = clip_delta_waste

        self.res = float(resolution)
        self.wasted_threshold = float(wasted_threshold)

    # ====== 指標計算 ======
    def compute_fill_rate(self, container, boxes):
        total_vol = container.length * container.width * container.height
        if total_vol <= 0:
            return 0.0
        occupied_vol = sum(b.l * b.w * b.h for b in boxes)
        return float(occupied_vol / total_vol)

    def compute_flatness(self, topview_map, box_region):
        """
        回傳區域高度標準差（越小越平）。
        box_region: (x_min, x_max, y_min, y_max) 以 heightmap 索引為單位
        """
        hm = topview_map.get_heightmap()
        x_min, x_max, y_min, y_max = box_region
        if hm.size == 0 or x_min >= x_max or y_min >= y_max:
            return 0.0
        region = hm[y_min:y_max, x_min:x_max]
        std = float(np.std(region)) if region.size else 0.0
        if self.clip_flatness is not None:
            lo, hi = self.clip_flatness
            std = float(np.clip(std, lo, hi))
        return std

    def compute_wasted_space(self, container, topview_map):
        """
        以相對最高高度 max_h 為參考，高度差 > threshold 的地方視為浪費。
        估算浪費體積比例（相對容器體積）。
        """
        hm = topview_map.get_heightmap()
        if hm.size == 0:
            return 0.0
        max_h = float(np.max(hm))
        waste_area = ((max_h - hm) > self.wasted_threshold).astype(float)
        wasted_volume = float(np.sum((max_h - hm) * waste_area)) * (self.res ** 2)
        container_vol = container.length * container.width * container.height
        if container_vol <= 0:
            return 0.0
        return float(wasted_volume / container_vol)

    # ====== 單步獎勵 ======
    def compute_step_reward(
        self,
        *,
        success: bool,
        flatness: float,
        wasted_before: float,
        wasted_after: float,
        done: bool = False,
        final_fill: float = 0.0,
        fill_before: float = None,
        fill_after: float = None,
    ):
        """
        回傳 (reward, done, info_terms)
          - success: 這一步是否成功放置，若 False 建議環境處理 done 或懲罰
          - flatness: 區域高度 std（已可被裁剪）
          - wasted_before/after: 放前/後浪費比例
          - fill_before/fill_after: 用於計算 per-step Δfill
          - done: 是否為最後一步
          - final_fill: 若 done=True 時，作為終局獎勵依據
        """
        # 失敗的處理交給環境（這裡不直接結束），如需在這裡結束可改 True
        if not success:
            return 0.0, done, {"reason": "fail"}

        terms = {}
        reward = 0.0

        # ---- Δfill 主獎勵 ----
        r_fill = 0.0
        if fill_before is not None and fill_after is not None:
            delta_fill = float(fill_after - fill_before)
            delta_fill_pos = max(0.0, delta_fill)  # 只獎勵上升
            if self.clip_delta_fill is not None:
                lo, hi = self.clip_delta_fill
                delta_fill_pos = float(np.clip(delta_fill_pos, lo, hi))
            r_fill = self.w_delta_fill * delta_fill_pos
            terms["delta_fill"] = delta_fill_pos
        terms["r_fill"] = r_fill
        reward += r_fill

        # ---- 平整度扣分（std 越大扣越多）----
        std = float(flatness)
        if self.clip_flatness is not None:
            lo, hi = self.clip_flatness
            std = float(np.clip(std, lo, hi))
        r_flat_pen = - self.w_flat_pen * std
        terms["flatness_std"] = std
        terms["r_flat_pen"] = r_flat_pen
        reward += r_flat_pen

        # ---- 浪費變化 shaping（可小）----
        delta_waste = float(wasted_after - wasted_before)
        if self.clip_delta_waste is not None:
            lo, hi = self.clip_delta_waste
            delta_waste = float(np.clip(delta_waste, lo, hi))
        if delta_waste > 0:
            r_waste = - self.w_waste * delta_waste
        elif delta_waste < 0:
            r_waste = + (self.w_waste * 0.5) * abs(delta_waste)
        else:
            r_waste = 0.0
        terms["delta_waste"] = delta_waste
        terms["r_waste"] = r_waste
        reward += r_waste

        # ---- 結束時的最終裝載率獎勵 ----
        if done:
            r_final = self.w_final_fill * float(final_fill)
            terms["final_fill"] = float(final_fill)
            terms["r_final"] = r_final
            reward += r_final

        return float(reward), bool(done), terms
