import numpy as np

class TopViewMap2P75D:
    def __init__(self, container_length, container_width, resolution=0.05):
        self.res = float(resolution)
        self.nx = int(round(container_length / self.res))
        self.ny = int(round(container_width / self.res))
        self.length = float(container_length)
        self.width = float(container_width)

        # 每個 cell 存一個 list of (z_min, z_max)
        self.cells = [[[] for _ in range(self.nx)] for _ in range(self.ny)]

    def reset(self):
        for y in range(self.ny):
            for x in range(self.nx):
                self.cells[y][x] = []

    def update_from_boxes(self, boxes):
        L, W, res = self.length, self.width, self.res
        for box in boxes:
            surf = box.get_top_surface()
            z_top = float(surf["z"])
            z_bottom = z_top - box.h

            x_min_idx = max(0, int((surf["x_min"] + L/2) / res))
            x_max_idx = min(self.nx, int((surf["x_max"] + L/2) / res))
            y_min_idx = max(0, int((surf["y_min"] + W/2) / res))
            y_max_idx = min(self.ny, int((surf["y_max"] + W/2) / res))

            for yi in range(y_min_idx, y_max_idx):
                for xi in range(x_min_idx, x_max_idx):
                    self.cells[yi][xi].append((z_bottom, z_top))

        # 每個 cell 的區間排序 & 合併
        for yi in range(self.ny):
            for xi in range(self.nx):
                segs = sorted(self.cells[yi][xi], key=lambda s: s[0])
                merged = []
                for z0,z1 in segs:
                    if not merged or z0 > merged[-1][1] + 1e-6:
                        merged.append([z0,z1])
                    else:
                        merged[-1][1] = max(merged[-1][1], z1)
                self.cells[yi][xi] = merged

    def get_intervals(self, x_idx, y_idx):
        """回傳該 cell 目前的 z 占用區間列表"""
        return self.cells[y_idx][x_idx]

    def get_highest(self, x_idx, y_idx):
        """回傳該 cell 的最高 z_max"""
        if not self.cells[y_idx][x_idx]:
            return 0.0
        return max(z1 for _,z1 in self.cells[y_idx][x_idx])

class CandidateGenerator:
    def __init__(self, container, resolution=0.01, min_coverage=0.5):
        self.container = container
        self.res = float(resolution)
        self.min_cov = float(min_coverage)

        self.top_view = TopViewMap2P75D(container.length, container.width, self.res)

    @staticmethod
    def _aabb_overlap(a, b, tol=1e-6):
        """AABB 重疊檢查"""
        return not (
            a["xmax"] <= b["xmin"] + tol or a["xmin"] >= b["xmax"] - tol or
            a["ymax"] <= b["ymin"] + tol or a["ymin"] >= b["ymax"] - tol or
            a["zmax"] <= b["zmin"] + tol or a["zmin"] >= b["zmax"] - tol
        )

    def update_map(self, placed_boxes):
        """根據目前已放的箱子更新高度圖"""
        self.top_view.update_from_boxes(placed_boxes)

    def generate(self, box, placed_boxes):
        """
        輸入：
            box: 要嘗試放的新箱 (Box)
            placed_boxes: 已放置的箱子列表
        輸出：
            候選點列表，每個元素 = (x, y, z, info)
        """
        L, W, H = self.container.length, self.container.width, self.container.height
        box_l, box_w, box_h = box.l, box.w, box.h

        # 更新高度圖
        self.update_map(placed_boxes)
        hm = self.top_view

        candidates = []

        # ============ 這裡由你自己寫生成邏輯 ============
        # 範例流程（建議）：
        # 1. 遍歷可能的 anchor (例如牆、已放箱子邊界)
        # 2. 計算候選 (x_center, y_center)
        # 3. 查詢 2.75D 高度圖 → 算出 base_h
        #       base_h = hm.get_support_height(x_idx, y_idx, box_h)
        #       z_center = base_h + box_h/2
        # 4. 檢查 AABB 是否與 placed_boxes 重疊
        # 5. 如果合法 → candidates.append((x,y,z,info))
        #
        # 你可以在 info 裡存：
        #   - base_height
        #   - contacts (front/side)
        #   - edge_tag (靠左牆/右牆/靠箱子)
        #   - coverage (覆蓋率)
        # =================================================

        return candidates
