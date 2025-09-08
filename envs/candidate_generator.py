# envs/candidate_generator.py
import numpy as np


class TopViewMap:
    def __init__(self, container_length, container_width, resolution=0.01):
        self.res = float(resolution)
        self.nx = int(round(container_length / self.res))
        self.ny = int(round(container_width / self.res))
        self.length = float(container_length)
        self.width = float(container_width)
        self.heightmap = np.zeros((self.ny, self.nx), dtype=np.float32)

    def reset(self):
        self.heightmap.fill(0.0)

    def update_from_boxes(self, boxes):
        hm = self.heightmap
        L, W, res = self.length, self.width, self.res

        for box in boxes:
            surf = box.get_top_surface()
            z_top = float(surf["z"])

            x_min_idx = max(0, int((surf["x_min"] + L / 2) / res))
            x_max_idx = min(self.nx, int((surf["x_max"] + L / 2) / res))
            y_min_idx = max(0, int((surf["y_min"] + W / 2) / res))
            y_max_idx = min(self.ny, int((surf["y_max"] + W / 2) / res))

            if x_min_idx < x_max_idx and y_min_idx < y_max_idx:
                region = hm[y_min_idx:y_max_idx, x_min_idx:x_max_idx]
                # 寫入更高者，避免低高度覆蓋
                np.maximum(region, z_top, out=region)

    def get_heightmap(self):
        return self.heightmap


class CandidateGenerator:
    """
    僅「靠牆/切齊牆」的候選點生成器：
      - 左牆：x_idx = 0
      - 右牆：x_idx = nx - lx_cells
      - 前牆：y_idx = 0
    其餘規則：
      - 以高度圖 region 的 max 當作底面高度 base_height
      - coverage = (region 在 [base_height±support_tol] 的比例)
      - AABB 與既有箱體不重疊
      - 若底面高度 >= h_push，做「前向通道」淨空檢查（沿 +y 方向）
    """
    def __init__(
        self,
        container,
        resolution=0.01,
        edge_stride=0.02,     # 沿牆步距（m）
        support_tol=0.02,     # 視為同層的高差容忍
        contact_tol=0.01,     # AABB/通道的安全間隙
        min_coverage=0.5,     # 最小支撐覆蓋率
        h_push=1.0,           # 低於此底面高度不做前向通道檢查
        require_first_corner=True,
        first_corner="right_front",
    ):
        self.container = container
        self.res = float(resolution)
        self.top_view = TopViewMap(container.length, container.width, self.res)

        self.edge_stride = float(edge_stride)
        self.support_tol = float(support_tol)
        self.contact_tol = float(contact_tol)
        self.min_coverage = float(min_coverage)
        self.h_push = float(h_push)

        self.require_first_corner = bool(require_first_corner)
        assert first_corner in ("left_back", "left_front", "right_back", "right_front")
        self.first_corner = first_corner

    # --------- 工具 ---------
    @staticmethod
    def _aabb_overlap(a, b, tol=1e-6):
        return not (
            a["xmax"] <= b["xmin"] + tol or a["xmin"] >= b["xmax"] - tol or
            a["ymax"] <= b["ymin"] + tol or a["ymin"] >= b["ymax"] - tol or
            a["zmax"] <= b["zmin"] + tol or a["zmin"] >= b["zmax"] - tol
        )

    def _first_corner_xy(self, box_l, box_w, eps):
        L, W = self.container.length, self.container.width
        fc = self.first_corner
        if fc == "left_back":
            x0 = -L / 2 + box_l / 2 + eps
            y0 =  W / 2 - box_w / 2 - eps
        elif fc == "left_front":
            x0 = -L / 2 + box_l / 2 + eps
            y0 = -W / 2 + box_w / 2 + eps
        elif fc == "right_back":
            x0 =  L / 2 - box_l / 2 - eps
            y0 =  W / 2 - box_w / 2 - eps
        else:  # right_front
            x0 =  L / 2 - box_l / 2 - eps
            y0 = -W / 2 + box_w / 2 + eps
        return x0, y0

    # --------- 主流程 ---------
    def update_map(self, boxes):
        self.top_view.reset()
        self.top_view.update_from_boxes(boxes)

    def generate(self, box, placed_boxes):
        L, W, H = self.container.length, self.container.width, self.container.height
        res = self.res
        sup = self.support_tol

        box_l, box_w, box_h = float(box.l), float(box.w), float(box.h)

        # ✅ 第一顆箱子固定角（切齊牆與地板）
        if self.require_first_corner and not placed_boxes:
            eps = max(self.contact_tol, self.res * 0.5)
            x0, y0 = self._first_corner_xy(box_l, box_w, eps)
            z0 = box_h / 2
            info0 = {
                "first_box": True,
                "coverage": 1.0,
                "base_height": 0.0,
                "left_contact": (self.first_corner.startswith("left")),
                "right_contact": (self.first_corner.startswith("right")),
                "front_contact": (self.first_corner.endswith("front")),
                "front_clear": True,
                "box_region": None,
                "edge_tag": "first_corner",
            }
            return [(x0, y0, z0, info0)]

        # 更新高度圖
        self.update_map(placed_boxes)
        hm = self.top_view.get_heightmap()
        ny, nx = hm.shape

        # 尺寸轉成 cell
        lx = max(1, int(round(box_l / res)))
        wy = max(1, int(round(box_w / res)))
        step_cells = max(1, int(round(self.edge_stride / res)))

        candidates = []

        def region_and_metrics(x_idx, y_idx):
            if x_idx < 0 or y_idx < 0 or x_idx + lx > nx or y_idx + wy > ny:
                return None, None, None
            region = hm[y_idx:y_idx + wy, x_idx:x_idx + lx]
            if region.size == 0:
                return None, None, None
            base_h = float(region.max())
            cov = float((np.abs(region - base_h) <= sup).mean())
            return region, base_h, cov

        def corridor_clear(x_idx, y_idx, bottom_h):
            # 只在底面高度 >= h_push 時檢查，方向為 +y（從 footprint 後緣到後牆）
            if bottom_h < self.h_push - 1e-9:
                return True
            y_rear = y_idx + wy
            if y_rear >= ny:
                return True
            corridor = hm[y_rear:ny, x_idx:x_idx + lx]
            if corridor.size == 0:
                return True
            clearance = max(self.support_tol, self.contact_tol)
            return float(corridor.max()) < (bottom_h - clearance)

        def try_add(x_idx, y_idx, edge_tag, contact_flags):
            region, base_h, cov = region_and_metrics(x_idx, y_idx)
            if region is None or cov < self.min_coverage:
                return
            z_m = base_h + box_h / 2
            if z_m + box_h / 2 > H + 1e-9:
                return
            bottom_h = z_m - box_h / 2
            if not corridor_clear(x_idx, y_idx, bottom_h):
                return

            # 世界座標中心
            x_m = (x_idx * res) - L / 2 + box_l / 2
            y_m = (y_idx * res) - W / 2 + box_w / 2

            # AABB 與既有箱不重疊
            x_min, x_max = x_m - box_l / 2, x_m + box_l / 2
            y_min, y_max = y_m - box_w / 2, y_m + box_w / 2
            cand_aabb = {
                "xmin": x_min, "xmax": x_max,
                "ymin": y_min, "ymax": y_max,
                "zmin": bottom_h, "zmax": z_m + box_h / 2,
            }
            for b in placed_boxes:
                s = b.get_top_surface()
                b_aabb = {
                    "xmin": s["x_min"], "xmax": s["x_max"],
                    "ymin": s["y_min"], "ymax": s["y_max"],
                    "zmin": s["z"] - b.h, "zmax": s["z"],
                }
                if self._aabb_overlap(cand_aabb, b_aabb):
                    return

            info = {
                "coverage": cov,
                "base_height": base_h,
                "left_contact": bool(contact_flags.get("left", False)),
                "right_contact": bool(contact_flags.get("right", False)),
                "front_contact": bool(contact_flags.get("front", False)),
                "front_clear": True if bottom_h < self.h_push else corridor_clear(x_idx, y_idx, bottom_h),
                "box_region": (x_idx, x_idx + lx, y_idx, y_idx + wy),
                "edge_tag": edge_tag,
            }
            candidates.append((x_m, y_m, z_m, info))

        # ---------------- 牆邊掃描（切齊牆）----------------
        # 左牆 x_idx=0
        x_idx = 0
        for y_idx in range(0, max(1, ny - wy + 1), step_cells):
            try_add(x_idx, y_idx, "left_wall", {"left": True})
        # 右牆 x_idx=nx-lx
        x_idx = max(0, nx - lx)
        for y_idx in range(0, max(1, ny - wy + 1), step_cells):
            try_add(x_idx, y_idx, "right_wall", {"right": True})
        # 前牆 y_idx=0
        y_idx = 0
        for x_idx in range(0, max(1, nx - lx + 1), step_cells):
            try_add(x_idx, y_idx, "front_wall", {"front": True})

        # ---------------- 箱邊掃描（可選，預設啟用）----------------
        if getattr(self, "touch_box_edges", True):
            for b in placed_boxes:
                s = b.get_top_surface()
                # footprint 轉 index 範圍
                bx0 = max(0, int((s["x_min"] + L / 2) / res))
                bx1 = min(nx, int((s["x_max"] + L / 2) / res))
                by0 = max(0, int((s["y_min"] + W / 2) / res))
                by1 = min(ny, int((s["y_max"] + W / 2) / res))

                # 左邊貼齊：讓新箱 x_max 對齊 s["x_min"] → x_idx = bx0 - lx
                x_idx = bx0 - lx
                if x_idx >= 0:
                    for y_idx in range(by0, max(by0, by1 - wy + 1), step_cells):
                        try_add(x_idx, y_idx, "touch_left_box", {"left": True})

                # 右邊貼齊：讓新箱 x_min 對齊 s["x_max"] → x_idx = bx1
                x_idx = bx1
                if x_idx + lx <= nx:
                    for y_idx in range(by0, max(by0, by1 - wy + 1), step_cells):
                        try_add(x_idx, y_idx, "touch_right_box", {"right": True})

                # 前邊貼齊：讓新箱 y_max 對齊 s["y_min"] → y_idx = by0 - wy
                y_idx = by0 - wy
                if y_idx >= 0:
                    for x_idx in range(bx0, max(bx0, bx1 - lx + 1), step_cells):
                        try_add(x_idx, y_idx, "touch_front_box", {"front": True})

        return candidates

        # --------- 左牆：x_idx = 0，沿 y 掃描 ---------
        x_idx = 0
        for y_idx in range(0, max(1, ny - wy + 1), step_cells):
            add_candidate(x_idx, y_idx, "left_wall", {"left": True})

        # --------- 右牆：x_idx = nx - lx，沿 y 掃描 ---------
        x_idx = max(0, nx - lx)
        for y_idx in range(0, max(1, ny - wy + 1), step_cells):
            add_candidate(x_idx, y_idx, "right_wall", {"right": True})

        # --------- 前牆：y_idx = 0，沿 x 掃描 ---------
        y_idx = 0
        for x_idx in range(0, max(1, nx - lx + 1), step_cells):
            add_candidate(x_idx, y_idx, "front_wall", {"front": True})

        return candidates


# ----------------- 測試 -----------------
if __name__ == "__main__":
    import json
    import random
    import pybullet as p
    import pybullet_data
    from envs.container_env import ContainerEnv, Box

    # 啟動 PyBullet GUI
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # 載入貨櫃配置
    with open("data/container_specs/container_20ft.json") as f:
        container_env = json.load(f)
    container_cfg = container_env["container"]

    container = ContainerEnv(container_cfg, client_id=client_id)
    container.reset()

    # 只靠牆/切齊牆
    gen = CandidateGenerator(
        container,
        resolution=0.01,     # 高度圖解析度 1 cm
        edge_stride=0.02,    # 沿牆步距 2 cm
        support_tol=0.02,    # 同層容忍 ±2 cm
        contact_tol=0.01,    # 接觸容忍 1 cm
        min_coverage=0.5,
        h_push=1.0,
        require_first_corner=True,
        first_corner="right_front"
    )

    placed = []
    for i in range(600):
        # 測試箱（可改成隨機尺寸）
        dummy_box = Box(0.30, 0.30, 0.30, body_id=None, client_id=client_id)

        cands = gen.generate(dummy_box, placed)
        print(f"\nStep {i+1}, 候選數量: {len(cands)}")

        if not cands:
            print("⚠️ 沒有候選點，停止放置")
            break

        # 隨機選一個候選點
        idx = random.randrange(len(cands))
        # idx = 0
        x, y, z, info = cands[idx]
        print(f"選擇點[{idx}]: ({x:.3f}, {y:.3f}, {z:.3f}), cov={info['coverage']:.2f}, tag={info['edge_tag']}")

        # 生成箱子（z 已是中心高度）
        new_box = Box.spawn(0.30, 0.30, 0.30, pos=(x, y), client_id=client_id)
        new_box.set_position((x, y, z))
        placed.append(new_box)

        # for _ in range(60):
        #     p.stepSimulation()

    print(f"\n最終放置了 {len(placed)} 個箱子")
    while True:
        p.stepSimulation()
