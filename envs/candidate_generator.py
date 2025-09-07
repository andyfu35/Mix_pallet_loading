# envs/candidate_generator.py
import numpy as np


class TopViewMap:
    """
    Heightmap：每個 cell 存「當前最高頂面 z」。
    resolution: 每格邊長 (m)。
    """
    def __init__(self, container_length, container_width, resolution=0.01):
        self.res = float(resolution)
        self.length = float(container_length)
        self.width  = float(container_width)
        self.nx = int(round(self.length  / self.res))
        self.ny = int(round(self.width   / self.res))
        self.heightmap = np.zeros((self.ny, self.nx), dtype=np.float32)

    def reset(self):
        self.heightmap.fill(0.0)

    def update_from_boxes(self, boxes):
        """
        把已放箱子的頂面投影到 heightmap（取區域最大值）。
        """
        hm, L, W, res = self.heightmap, self.length, self.width, self.res
        for box in boxes:
            s = box.get_top_surface()  # {x_min,x_max,y_min,y_max,z(頂面)}
            z = float(s["z"])
            x0 = max(0, int((s["x_min"] + L/2) / res))
            x1 = min(self.nx, int((s["x_max"] + L/2) / res))
            y0 = max(0, int((s["y_min"] + W/2) / res))
            y1 = min(self.ny, int((s["y_max"] + W/2) / res))
            if x0 < x1 and y0 < y1:
                region = hm[y0:y1, x0:x1]
                np.maximum(region, z, out=region)

    def get_heightmap(self):
        return self.heightmap


class CandidateGenerator:
    """
    純高度圖候選生成：
      - 第一顆固定右前角
      - 之後：必須「前 or 左 or 右」至少一邊貼（牆或同高箱）
      - 當底面高度 >= h_push 時，做通道檢查：
          從候選 footprint 的『後緣』(y+wy) 到容器後牆 (ny) 必須淨空
    """
    def __init__(
        self,
        container,
        resolution=0.01,        # heightmap 解析度（m/格）
        scan_step_m=0.005,      # 掃描步長（世界座標 m）
        support_tol=0.02,       # 同層容忍 (±m)
        contact_tol=None,       # 接觸容忍，若 None 則取 max(resolution, 0.005)
        h_push=1.4,             # 只在底面高度 >= 此值時做通道檢查
        require_first_corner=True,
        first_corner="right_front",
        side_contact_ratio=0.5, # 條帶同層比例門檻（貼邊判定）
        min_coverage=0.5,       # 支撐率門檻
        debug=False,
    ):
        self.container = container
        self.res = float(resolution)
        self.top_view = TopViewMap(container.length, container.width, self.res)

        self.scan_step_m = max(1e-4, float(scan_step_m))
        self.support_tol = float(support_tol)
        self.contact_tol = float(contact_tol) if contact_tol is not None else max(self.res, 0.005)
        self.h_push = float(h_push)

        self.require_first_corner = bool(require_first_corner)
        assert first_corner in ("left_back", "left_front", "right_back", "right_front")
        self.first_corner = first_corner

        self.side_contact_ratio = float(side_contact_ratio)
        self.min_coverage = float(min_coverage)
        self.debug = bool(debug)

        # ✅ 地板/鄰接條帶的高度門檻：小於這個高度不算“鄰接箱子”
        #   取 max( 0.5*support_tol, 0.01m )：避免把 z≈0 的地板當成貼邊
        self.floor_epsilon = max(0.5 * self.support_tol, 0.01)
        #   條帶若要算“鄰接箱面”，高度必須 >= 這個值
        self.min_neighbor_z = self.floor_epsilon

    # ---------- 工具 ----------

    @staticmethod
    def _overlap_1d(a_min, a_max, b_min, b_max):
        return not (a_max <= b_min or a_min >= b_max)

    @staticmethod
    def _aabb_overlap(a, b, tol=1e-6):
        return not (
            a["xmax"] <= b["xmin"] + tol or a["xmin"] >= b["xmax"] - tol or
            a["ymax"] <= b["ymin"] + tol or a["ymin"] >= b["ymax"] - tol or
            a["zmax"] <= b["zmin"] + tol or a["zmin"] >= b["zmax"] - tol
        )

    def _first_corner_xy(self, box_l, box_w, pad):
        """
        第一顆固定角（預設右前），pad 用於避免浮點貼牆抖動。
        """
        L, W = self.container.length, self.container.width
        fc = self.first_corner
        if fc == "left_back":
            x0 = -L / 2 + box_l / 2 + pad
            y0 =  W / 2 - box_w / 2 - pad
        elif fc == "left_front":
            x0 = -L / 2 + box_l / 2 + pad
            y0 = -W / 2 + box_w / 2 + pad
        elif fc == "right_back":
            x0 =  L / 2 - box_l / 2 - pad
            y0 =  W / 2 - box_w / 2 - pad
        else:  # right_front
            x0 =  L / 2 - box_l / 2 - pad
            y0 = -W / 2 + box_w / 2 + pad
        return x0, y0

    # ---------- 主流程 ----------

    def update_map(self, boxes):
        self.top_view.reset()
        self.top_view.update_from_boxes(boxes)

    def generate(self, box, placed_boxes):
        L, W, H = self.container.length, self.container.width, self.container.height
        res = self.res
        sup = self.support_tol

        box_l, box_w, box_h = float(box.l), float(box.w), float(box.h)

        # ✅ 第一顆箱子固定角
        if self.require_first_corner and not placed_boxes:
            pad = self.res * 0.5
            x0, y0 = self._first_corner_xy(box_l, box_w, pad)
            z0 = box_h / 2
            info0 = {
                "first_box": True,
                "coverage": 1.0,
                "left_contact":  False,
                "right_contact": True,
                "front_contact": True,   # 右前角等價於前牆/右牆貼邊
                "front_clear":   True,
                "base_height":   0.0,
                "box_region":    None,
            }
            return [(x0, y0, z0, info0)]

        # 常規掃描
        candidates = []
        self.update_map(placed_boxes)
        hm = self.top_view.get_heightmap()
        ny, nx = hm.shape

        step_x_cells = max(1, int(round(self.scan_step_m / res)))
        step_y_cells = max(1, int(round(self.scan_step_m / res)))
        lx = max(1, int(round(box_l / res)))
        wy = max(1, int(round(box_w / res)))

        # 掃描 footprint
        for y in range(0, max(1, ny - wy + 1), step_y_cells):
            for x in range(0, max(1, nx - lx + 1), step_x_cells):
                region = hm[y:y + wy, x:x + lx]
                if region.size == 0:
                    continue

                # 支撐率（以區域最大值為 base）
                base_height = float(region.max())
                support_mask = (np.abs(region - base_height) <= sup)
                coverage = float(support_mask.sum()) / float(region.size)
                if coverage < self.min_coverage:
                    continue

                # 世界座標中心
                x_m = (x * res) - L / 2 + box_l / 2
                y_m = (y * res) - W / 2 + box_w / 2
                z_m = base_height + box_h / 2

                # 高度上限
                if z_m + box_h / 2 > H + 1e-9:
                    continue

                # ---------- AABB 碰撞（保守） ----------
                x_min, x_max = x_m - box_l / 2, x_m + box_l / 2
                y_min, y_max = y_m - box_w / 2, y_m + box_w / 2
                cand_aabb = {
                    "xmin": x_min, "xmax": x_max,
                    "ymin": y_min, "ymax": y_max,
                    "zmin": z_m - box_h / 2, "zmax": z_m + box_h / 2,
                }
                overlap = False
                for b in placed_boxes:
                    s = b.get_top_surface()
                    b_aabb = {
                        "xmin": s["x_min"], "xmax": s["x_max"],
                        "ymin": s["y_min"], "ymax": s["y_max"],
                        "zmin": s["z"] - b.h, "zmax": s["z"],
                    }
                    if self._aabb_overlap(cand_aabb, b_aabb):
                        overlap = True
                        break
                if overlap:
                    continue

                # ---------- 三向貼邊（牆或同層箱） ----------
                # 牆用「索引貼牆」判定；貼箱用「高度圖鄰接條帶」比例判定。
                on_left_wall  = (x == 0)
                on_right_wall = (x + lx == nx)
                on_front_wall = (y == 0)

                # 條帶比例：除同層(±sup)外，還需 stripe 高度 >= min_neighbor_z（避免把地板當成鄰接箱）
                def stripe_ratio_left():
                    xs = x - 1
                    if xs < 0 or wy <= 0:
                        return 0.0
                    stripe = hm[y:y+wy, xs:xs+1]
                    same_level = (np.abs(stripe - base_height) <= sup)
                    occupied   = (stripe >= self.min_neighbor_z)
                    ok = same_level & occupied
                    return float(ok.sum()) / float(stripe.size)

                def stripe_ratio_right():
                    xs = x + lx
                    if xs >= nx or wy <= 0:
                        return 0.0
                    stripe = hm[y:y+wy, xs:xs+1]
                    same_level = (np.abs(stripe - base_height) <= sup)
                    occupied   = (stripe >= self.min_neighbor_z)
                    ok = same_level & occupied
                    return float(ok.sum()) / float(stripe.size)

                def stripe_ratio_front():
                    ys = y - 1
                    if ys < 0 or lx <= 0:
                        return 0.0
                    stripe = hm[ys:ys+1, x:x+lx]
                    same_level = (np.abs(stripe - base_height) <= sup)
                    occupied   = (stripe >= self.min_neighbor_z)
                    ok = same_level & occupied
                    return float(ok.sum()) / float(stripe.size)

                # 🚫 第一層（base≈0）時，禁止用條帶判貼邊，避免把“地板=0”當成貼邊
                if base_height <= self.floor_epsilon:
                    left_ok  = on_left_wall
                    right_ok = on_right_wall
                    front_ok = on_front_wall
                    left_ratio = right_ratio = front_ratio = 0.0
                else:
                    left_ratio  = stripe_ratio_left()
                    right_ratio = stripe_ratio_right()
                    front_ratio = stripe_ratio_front()
                    left_ok  = on_left_wall  or (left_ratio  >= self.side_contact_ratio)
                    right_ok = on_right_wall or (right_ratio >= self.side_contact_ratio)
                    front_ok = on_front_wall or (front_ratio >= self.side_contact_ratio)

                # 至少一邊要成立
                if not (left_ok or right_ok or front_ok):
                    continue

                # ---------- 前向通道檢查（沿 +y；僅底面 >= h_push 啟用） ----------
                bottom_h = z_m - box_h / 2
                front_clear = True
                if bottom_h >= self.h_push - 1e-9:
                    y_rear = y + wy
                    if y_rear < ny:
                        corridor = hm[y_rear:ny, x:x+lx]
                        if corridor.size > 0:
                            clearance = max(self.support_tol, self.contact_tol)
                            front_clear = float(corridor.max()) < (bottom_h - clearance)
                    if not front_clear:
                        continue

                info = {
                    "coverage": coverage,
                    "base_height": base_height,
                    "left_contact":  bool(left_ok),
                    "right_contact": bool(right_ok),
                    "front_contact": bool(front_ok),
                    "front_clear":   bool(front_clear),
                    "box_region": (x, x + lx, y, y + wy),
                    "left_ratio":  float(left_ratio)  if base_height > self.floor_epsilon else 0.0,
                    "right_ratio": float(right_ratio) if base_height > self.floor_epsilon else 0.0,
                    "front_ratio": float(front_ratio) if base_height > self.floor_epsilon else 0.0,
                }
                candidates.append((x_m, y_m, z_m, info))

        return candidates


# ------------------------- 測試主程式 -------------------------
if __name__ == "__main__":
    import json
    import random
    import pybullet as p
    import pybullet_data
    from envs.container_env import ContainerEnv, Box

    # 啟動 GUI
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # 載入貨櫃
    with open("data/container_specs/container_20ft.json") as f:
        cfg = json.load(f)["container"]

    container = ContainerEnv(cfg, client_id=client_id)
    container.reset()

    # 建立候選生成器
    gen = CandidateGenerator(
        container,
        resolution=0.01,        # 1 cm 高度圖
        scan_step_m=0.005,      # 5 mm 掃描步長
        support_tol=0.02,       # ±2 cm 同層容忍
        contact_tol=0.01,       # 1 cm 接觸容忍（用在通道裕度）
        h_push=1.0,             # 底面 >= 1.0 m 才檢查通道
        require_first_corner=True,
        first_corner="right_front",
        side_contact_ratio=0.5, # 條帶≥50% 同層且高度>=min_neighbor_z 才算貼邊
        min_coverage=0.5,
        debug=False,
    )

    placed = []
    # 放置 40 箱（固定 30cm 立方；可改成隨機尺寸做壓測）
    for i in range(200):
        lwh = (0.30, 0.30, 0.30)
        probe = Box(*lwh, body_id=None, client_id=client_id)

        cands = gen.generate(probe, placed)
        print(f"\nStep {i+1}: 候選數={len(cands)}")
        if not cands:
            print("⚠️ 沒有候選點，停止")
            break

        # 建議：若想鼓勵往上疊，可用 max(..., key=lambda c: c[3]['base_height'])
        x, y, z, info = random.choice(cands)
        # x, y, z, info = cands[0]
        print(f"放置: x={x:.3f}, y={y:.3f}, z={z:.3f}, info={info}")

        new_box = Box.spawn(*lwh, pos=(x, y), client_id=client_id)
        new_box.set_position((x, y, z))
        placed.append(new_box)

        for _ in range(60):
            p.stepSimulation()

    print(f"\n最終放置數：{len(placed)}")
    while True:
        p.stepSimulation()
