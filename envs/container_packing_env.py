# envs/candidate_generator_virtual_boxes.py
import numpy as np
import pybullet as p


class TopViewMap:
    def __init__(self, container_length, container_width, resolution=0.01):
        self.res = float(resolution)
        self.nx = int(round(container_length / self.res))
        self.ny = int(round(container_width  / self.res))
        self.length = float(container_length)
        self.width  = float(container_width)
        self.heightmap = np.zeros((self.ny, self.nx), dtype=np.float32)

    def reset(self):
        self.heightmap.fill(0.0)

    def update_from_boxes(self, boxes):
        hm = self.heightmap
        L, W, res = self.length, self.width, self.res
        for box in boxes:
            surf = box.get_top_surface()
            z_top = float(surf["z"])
            x0 = max(0, int((surf["x_min"] + L/2) / res))
            x1 = min(self.nx, int((surf["x_max"] + L/2) / res))
            y0 = max(0, int((surf["y_min"] + W/2) / res))
            y1 = min(self.ny, int((surf["y_max"] + W/2) / res))
            if x0 < x1 and y0 < y1:
                region = hm[y0:y1, x0:x1]
                np.maximum(region, z_top, out=region)

    def get_heightmap(self):
        return self.heightmap


class CandidateGeneratorVirtualBoxes:
    def __init__(self, container, resolution=0.02,
                 support_tol=0.02, min_coverage=0.5,
                 debug=False):
        self.container   = container
        self.res         = float(resolution)
        self.top_view    = TopViewMap(container.length, container.width, self.res)
        self.support_tol = float(support_tol)
        self.min_cov     = float(min_coverage)
        self.debug       = bool(debug)

    @staticmethod
    def _aabb_overlap(a, b, tol=1e-6):
        return not (
            a["xmax"] <= b["xmin"] + tol or a["xmin"] >= b["xmax"] - tol or
            a["ymax"] <= b["ymin"] + tol or a["ymin"] >= b["ymax"] - tol or
            a["zmax"] <= b["zmin"] + tol or a["zmin"] >= b["zmax"] - tol
        )

    def _region_metrics(self, hm, x_idx, y_idx, lx, wy):
        ny, nx = hm.shape
        if x_idx < 0 or y_idx < 0 or x_idx + lx > nx or y_idx + wy > ny:
            return None, None
        region = hm[y_idx:y_idx+wy, x_idx:x_idx+lx]
        if region.size == 0:
            return None, None
        base_h = float(region.max())
        cov    = float((np.abs(region - base_h) <= self.support_tol).mean())
        return base_h, cov

    def _virtual_walls(self, L, W, H):
        """虛擬四面牆，用極薄 box 表示"""
        eps = 1e-6
        return [
            {"x_min": -L/2, "x_max": -L/2+eps, "y_min": -W/2, "y_max": W/2, "z": H, "id": "wall_left"},
            {"x_min":  L/2-eps, "x_max":  L/2, "y_min": -W/2, "y_max": W/2, "z": H, "id": "wall_right"},
            {"y_min": -W/2, "y_max": -W/2+eps, "x_min": -L/2, "x_max": L/2, "z": H, "id": "wall_back"},
            {"y_min":  W/2-eps, "y_max":  W/2, "x_min": -L/2, "x_max": L/2, "z": H, "id": "wall_front"},
        ]

    def update_map(self, boxes):
        self.top_view.reset()
        self.top_view.update_from_boxes(boxes)

    def generate(self, box, placed_boxes):
        L, W, H = self.container.length, self.container.width, self.container.height
        res = self.res
        self.update_map(placed_boxes)
        hm  = self.top_view.get_heightmap()
        ny, nx = hm.shape

        box_l, box_w, box_h = float(box.l), float(box.w), float(box.h)
        lx = max(1, int(round(box_l / res)))
        wy = max(1, int(round(box_w / res)))

        candidates = []
        added = set()

        def try_add(x_idx, y_idx, edge_tag, src_id, left=False, right=False, back=False, front=False):
            key = (x_idx, y_idx, src_id)
            if key in added:
                return
            base_h, cov = self._region_metrics(hm, x_idx, y_idx, lx, wy)
            if base_h is None or cov < self.min_cov:
                return
            z_center = base_h + box_h/2
            if z_center + box_h/2 > H + 1e-9:
                return
            # 世界座標
            x_center = (x_idx * res) - L/2 + box_l/2
            y_center = (y_idx * res) - W/2 + box_w/2
            # 碰撞檢查
            cand = {"xmin": x_center - box_l/2, "xmax": x_center + box_l/2,
                    "ymin": y_center - box_w/2, "ymax": y_center + box_w/2,
                    "zmin": z_center - box_h/2, "zmax": z_center + box_h/2}
            for b in placed_boxes:
                s = b.get_top_surface()
                bb = {"xmin": s["x_min"], "xmax": s["x_max"],
                      "ymin": s["y_min"], "ymax": s["y_max"],
                      "zmin": s["z"] - b.h, "zmax": s["z"]}
                if self._aabb_overlap(cand, bb):
                    return
            info = {"base_height": base_h, "coverage": cov,
                    "left_contact": left, "right_contact": right,
                    "back_contact": back, "front_contact": front,
                    "edge_tag": edge_tag, "source": src_id}
            candidates.append((x_center, y_center, z_center, info))
            added.add(key)

        # === 把牆壁也當作虛擬箱子 ===
        all_boxes = []
        all_boxes.extend(placed_boxes)
        for wall in self._virtual_walls(L, W, H):
            dummy = type("Dummy", (), {"get_top_surface": lambda self, d=wall: d, "h": H})()
            dummy._src_id = wall["id"]
            all_boxes.append(dummy)

        # === 對每個 box (牆或真箱) 生成候選 ===
        for b in all_boxes:
            s = b.get_top_surface()
            src_id = getattr(b, "_src_id", id(b))
            bx0 = int(round((s["x_min"] + L/2) / res))
            bx1 = int(round((s["x_max"] + L/2) / res))
            by0 = int(round((s["y_min"] + W/2) / res))
            by1 = int(round((s["y_max"] + W/2) / res))

            # 後邊 (y-)
            y_idx = by0 - wy
            if y_idx >= 0:
                try_add(bx0 - lx, y_idx, "back_left", src_id, left=True, back=True)
                try_add(bx0,      y_idx, "back", src_id, back=True)
                try_add(bx1,      y_idx, "back_right", src_id, right=True, back=True)

            # 前邊 (y+)
            y_idx = by1
            if y_idx + wy <= ny:
                try_add(bx0 - lx, y_idx, "front_left", src_id, left=True, front=True)
                try_add(bx0,      y_idx, "front", src_id, front=True)
                try_add(bx1,      y_idx, "front_right", src_id, right=True, front=True)

            # 左邊 (x-)
            x_idx = bx0 - lx
            if x_idx >= 0:
                try_add(x_idx, by0, "left_back", src_id, left=True, back=True)
                try_add(x_idx, by1 - wy, "left_front", src_id, left=True, front=True)

            # 右邊 (x+)
            x_idx = bx1
            if x_idx + lx <= nx:
                try_add(x_idx, by0, "right_back", src_id, right=True, back=True)
                try_add(x_idx, by1 - wy, "right_front", src_id, right=True, front=True)

        return candidates


# ---------------- 測試 ----------------
if __name__ == "__main__":
    import json, random
    import pybullet_data
    from envs.container_env import ContainerEnv, Box

    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    with open("data/container_specs/container_20ft.json") as f:
        cfg = json.load(f)["container"]

    container = ContainerEnv(cfg, client_id=client_id)
    container.reset()

    gen = CandidateGeneratorVirtualBoxes(container, resolution=0.05)

    placed = []
    for step in range(500):
        # 🎲 隨機尺寸 (0.2–0.6 m)
        box_l = random.uniform(0.2, 0.6)
        box_w = random.uniform(0.2, 0.6)
        box_h = random.uniform(0.2, 0.6)
        box = Box(box_l, box_w, box_h, body_id=None, client_id=client_id)

        cands = gen.generate(box, placed)
        print(f"\nStep {step+1}, 候選數量: {len(cands)}")
        for i, (x, y, z, info) in enumerate(cands[:8]):
            print(f"[{i}] pos=({x:.2f},{y:.2f},{z:.2f}), tag={info['edge_tag']}, src={info['source']}")

        if not cands:
            print("⚠️ 沒有候選點")
            break

        x, y, z, info = random.choice(cands)
        new_box = Box.spawn(box_l, box_w, box_h, pos=(x, y), client_id=client_id)
        new_box.set_position((x, y, z))
        placed.append(new_box)

        for _ in range(60):
            p.stepSimulation()

    while True:
        p.stepSimulation()
