import numpy as np

class TopViewMap:
    def __init__(self, container_length, container_width, resolution=0.002):
        self.res = resolution
        self.nx = int(container_length / resolution)
        self.ny = int(container_width / resolution)
        self.length = container_length
        self.width = container_width
        self.heightmap = np.zeros((self.ny, self.nx))

    def reset(self):
        self.heightmap[:] = 0

    def update_from_boxes(self, boxes):
        for box in boxes:
            surf = box.get_top_surface()
            z = surf["z"]

            x_min_idx = int((surf["x_min"] + self.length / 2) / self.res)
            x_max_idx = int((surf["x_max"] + self.length / 2) / self.res)
            y_min_idx = int((surf["y_min"] + self.width / 2) / self.res)
            y_max_idx = int((surf["y_max"] + self.width / 2) / self.res)

            self.heightmap[y_min_idx:y_max_idx, x_min_idx:x_max_idx] = z

    def get_heightmap(self):
        return self.heightmap


class CandidateGenerator:
    def __init__(self, container, resolution=0.002):
        self.container = container
        self.res = resolution
        self.top_view = TopViewMap(container.length, container.width, resolution)

    def update_map(self, boxes):
        self.top_view.reset()
        self.top_view.update_from_boxes(boxes)

    def generate(self, box, placed_boxes):
        candidates = []
        self.update_map(placed_boxes)
        hm = self.top_view.get_heightmap()

        box_l, box_w, box_h = box.l, box.w, box.h
        lx = int(box_l / self.res)
        wy = int(box_w / self.res)
        eps = self.res

        for y in range(self.top_view.ny - wy):
            for x in range(self.top_view.nx - lx):
                region = hm[y:y+wy, x:x+lx]
                base_height = np.max(region)

                support = np.sum(region == base_height)
                coverage = support / (lx * wy)

                # 算實際座標
                x_m = (x * self.res) - self.container.length/2 + box_l/2
                y_m = (y * self.res) - self.container.width/2 + box_w/2
                z_m = base_height + box_h / 2  # 第一次

                # z_m = base_height + 0.001

                # 箱子邊界
                x_min, x_max = x_m - box_l/2, x_m + box_l/2
                y_min, y_max = y_m - box_w/2, y_m + box_w/2

                # ----------- 前牆 or 前箱檢查 -----------
                is_front_wall = (
                    self.container.walls.get("front", False) and abs(y_min + self.container.width/2) <= eps
                )
                is_front_box = False
                for b in placed_boxes:
                    surf = b.get_top_surface()
                    # y_min 與某箱子的 y_max 對齊，且 x 範圍有重疊
                    if abs(y_min - surf["y_max"]) <= eps and not (x_max <= surf["x_min"] or x_min >= surf["x_max"]):
                        is_front_box = True
                        break

                if not (is_front_wall or is_front_box):
                    continue  # 必須靠前牆或前箱

                # ----------- 左右支撐檢查 -----------
                has_left = False
                has_right = False
                for b in placed_boxes:
                    surf = b.get_top_surface()
                    # 左支撐
                    if abs(x_min - surf["x_max"]) <= eps and not (y_max <= surf["y_min"] or y_min >= surf["y_max"]):
                        has_left = True
                    # 右支撐
                    if abs(x_max - surf["x_min"]) <= eps and not (y_max <= surf["y_min"] or y_min >= surf["y_max"]):
                        has_right = True

                # ----------- 動態覆蓋率 -----------
                min_cov = 0.5 if (has_left or has_right) else 0.75

                if coverage >= min_cov:
                    z_min, z_max = z_m - box_h / 2, z_m + box_h / 2
                    overlap = False
                    for b in placed_boxes:
                        surf = b.get_top_surface()
                        b_xmin, b_xmax = surf["x_min"], surf["x_max"]
                        b_ymin, b_ymax = surf["y_min"], surf["y_max"]
                        b_zmin, b_zmax = surf["z"] - b.h, surf["z"]

                        # 3D AABB 檢查：如果沒有完全錯開 → 重疊
                        if not (x_max <= b_xmin or x_min >= b_xmax or
                                y_max <= b_ymin or y_min >= b_ymax or
                                z_max <= b_zmin or z_min >= b_zmax):
                            overlap = True
                            break

                    if overlap:
                        continue  # 🚫 跳過這個候選點

                    z_top = z_m + box_h / 2
                    if z_top > self.container.height:
                        continue

                    candidates.append((x_m, y_m, z_m, {
                        "coverage": coverage,
                        "front_wall": is_front_wall,
                        "front_box": is_front_box,
                        "left_support": has_left,
                        "right_support": has_right
                    }))

        return candidates


if __name__ == "__main__":
    import json
    import pybullet as p
    import pybullet_data
    from envs.container_env import ContainerEnv, Box
    from envs.candidate_generator import CandidateGenerator

    # 啟動 PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # 載入 container 配置
    with open("data/container_specs/container_20ft.json") as f:
        container_env = json.load(f)
    container_cfg = container_env["container"]

    # 初始化 container
    container = ContainerEnv(container_cfg)
    container.reset()

    # 初始化候選生成器
    gen = CandidateGenerator(container, resolution=0.1)

    # 已放的箱子
    placed = []

    # 連續放 5 個箱子
    for i in range(120):
        # 新箱子規格 (這裡固定尺寸，之後你也可以改成隨機/不同尺寸)
        dummy_box = Box(0.4, 0.4, 0.4, None)

        # 生成候選點
        candidates = gen.generate(dummy_box, placed)
        print(f"\nStep {i+1}, 候選數量: {len(candidates)}")

        if not candidates:
            print("⚠️ 沒有候選點，停止放置")
            break

        # 選一個候選點 (這裡簡單取第一個，你也可以改成隨機 random.choice)
        x, y, z, info = candidates[0]
        print(f"選擇點: ({x:.2f}, {y:.2f}, {z:.2f}), info={info}")

        # 真正生成箱子 (z 已經是中心高度，不要再加 h/2)
        new_box = Box.spawn(0.4, 0.4, 0.4, pos=(x, y))
        new_box.set_position((x, y, z))
        placed.append(new_box)

    print(f"\n最終放置了 {len(placed)} 個箱子")

    while True:
        p.stepSimulation()
        pass


