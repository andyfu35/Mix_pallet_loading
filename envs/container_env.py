import json
import pybullet as p
import pybullet_data


class ContainerEnv:
    def __init__(self, cfg: dict, client_id: int):
        if client_id is None:
            raise RuntimeError("ContainerEnv 初始化失敗：請先在外部呼叫 p.connect()，並傳入 client_id")

        self.client_id = client_id
        self.length = cfg["length"]
        self.width = cfg["width"]
        self.height = cfg["height"]
        self.t = cfg.get("wall_thickness", 0.05)
        self.pose = cfg.get("pose", [0, 0, 0])

        # 酒紅半透明 RGBA
        self.rgba = [0.5, 0.0, 0.125, 0.3]

        # 牆的資訊
        self.walls = {
            "left": True,
            "right": True,
            "front": True,
            "back": False
        }

        self._create_container()

    def _create_container(self):
        x, y, z = self.pose
        l, w, h, t = self.length, self.width, self.height, self.t

        # 地板
        floor_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[l / 2, w / 2, t / 2], physicsClientId=self.client_id
        )
        floor_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[l / 2, w / 2, t / 2], rgbaColor=self.rgba, physicsClientId=self.client_id
        )
        self.floor_id = p.createMultiBody(
            baseCollisionShapeIndex=floor_shape,
            baseVisualShapeIndex=floor_vis,
            basePosition=[x, y, z - t / 2],
            physicsClientId=self.client_id
        )

        # 左右牆
        wall_shape_x = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[t / 2, w / 2, h / 2], physicsClientId=self.client_id
        )
        wall_vis_x = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[t / 2, w / 2, h / 2], rgbaColor=self.rgba, physicsClientId=self.client_id
        )

        left_wall = p.createMultiBody(
            baseCollisionShapeIndex=wall_shape_x,
            baseVisualShapeIndex=wall_vis_x,
            basePosition=[x - l / 2 - t / 2, y, z + h / 2],
            physicsClientId=self.client_id
        )
        right_wall = p.createMultiBody(
            baseCollisionShapeIndex=wall_shape_x,
            baseVisualShapeIndex=wall_vis_x,
            basePosition=[x + l / 2 + t / 2, y, z + h / 2],
            physicsClientId=self.client_id
        )

        # 前牆
        wall_shape_y = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[l / 2, t / 2, h / 2], physicsClientId=self.client_id
        )
        wall_vis_y = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[l / 2, t / 2, h / 2], rgbaColor=self.rgba, physicsClientId=self.client_id
        )
        front_wall = p.createMultiBody(
            baseCollisionShapeIndex=wall_shape_y,
            baseVisualShapeIndex=wall_vis_y,
            basePosition=[x, y - w / 2 - t / 2, z + h / 2],
            physicsClientId=self.client_id
        )

        # 後牆不要建 → 開口

        # 天花板（可選，這裡先保留）
        ceiling_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[l / 2, w / 2, t / 2], physicsClientId=self.client_id
        )
        ceiling_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[l / 2, w / 2, t / 2], rgbaColor=self.rgba, physicsClientId=self.client_id
        )
        ceiling = p.createMultiBody(
            baseCollisionShapeIndex=ceiling_shape,
            baseVisualShapeIndex=ceiling_vis,
            basePosition=[x, y, z + h + t / 2],
            physicsClientId=self.client_id
        )

        self.wall_ids = [left_wall, right_wall, front_wall, ceiling]

    def get_ids(self):
        return [self.floor_id] + self.wall_ids

    def reset(self):
        # 刪掉舊的 container
        for obj_id in self.get_ids():
            if obj_id is not None:
                p.removeBody(obj_id, physicsClientId=self.client_id)
        # 重建新的 container
        self._create_container()


class Box:
    def __init__(self, l, w, h, body_id, client_id: int):
        self.l = l
        self.w = w
        self.h = h
        self.body_id = body_id
        self.client_id = client_id

    @classmethod
    def spawn(cls, l, w, h, pos, client_id: int,
              color=[0.8, 0.3, 0.3, 1.0], auto_ground=True, mass=0.0):
        """
        建立並回傳一個 Box 物件
        l, w, h: 尺寸 (m)
        pos: (x, y)
        auto_ground: True 時，會自動讓 z = h/2
        mass: 0.0 = 靜態, >0 = 動態
        """
        x, y = pos
        z = h / 2 if auto_ground else 0.0

        # 建立碰撞與視覺形狀
        shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[l / 2, w / 2, h / 2], physicsClientId=client_id
        )
        visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[l / 2, w / 2, h / 2], rgbaColor=color, physicsClientId=client_id
        )

        # 在 PyBullet 建立實體
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=visual,
            basePosition=(x, y, z),
            physicsClientId=client_id
        )

        return cls(l, w, h, body_id, client_id)

    def get_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id)
        return pos

    def get_top_surface(self):
        x, y, z = self.get_position()
        return {
            "z": z + self.h / 2,
            "x_min": x - self.l / 2,
            "x_max": x + self.l / 2,
            "y_min": y - self.w / 2,
            "y_max": y + self.w / 2,
        }

    def set_position(self, pos, orn=None):
        if orn is None:
            _, orn = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, orn, physicsClientId=self.client_id)

    def remove(self):
        p.removeBody(self.body_id, physicsClientId=self.client_id)


if __name__ == "__main__":
    with open("data/container_specs/container_20ft.json") as f:
        container_env = json.load(f)

    container_cfg = container_env["container"]

    # === 外層統一 connect ===
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)

    container = ContainerEnv(container_cfg, client_id=client_id)
    container.reset()

    box1 = Box.spawn(0.4, 0.4, 0.4, pos=(0, 0), client_id=client_id)
    box2 = Box.spawn(0.5, 0.3, 0.2, pos=(0.5, 0.5), client_id=client_id)

    while True:
        p.stepSimulation(physicsClientId=client_id)
