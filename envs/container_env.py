import json
import pybullet as p
import pybullet_data

with open("data/container_specs/container_20ft.json") as f:
    container_env = json.load(f)

container_cfg = container_env["container"]

class ContainerEnv:
    def __init__(self, cfg: dict):
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
        floor_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[l / 2, w / 2, t / 2])
        floor_vis   = p.createVisualShape(p.GEOM_BOX, halfExtents=[l / 2, w / 2, t / 2], rgbaColor=self.rgba)
        self.floor_id = p.createMultiBody(
            baseCollisionShapeIndex=floor_shape,
            baseVisualShapeIndex=floor_vis,
            basePosition=[x, y, z - t / 2]
        )

        # 左右牆
        wall_shape_x = p.createCollisionShape(p.GEOM_BOX, halfExtents=[t / 2, w / 2, h / 2])
        wall_vis_x   = p.createVisualShape(p.GEOM_BOX, halfExtents=[t / 2, w / 2, h / 2], rgbaColor=self.rgba)

        left_wall = p.createMultiBody(
            baseCollisionShapeIndex=wall_shape_x,
            baseVisualShapeIndex=wall_vis_x,
            basePosition=[x - l / 2 - t / 2, y, z + h / 2]
        )
        right_wall = p.createMultiBody(
            baseCollisionShapeIndex=wall_shape_x,
            baseVisualShapeIndex=wall_vis_x,
            basePosition=[x + l / 2 + t / 2, y, z + h / 2]
        )

        # 前牆
        wall_shape_y = p.createCollisionShape(p.GEOM_BOX, halfExtents=[l / 2, t / 2, h / 2])
        wall_vis_y   = p.createVisualShape(p.GEOM_BOX, halfExtents=[l / 2, t / 2, h / 2], rgbaColor=self.rgba)
        front_wall = p.createMultiBody(
            baseCollisionShapeIndex=wall_shape_y,
            baseVisualShapeIndex=wall_vis_y,
            basePosition=[x, y - w / 2 - t / 2, z + h / 2]
        )

        # 後牆不要建 → 開口

        # 天花板
        ceiling_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[l / 2, w / 2, t / 2])
        ceiling_vis   = p.createVisualShape(p.GEOM_BOX, halfExtents=[l / 2, w / 2, t / 2], rgbaColor=self.rgba)
        ceiling = p.createMultiBody(
            baseCollisionShapeIndex=ceiling_shape,
            baseVisualShapeIndex=ceiling_vis,
            basePosition=[x, y, z + h + t / 2]
        )

        self.wall_ids = [left_wall, right_wall, front_wall, ceiling]

    def get_ids(self):
        return [self.floor_id] + self.wall_ids

    def reset(self):
        for obj_id in self.get_ids():
            if obj_id is not None:
                p.removeBody(obj_id)
        self._create_container()

class Box:
    def __init__(self, l, w, h, body_id):
        self.l = l
        self.w = w
        self.h = h
        self.body_id = body_id

    @classmethod
    def spawn(cls, l, w, h, pos, color=[0.8, 0.3, 0.3, 1.0], auto_ground=True):
        """
        建立並回傳一個 Box 物件
        l, w, h: 尺寸 (m)
        pos: (x, y)
        auto_ground: True 時，會自動讓 z = h/2
        """
        x, y = pos
        z = h / 2 if auto_ground else 0.0

        # 建立碰撞與視覺形狀
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[l/2, w/2, h/2])
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[l/2, w/2, h/2], rgbaColor=color)

        # 在 PyBullet 建立實體
        body_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=visual,
            basePosition=(x, y, z)
        )

        return cls(l, w, h, body_id)

    def get_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return pos

    def get_top_surface(self):
        x, y, z = self.get_position()
        return {
            "z": z + self.h/2,
            "x_min": x - self.l/2,
            "x_max": x + self.l/2,
            "y_min": y - self.w/2,
            "y_max": y + self.w/2,
        }

    def set_position(self, pos, orn=None):
        if orn is None:
            _, orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, orn)

    def remove(self):
        p.removeBody(self.body_id)

if __name__ == "__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # p.setGravity(0, 0, 0)
    plane_id = p.loadURDF("plane.urdf")

    container = ContainerEnv(container_cfg)
    container.reset()

    box1 = Box.spawn(0.4, 0.4, 0.4, pos=(0, 0))
    box2 = Box.spawn(0.5, 0.3, 0.2, pos=(0.5, 0.5))

    while True:
        pass
