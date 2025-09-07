import os
import pytest
import pybullet as p
import pybullet_data
import json
import time

from envs.container_env import ContainerEnv, Box
from envs.reward_functions import RewardCalculator
from envs.candidate_generator import CandidateGenerator


@pytest.fixture(scope="module")
def setup_env():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    json_path = os.path.join(base_dir, "data", "container_specs", "container_20ft.json")

    with open(json_path) as f:
        container_env = json.load(f)
    container_cfg = container_env["container"]

    p.connect(p.GUI)  # ⚠️ 改成 GUI
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    container = ContainerEnv(container_cfg)
    container.reset()

    calc = RewardCalculator(resolution=0.02, wasted_threshold=0.2)
    gen = CandidateGenerator(container, resolution=0.1)

    yield container, calc, gen
    p.disconnect()


def test_three_different_boxes_visual(setup_env):
    container, calc, gen = setup_env
    placed = []

    box_sizes = [
        (0.3, 0.3, 0.3),
        (0.4, 0.4, 0.4),
        (0.4, 0.4, 0.4),
        (0.15, 0.15, 0.15),
        (0.2, 0.2, 0.2),
    ]

    wasted_before = 0.0
    for i, size in enumerate(box_sizes):
        dummy_box = Box(*size, body_id=None)
        candidates = gen.generate(dummy_box, placed)
        assert candidates, f"Step {i+1}: 沒有候選點"

        x, y, z, info = candidates[0]
        new_box = Box.spawn(*size, pos=(x, y))
        new_box.set_position((x, y, z))
        placed.append(new_box)

        wasted_after = calc.compute_wasted_space(container, gen.top_view)
        reward, done = calc.compute_step_reward(
            success=True,
            wasted_before=wasted_before,
            wasted_after=wasted_after,
            done=False
        )
        wasted_before = wasted_after

        print(f"Step {i+1}: Box={size}, reward={reward:.3f}, waste={wasted_after:.3f}, done={done}")
        assert done is False

    fill_rate = calc.compute_fill_rate(container, placed)
    print(f"Final fill rate = {fill_rate:.3f}")

    # 🔴 保持可視化視窗
    for _ in range(500):
        p.stepSimulation()
        time.sleep(1. / 60.)
