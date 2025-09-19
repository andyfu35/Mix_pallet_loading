import copy
import numpy as np
import pybullet as p

def snapshot_env(env):
    """保存環境狀態 (PyBullet + Python 部分)"""
    sid = p.saveState(physicsClientId=env.client_id)
    state = {
        "placed": copy.deepcopy(env.placed),
        "candidates": copy.deepcopy(env.candidates),
        "last_waste": env.last_waste,
        "current_box": copy.deepcopy(env.current_box),
    }
    return sid, state

def restore_env(env, sid, state):
    """恢復環境狀態 (先清理 rollout 新 spawn 的 box，再 restore)"""
    # 1. 找出 rollout 新增的 box
    if len(env.placed) > len(state["placed"]):
        new_boxes = env.placed[len(state["placed"]):]
        for b in new_boxes:
            try:
                if getattr(b, "body_id", None) is not None:
                    p.removeBody(b.body_id, physicsClientId=env.client_id)
            except Exception:
                pass

    # 2. 還原 Bullet 狀態
    p.restoreState(sid, physicsClientId=env.client_id)
    p.removeState(sid, physicsClientId=env.client_id)

    # 3. 還原 Python 狀態
    env.placed = state["placed"]
    env.candidates = state["candidates"]
    env.last_waste = state["last_waste"]
    env.current_box = state["current_box"]

def rollout_value(env, depth=2, gamma=0.99, beam_k=5):
    """
    短視野 lookahead：
    - 每次從 Top-K 候選動作展開
    - rollout depth 步，累加 reward
    """
    if len(env.candidates) == 0:
        return 0, -1e9

    cand_idxs = list(range(min(len(env.candidates), beam_k)))
    best_score, best_a = -1e9, 0

    for a0 in cand_idxs:
        sid, state = snapshot_env(env)

        score, discount = 0.0, 1.0
        done = False

        # 第 0 步
        _, reward, terminated, truncated, _ = env.step(a0)
        score += discount * reward
        discount *= gamma
        done = terminated or truncated

        # 再往前模擬 depth-1 步
        d = 1
        while d < depth and not done and len(env.candidates) > 0:
            a = np.random.randint(len(env.candidates))  # 最簡單: 隨機挑
            _, reward, terminated, truncated, _ = env.step(a)
            score += discount * reward
            discount *= gamma
            done = terminated or truncated
            d += 1

        # 回復現場
        restore_env(env, sid, state)

        if score > best_score:
            best_score, best_a = score, a0

    return best_a, best_score
