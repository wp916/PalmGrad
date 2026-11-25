"""实验用原生几何前端：MuJoCo只做运动学/碰撞，Nabla负责全部动力学。

输入HandState/PalmMotion；输出标准ContactFrame及显式pair编号。
几何JVP采用局部切空间中心差分，检查接触点数量、匹配及法向变化。
这是CPU正确性桥，不是CUDA碰撞或解析几何导数；不调用mj_forward/mj_step/求解器。
"""

import threading
import numpy as np
import mujoco
from scipy.optimize import linear_sum_assignment
from nablaengine.model.hand import HandState
from .contacts import ContactFrame
from .collision_adjoint import CollisionGeometryJVP


class MujocoCollisionBridge:
    def __init__(self, model, layout, *, max_contacts=128):
        self.model = model
        self.layout = layout
        self.data = mujoco.MjData(model)
        self.max_contacts = max_contacts
        self._lock = threading.RLock()
        self._closed = False
        self.calls = 0
        if np.any(model.geom_contype) or np.any(model.geom_conaffinity):
            raise ValueError("仅支持显式pair，必须关闭隐式碰撞")
        self.pairs = {
            tuple(sorted((int(a), int(b)))): p
            for p, (a, b) in enumerate(zip(model.pair_geom1, model.pair_geom2))
        }
        if len(self.pairs) != model.npair:
            raise ValueError("重复pair")
        if np.any(model.pair_margin) or np.any(model.pair_gap):
            raise ValueError("仅支持零margin/gap")

    def close(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __call__(self, state, palm):
        contacts, pairs, _ = self.evaluate_with_slots(state, palm)
        return contacts, pairs

    def evaluate_with_slots(self, state, palm):
        with self._lock:
            if self._closed:
                raise RuntimeError("碰撞桥已关闭")
            self.layout.validate_state(state)
            self.data.qpos[:], self.data.qvel[:] = self.layout.unpack(state)
            mocap = int(self.model.body_mocapid[self.layout.palm_body_id])
            if mocap < 0:
                raise ValueError("碰撞桥要求mocap手掌")
            self.data.mocap_pos[mocap] = palm.position
            self.data.mocap_quat[mocap] = palm.quaternion
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_collision(self.model, self.data)
            groups = {p: [] for p in range(self.model.npair)}
            for contact in self.data.contact:
                if contact.dist >= 0:
                    continue
                actual = (int(contact.geom[0]), int(contact.geom[1]))
                pair = self.pairs[tuple(sorted(actual))]
                expected = (int(self.model.pair_geom1[pair]), int(self.model.pair_geom2[pair]))
                frame = contact.frame.reshape(3, 3).copy()
                if actual != expected:
                    frame[[0, 2]] *= -1
                groups[pair].append(
                    ContactFrame(
                        int(self.model.geom_bodyid[expected[0]]),
                        int(self.model.geom_bodyid[expected[1]]),
                        contact.pos.copy(),
                        frame,
                        float(contact.dist),
                    )
                )
            contacts = []
            pairs = []
            slots = []
            for pair, points in groups.items():
                # 本次MuJoCo输出顺序保留；扰动后的点由JVP按空间位置一一匹配。
                for slot, contact in enumerate(points):
                    contacts.append(contact)
                    pairs.append(pair)
                    slots.append(slot)
            if len(contacts) > self.max_contacts:
                raise ValueError("碰撞容量溢出，未截断")
            self.calls += 1
            return contacts, np.asarray(pairs, np.int32), np.asarray(slots, np.int32)


class FiniteDifferenceCollisionJVP:
    """显式数值几何导数；只在点数及匹配稳定时返回，不冻结几何。

    差分从局部单位方向步长2e-6开始，并用半步长核对；不稳定时逐次缩小10倍，
    最多5组步长，每个状态缓存整个nv维几何Jacobian；全部失败时明确报错。
    位置切空间：自由物体平移世界系、旋转局部右乘，沿用hand引擎约定。
    """

    def __init__(self, provider, *, epsilon=2e-6, diagnostics_dir=None):
        self.provider = provider
        self.layout = provider.layout
        self.epsilon = epsilon
        self._cache_key = None
        self._cache = None
        self.max_scaled_disagreement = 0.0
        self.failures = 0
        self.adaptive_reductions = 0
        self.minimum_epsilon = epsilon
        self.diagnostics_dir = diagnostics_dir

    def retract(self, state, direction, epsilon):
        q = state.qpos.copy()
        for block in self.layout.blocks:
            qi, vi = block.qpos_start, block.qvel_start
            if block.kind == "hinge":
                q[qi] += epsilon * direction[vi]
            else:
                q[qi : qi + 3] += epsilon * direction[vi : vi + 3]
                velocity = np.zeros(6)
                velocity[3:] = direction[vi + 3 : vi + 6]
                mujoco.mj_integratePos(self._free_model, q[qi : qi + 7], velocity, epsilon)
        return HandState(q, state.qvel)

    @property
    def _free_model(self):
        if not hasattr(self, "_quaternion_model"):
            self._quaternion_model = mujoco.MjModel.from_xml_string(
                '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".01"/></body></worldbody></mujoco>'
            )
        return self._quaternion_model

    @staticmethod
    def packed(contacts):
        return np.asarray(
            [np.r_[c.distance, c.position, c.frame.ravel()] for c in contacts], dtype=float
        ).reshape(-1, 13)

    def aligned(self, baseline, pairs, other, other_pairs):
        if len(baseline) != len(other):
            raise ValueError("数值几何导数跨越接触数量分支")
        aligned = np.empty((len(baseline), 13))
        for pair in np.unique(pairs):
            ids = np.flatnonzero(pairs == pair)
            other_ids = np.flatnonzero(other_pairs == pair)
            if len(ids) != len(other_ids):
                raise ValueError("数值几何导数跨越pair激活分支")
            cost = np.array(
                [
                    [np.linalg.norm(baseline[i].position - other[j].position) for j in other_ids]
                    for i in ids
                ]
            )
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                i, j = ids[row], other_ids[col]
                if cost[row, col] > 0.0002 or baseline[i].frame[0] @ other[j].frame[0] < 0.999:
                    raise ValueError("接触点/法向匹配不稳定")
                aligned[i] = np.r_[other[j].distance, other[j].position, other[j].frame.ravel()]
        return aligned

    def evaluate(self, state, palm, direction):
        direction = np.asarray(direction, float)
        key = (state.qpos.tobytes(), palm.position.tobytes(), palm.quaternion.tobytes())
        contacts, pairs, slots = self.provider.evaluate_with_slots(state, palm)
        count = len(contacts)
        branch = np.zeros(count, np.int32)
        if not np.any(direction) or count == 0:
            derivative = np.zeros((count, 13))
        else:
            if key != self._cache_key:
                jacobian = np.empty((count, 13, self.layout.nv))
                for axis in range(self.layout.nv):
                    basis = np.zeros(self.layout.nv)
                    basis[axis] = 1.0
                    accepted = False
                    last_error = None
                    for reduction in range(5):
                        base_epsilon = self.epsilon * 10.0 ** (-reduction)
                        try:
                            estimates = []
                            for epsilon in (base_epsilon, base_epsilon / 2):
                                values = []
                                for sign in (1, -1):
                                    other, other_pairs, _ = self.provider.evaluate_with_slots(
                                        self.retract(state, basis, sign * epsilon), palm
                                    )
                                    values.append(self.aligned(contacts, pairs, other, other_pairs))
                                estimates.append((values[0] - values[1]) / (2 * epsilon))
                            scaled = float(
                                np.max(abs(estimates[0] - estimates[1]) / (1 + abs(estimates[1])))
                            )
                            if scaled > 2e-4:
                                raise ValueError(f"几何两步长差分不一致: {scaled}")
                            self.max_scaled_disagreement = max(self.max_scaled_disagreement, scaled)
                            self.adaptive_reductions += reduction
                            self.minimum_epsilon = min(self.minimum_epsilon, base_epsilon / 2)
                            accepted = True
                            break
                        except ValueError as error:
                            last_error = error
                    if not accepted:
                        self.failures += 1
                        if self.diagnostics_dir is not None:
                            from pathlib import Path

                            path = Path(self.diagnostics_dir)
                            path.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(
                                path / "geometry_failure.npz",
                                qpos=state.qpos,
                                qvel=state.qvel,
                                axis=axis,
                                palm_position=palm.position,
                                palm_quaternion=palm.quaternion,
                            )
                        raise ValueError(f"所有差分步长均不稳定，axis={axis}: {last_error}")
                    jacobian[:, :, axis] = estimates[1]
                self._cache_key = key
                self._cache = jacobian
            derivative = np.einsum("cik,k->ci", self._cache, direction)
            # 中心差分有微小正交性舍入误差；投影到SO(3)合法切空间，不改变前向架。
            for contact_index, contact in enumerate(contacts):
                frame_delta = derivative[contact_index, 4:].reshape(3, 3)
                angular = frame_delta @ contact.frame.T
                frame_delta[:] = 0.5 * (angular - angular.T) @ contact.frame
        return CollisionGeometryJVP(
            tuple(contacts),
            pairs,
            slots,
            branch,
            derivative[:, 0],
            derivative[:, 1:4],
            derivative[:, 4:].reshape(count, 3, 3),
            0.0,
        )
