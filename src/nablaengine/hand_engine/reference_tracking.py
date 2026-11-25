"""真实手物参考轨迹的Mimic任务接口：观测、奖励、reset和动作目标。

输入：原生顺序参考qpos/qvel及每帧所属片段末端；原关节限位内PD目标。
输出：可供原NablaMimic rollout/actor/critic使用的状态与任务函数。
物理状态只由physics.step推进；参考只用于reset、观测、奖励和PD目标，不覆写仿真物体。
"""

from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from .mimic_interface import HandMimicInterface, NativeHandData, FixedHandContext
from .pose_tracking import rotation6


class ReferenceTrackingState(NamedTuple):
    dx: NativeHandData
    frame: object
    ep: object
    prev_a: object
    xi: FixedHandContext
    key: object


class HandReferenceTracking(HandMimicInterface):
    def __init__(
        self,
        physics,
        action_map,
        *,
        reference_qpos,
        reference_qvel,
        reference_last,
        control_dt,
        max_distance=0.15,
        velocity_feedforward=False,
    ):
        super().__init__(physics, action_map, control_dt=control_dt)
        q = np.asarray(reference_qpos, float)
        v = np.asarray(reference_qvel, float)
        last = np.asarray(reference_last, np.int32)
        if (
            q.shape != (len(q), self.layout.source_nq)
            or v.shape != (len(q), self.layout.source_nv)
            or last.shape != (len(q),)
        ):
            raise ValueError("参考数组形状不一致")
        if (
            not np.isfinite(q).all()
            or not np.isfinite(v).all()
            or np.any(last < np.arange(len(q)))
            or np.any(last >= len(q))
        ):
            raise ValueError("参考数值/片段边界不合法")
        self.reference_qpos = jnp.asarray(q)
        self.reference_qvel = jnp.asarray(v)
        self.ref_last = jnp.asarray(last)
        self.object = next(block for block in self.layout.blocks if block.kind == "free_object")
        if sum(block.kind == "free_object" for block in self.layout.blocks) != 1:
            raise ValueError("当前轨迹任务只支持一个物体")
        self.jq = np.array([b.source_qpos_start for b in self.layout.blocks if b.kind == "hinge"])
        self.jv = np.array([b.source_qvel_start for b in self.layout.blocks if b.kind == "hinge"])
        self.nu = action_map.nu
        self.n_frames = len(q)
        self.action_scale = jnp.asarray(action_map.scale)
        self.hist_dim = 3 * self.nu + 15
        self.goal_dim = 2 * self.nu + 9
        self.critic_dim = self.hist_dim + self.goal_dim
        self.max_distance = float(max_distance)
        self.velocity_feedforward = bool(velocity_feedforward)
        self.velocity_feedforward_ratio = jnp.asarray(physics.kv / physics.kp)
        self.reset_margin = 0.001  # 避免参考恰落限位时，初始位置导数处于激活边界。

    def reset(self, key, frame):
        frame = jnp.asarray(frame, jnp.int32)
        return ReferenceTrackingState(
            NativeHandData(
                self.reference_qpos[frame]
                .at[self.jq]
                .set(
                    jnp.clip(
                        self.reference_qpos[frame, self.jq],
                        jnp.asarray(self.action_map.lower) + self.reset_margin,
                        jnp.asarray(self.action_map.upper) - self.reset_margin,
                    )
                ),
                self.reference_qvel[frame],
                jnp.zeros(self.layout.source_nv, jnp.float64),
            ),
            frame,
            jnp.int32(0),
            jnp.zeros(self.nu, jnp.float64),
            FixedHandContext(jnp.float64(1.0), jnp.zeros(3)),
            jax.random.key_data(key),
        )

    def ctrl_of(self, action, frame=None):
        target_frame = jnp.minimum(frame + 1, self.ref_last[frame])
        # 参考只给PD目标；物体始终自由运动，不执行物体位姿覆盖。
        target = self.reference_qpos[target_frame, self.jq]
        if self.velocity_feedforward:
            # kp*(qref-q)+kv*(vref-v) 与此位置目标偏置严格等价；物理内核不变。
            target = (
                target
                + self.velocity_feedforward_ratio * self.reference_qvel[target_frame, self.jv]
            )
        return jnp.clip(
            target + self.action_scale * action,
            jnp.asarray(self.action_map.lower),
            jnp.asarray(self.action_map.upper),
        )

    def policy_obs(self, state, key):
        qi = self.object.source_qpos_start
        vi = self.object.source_qvel_start
        p = state.dx.qpos[qi : qi + 3]
        quat = state.dx.qpos[qi + 3 : qi + 7]
        target = jnp.minimum(state.frame + 1, self.ref_last[state.frame])
        qref = self.reference_qpos[target]
        vref = self.reference_qvel[target]
        history = jnp.concatenate(
            (
                state.dx.qpos[self.jq],
                0.1 * state.dx.qvel[self.jv],
                p / 0.1,
                rotation6(quat),
                0.1 * state.dx.qvel[vi : vi + 6],
                state.prev_a,
            )
        )
        goal = jnp.concatenate(
            (
                qref[self.jq] - state.dx.qpos[self.jq],
                (qref[qi : qi + 3] - p) / 0.05,
                rotation6(qref[qi + 3 : qi + 7]),
                0.1 * vref[self.jv],
            )
        )
        return history, goal

    def critic_obs(self, state):
        history, goal = self.policy_obs(state, None)
        return jnp.concatenate((history, goal))

    def errors(self, state):
        reference = self.reference_qpos[state.frame]
        qi = self.object.source_qpos_start
        position = jnp.sum((state.dx.qpos[qi : qi + 3] - reference[qi : qi + 3]) ** 2)
        rotation = jnp.maximum(
            0.0, 1.0 - jnp.dot(state.dx.qpos[qi + 3 : qi + 7], reference[qi + 3 : qi + 7]) ** 2
        )
        joints = jnp.mean((state.dx.qpos[self.jq] - reference[self.jq]) ** 2)
        return position, rotation, joints

    def reward(self, state, action, prev_action, prev=None):
        position, rotation, joints = self.errors(state)
        velocity = jnp.mean(
            (state.dx.qvel[self.jv] - self.reference_qvel[state.frame, self.jv]) ** 2
        )
        terms = dict(
            position=jnp.exp(-position / 0.02**2),
            orientation=jnp.exp(-rotation / 0.3**2),
            joint_pose=jnp.exp(-joints / 0.2**2),
            joint_velocity=0.2 * jnp.exp(-velocity / 2.0**2),
            action_rate=-0.001 * jnp.mean((action - prev_action) ** 2),
        )
        return self.dt * sum(terms.values()), terms

    def terminated(self, state):
        position, _, _ = self.errors(state)
        return (~jnp.all(jnp.isfinite(self.critic_obs(state)))) | (position > self.max_distance**2)

    def clip_ended(self, state):
        return state.frame >= self.ref_last[state.frame]

    def finish_physics(self, state, action, qpos, qvel, warm):
        return state._replace(
            dx=NativeHandData(qpos, qvel, warm),
            frame=jnp.minimum(state.frame + 1, self.ref_last[state.frame]),
            ep=state.ep + 1,
            prev_a=action,
        )

    def hazard_features(self, state):
        raise NotImplementedError("不用人体终止惩罚")

    @staticmethod
    def compact(state):
        return state._asdict()

    @staticmethod
    def restore(compact):
        return ReferenceTrackingState(**compact)
