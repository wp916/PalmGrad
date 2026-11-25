"""把关节约束与condim1/3/4接触组织成统一表，供现有Newton求解。

assemble输入单环境状态、手掌运动、活动接触和显式pair编号，输出ConstraintTable。
行顺序：关节frictionloss、活动限位、按输入接触顺序的1/4/6行。
接触几何/J/软参数在CUDA计算；本文件在CPU按地址拼接，不运行Newton或时间积分。
contact_wrenches将解出的边力还原为接触点处的世界力(N)和纯扭矩(N·m)，作用于body2；body1取反。
纯扭矩不包含相对质心的r×force。必须使用同次前向的表、几何和行力。
"""

from dataclasses import dataclass
import numpy as np
from .contacts import CudaContactJacobian, FrameRows
from .constraint_rows import CudaNormalSoftRows
from .friction_rows import CudaPyramidRows
from .joint_rows import CudaJointRows


@dataclass(frozen=True)
class ConstraintTable:
    jacobian: np.ndarray
    base_jacobian: np.ndarray
    velocity: np.ndarray
    prescribed_velocity: np.ndarray
    prescribed_acceleration: np.ndarray
    D: np.ndarray
    aref: np.ndarray
    effective_aref: np.ndarray
    frictionloss: np.ndarray
    kind: np.ndarray
    a: np.ndarray
    b: np.ndarray
    contact_row: np.ndarray
    contact_nrow: np.ndarray
    nfrictionloss: int
    njoint_rows: int
    velocity_damping: np.ndarray | None = None


class CudaConstraintTable:
    """CUDA计算行参数，CPU按原顺序拼接稠密表。不是完整GPU生产工作区。

    顺序：hinge frictionloss、活动hinge限位、输入接触顺序(每点1、4或6行)。
    contacts为活动ContactFrame；pair_ids为每点的显式模型pair编号。
    """

    def __init__(self, model, plan, motion):
        if motion.layout is not plan.layout:
            raise ValueError("motion和plan必须使用同一layout")
        if np.any(model.pair_solreffriction):
            raise ValueError("当前不支持独立的pair solreffriction")
        self.numeric = motion.numeric
        self.plan = plan
        self.layout = plan.layout
        self.joints = CudaJointRows(model, plan, numerics=self.numeric.mode)
        self.jac = CudaContactJacobian(motion)
        self.normal = CudaNormalSoftRows(plan, numerics=self.numeric.mode)
        self.pyramid = CudaPyramidRows(plan, numerics=self.numeric.mode)
        self.dim = np.array(model.pair_dim, copy=True)
        self.mu = np.array(model.pair_friction[:, 0], copy=True)
        self.torsion_mu = np.array(model.pair_friction[:, 2], copy=True)
        self.sr = np.array(model.pair_solref, copy=True)
        self.si = np.array(model.pair_solimp, copy=True)
        self.bodies = np.array(
            [
                [model.geom_bodyid[a], model.geom_bodyid[b]]
                for a, b in zip(model.pair_geom1, model.pair_geom2)
            ],
            dtype=int,
        ).reshape(-1, 2)
        self.invw = np.array(model.body_invweight0[:, 0], copy=True)
        if np.any(self.sr <= 0) or np.any(self.mu[self.dim >= 3] <= 0):
            raise ValueError("需要正solref和摩擦接触正滑动mu")
        if np.any(self.torsion_mu[self.dim == 4] <= 0):
            raise ValueError("condim4需要正扭转摩擦系数")
        for array in (self.dim, self.mu, self.torsion_mu, self.sr, self.si, self.bodies, self.invw):
            array.setflags(write=False)

    def assemble(self, state, palm, contacts, pair_ids):
        contacts = list(contacts)
        nc = len(contacts)
        self.plan.check_counts(nc, 0)
        pairs = np.asarray(pair_ids)
        if pairs.shape != (nc,) or (
            nc
            and (
                pairs.dtype.kind not in "iu" or np.any(pairs < 0) or np.any(pairs >= len(self.dim))
            )
        ):
            raise ValueError("每个接触必须提供合法显式pair编号")
        pairs = pairs.astype(int)
        for c, p in zip(contacts, pairs):
            if (c.body1, c.body2) != tuple(self.bodies[p]):
                raise ValueError("接触双方顺序与显式pair不一致")
        joint = self.joints.assemble(state)
        nj = len(joint.D)
        nv = self.layout.nv
        nr = np.where(self.dim[pairs] == 4, 6, np.where(self.dim[pairs] == 3, 4, 1))
        total = nj + int(nr.sum())
        self.plan.check_counts(nc, total)
        include_torsion = bool(np.any(self.dim[pairs] == 4))
        frames = self.jac.assemble_frames(state, palm, contacts, include_torsion=include_torsion)
        # 固定顺序的索引/复制在CPU执行；三方向J、软参数及关节活动判断均为CUDA结果。
        columns = [
            np.zeros((total, nv), self.numeric.dtype),
            np.zeros((total, 6), self.numeric.dtype),
        ]
        columns += [np.zeros(total, self.numeric.dtype) for _ in range(7)]
        J, Jp, vel, pv, pa, D, aref, eff, F = columns
        damping = np.zeros(total, self.numeric.dtype)
        damping[:nj] = joint.damping
        J[:nj] = joint.jacobian
        vel[:nj] = joint.velocity
        D[:nj] = joint.D
        aref[:nj] = joint.aref
        eff[:nj] = joint.aref
        F[:nj] = joint.frictionloss
        kind = np.zeros(total, np.uint8)
        a = np.zeros(total, np.uint16)
        b = np.zeros(total, np.int8)
        kind[:nj] = joint.kind
        a[:nj] = joint.a
        b[:nj] = joint.b
        starts = nj + np.r_[0, np.cumsum(nr[:-1])] if nc else np.empty(0, int)
        for dim in (1, 3, 4):
            selected = np.flatnonzero(self.dim[pairs] == dim)
            if not len(selected):
                continue
            direction_count = 4 if dim == 4 else 3
            subset = FrameRows(
                *[
                    getattr(frames, name)[selected, :direction_count]
                    for name in frames.__dataclass_fields__
                ]
            )
            ps = pairs[selected]
            dist = [contacts[c].distance for c in selected]
            iw = [self.invw[contacts[c].body1] + self.invw[contacts[c].body2] for c in selected]
            if dim in (3, 4):
                soft = self.pyramid.assemble(
                    subset,
                    self.sr[ps],
                    self.si[ps],
                    dist,
                    iw,
                    self.mu[ps],
                    torsion_friction=self.torsion_mu[ps] if dim == 4 else None,
                )
            else:
                soft = self.normal.assemble(
                    self.sr[ps],
                    self.si[ps],
                    dist,
                    iw,
                    subset.velocity[:, 0],
                    subset.prescribed_acceleration[:, 0],
                )
            for index, c in enumerate(selected):
                size = int(nr[c])
                target = slice(starts[c], starts[c] + size)
                src = slice(index * size, (index + 1) * size)
                if dim in (3, 4):
                    J[target] = soft.jacobian[src]
                    Jp[target] = soft.base_jacobian[src]
                    vel[target] = soft.velocity[src]
                    pv[target] = soft.prescribed_velocity[src]
                    pa[target] = soft.prescribed_acceleration[src]
                else:
                    J[target] = subset.jacobian[index, 0]
                    Jp[target] = subset.base_jacobian[index, 0]
                    vel[target] = subset.velocity[index, 0]
                    pv[target] = subset.prescribed_velocity[index, 0]
                    pa[target] = subset.prescribed_acceleration[index, 0]
                damping[target] = soft.damping[src]
                D[target] = soft.D[src]
                aref[target] = soft.aref[src]
                eff[target] = soft.effective_aref[src]
                kind[target] = 2 if dim in (3, 4) else 3
                a[target] = c
                b[target] = np.arange(size)
        starts = starts.astype(np.uint16)
        nr = nr.astype(np.uint8)
        for col in [*columns, kind, a, b, starts, nr, damping]:
            col.setflags(write=False)
        return ConstraintTable(
            *columns, kind, a, b, starts, nr, int(np.count_nonzero(joint.kind == 0)), nj, damping
        )

    def contact_wrenches(self, table, contacts, pair_ids, row_force):
        """把解出的金字塔边力还原为每接触的世界系力/纯力矩，作用于body2。

        返回 (force_world, torque_world)，均为 (ncontact,3)，单位N与N·m。
        torque_world 是接触点处的扭转力矩，不含相对质心的 r×force；body1取相反号。
        contacts/pair_ids 必须与本次前向表相同；不重新求解，也不修改输入。
        """
        contacts = list(contacts)
        pairs = np.asarray(pair_ids)
        forces = np.asarray(row_force, dtype=float)
        if pairs.shape != (len(contacts),) or (
            len(contacts)
            and (
                pairs.dtype.kind not in "iu" or np.any(pairs < 0) or np.any(pairs >= len(self.dim))
            )
        ):
            raise ValueError("接触pair编号不合法")
        if forces.shape != table.D.shape or not np.isfinite(forces).all():
            raise ValueError("行力形状或数值错误")
        if len(table.contact_row) != len(contacts):
            raise ValueError("接触数量与前向表不一致")
        world_force = np.zeros((len(contacts), 3))
        world_torque = np.zeros_like(world_force)
        for index, (contact, pair) in enumerate(zip(contacts, pairs)):
            if (contact.body1, contact.body2) != tuple(self.bodies[pair]):
                raise ValueError("接触双方与pair不一致")
            count = int(table.contact_nrow[index])
            start = int(table.contact_row[index])
            expected = 6 if self.dim[pair] == 4 else 4 if self.dim[pair] == 3 else 1
            if count != expected:
                raise ValueError("接触行数与pair维度不一致")
            weights = forces[start : start + count]
            local_force = np.zeros(3)
            local_force[0] = weights.sum()
            if count >= 4:
                local_force[1] = self.mu[pair] * (weights[0] - weights[1])
                local_force[2] = self.mu[pair] * (weights[2] - weights[3])
            if count == 6:
                world_torque[index] = (
                    contact.frame[0] * self.torsion_mu[pair] * (weights[4] - weights[5])
                )
            world_force[index] = local_force @ contact.frame
        world_force.setflags(write=False)
        world_torque.setflags(write=False)
        return world_force, world_torque
