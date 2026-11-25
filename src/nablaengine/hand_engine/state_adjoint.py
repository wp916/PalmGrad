"""分层单步状态/动作VJP：无约束、关节约束及基本形状几何接触。"""

from dataclasses import dataclass
import numpy as np
from .action_adjoint import ActionTape
from .velocity_adjoint import CudaVelocityAdjoint
from .rne_position import CudaRNEPositionAdjoint


@dataclass(frozen=True)
class StepStateVJP:
    position_tangent: np.ndarray
    qvel: np.ndarray
    action: np.ndarray
    convention: str = "unconstrained_step_local_right_rotation_tangent"

    def qpos_cotangent(self, state, layout):
        """将局部位置余切提升为nq维数组余切，四元数采用零径向的最小范数代表。

        只约定单位四元数流形上的导数，不声称有唯一的离开单位球面的延拓。
        """
        layout.validate_state(state)
        if self.position_tangent.shape != (layout.nv,):
            raise ValueError("位置余切与layout不一致")
        out = np.zeros(layout.nq)
        for b in layout.blocks:
            qi, vi = b.qpos_start, b.qvel_start
            if b.kind == "hinge":
                out[qi] = self.position_tangent[vi]
            else:
                out[qi : qi + 3] = self.position_tangent[vi : vi + 3]
                q = state.qpos[qi + 3 : qi + 7]
                g = self.position_tangent[vi + 3 : vi + 6]
                n2 = float(q @ q)
                out[qi + 3] = -2 * float(q[1:] @ g) / n2
                out[qi + 4 : qi + 7] = 2 * (q[0] * g + np.cross(q[1:], g)) / n2
        out.setflags(write=False)
        return out


class CudaUnconstrainedStateAdjoint:
    """固定规定手掌/增益，对无约束implicitfast步返回位置局部扰动、速度、动作VJP。

    明确拒绝任何接触/限位/干摩擦行。位置梯度采用canonical nv维局部扰动，
    不是nq维四元数数组余切；当前不是JAX/PyTorch注册接口。
    """

    def __init__(self, engine):
        self.engine = engine
        self.velocity = CudaVelocityAdjoint(engine)
        self.position = CudaRNEPositionAdjoint(engine.dynamics)

    def record(self, inputs, **kwargs):
        tape = self.velocity.record(inputs, **kwargs)
        if len(tape.forward.constraints.D):
            raise ValueError("无约束状态反向不支持接触/限位/干摩擦行")
        return tape

    def vjp(self, tape, qpos_next_bar, qvel_next_bar):
        if not isinstance(tape, ActionTape) or tape._owner is not self.velocity.action._owner:
            raise ValueError("tape不属于此状态反向实例")
        with self.engine._lock:
            if len(tape.forward.constraints.D):
                raise ValueError("无约束状态反向不支持约束行")
            pv = self.velocity.vjp(tape, qpos_next_bar, qvel_next_bar)
            r = tape.forward
            s = tape.inputs.state
            p = tape.inputs.palm
            h = self.engine.h
            direct = self.velocity.action.integration.evaluate(
                s, r.advancement_acceleration, h, qpos_next_bar, qvel_next_bar
            )
            y = self.engine.solver.linear_solve(tape.implicit_matrix, direct.qacc)
            # 无约束H=M，lambda=y。filter的dM项与RNE中的dM*a_star合并，
            # 因而只需ID(q,v,a_adv)的位置VJP，不可推广到有约束情形。
            inverse = self.position.evaluate(s, p, y, qacc=r.advancement_acceleration)
            grad = direct.position_tangent - inverse.position_tangent_bar
            if tape.inputs.command.mode == "position_pd":
                n = self.engine.layout.hand_dofs
                grad[:n] -= tape.inputs.command.kp * y[:n]
            grad = self.engine.numeric.finite(grad, "无约束单步位置梯度")
            grad.setflags(write=False)
            return StepStateVJP(grad, pv.qvel, pv.action)


class CudaJointStateAdjoint(CudaUnconstrainedStateAdjoint):
    """无几何接触、允许hinge限位/干摩擦的完整单步状态VJP。

    使用一般H!=M公式；拒绝接触行，不冻结碰撞几何冒充完整位置反向。
    """

    def __init__(self, engine):
        super().__init__(engine)
        from .constraint_adjoint import CudaConstraintAdjoint

        self.constraint = CudaConstraintAdjoint(engine.layout.nv, numerics=engine.numeric.mode)

    @staticmethod
    def _check(table):
        if np.any(table.kind >= 2):
            raise ValueError("关节约束状态反向尚不支持几何接触行")

    def record(self, inputs, **kwargs):
        tape = self.velocity.record(inputs, **kwargs)
        self._check(tape.forward.constraints)
        return tape

    def vjp(self, tape, qpos_next_bar, qvel_next_bar):
        if not isinstance(tape, ActionTape) or tape._owner is not self.velocity.action._owner:
            raise ValueError("tape不属于此状态反向实例")
        with self.engine._lock:
            table = tape.forward.constraints
            self._check(table)
            pv = self.velocity.vjp(tape, qpos_next_bar, qvel_next_bar)
            r = tape.forward
            s = tape.inputs.state
            p = tape.inputs.palm
            h = self.engine.h
            direct = self.velocity.action.integration.evaluate(
                s, r.advancement_acceleration, h, qpos_next_bar, qvel_next_bar
            )
            y = self.engine.solver.linear_solve(tape.implicit_matrix, direct.qacc)
            astar_bar = self.velocity.action._kernel(
                1, np.r_[direct.qacc, r.velocity_coefficient, y]
            )
            lam = self.engine.solver.linear_solve(r.newton.evaluation.hessian, astar_bar)
            mass = self.position.mass_product(
                s, p, r.constrained_acceleration - r.advancement_acceleration, y
            )
            inverse = self.position.evaluate(s, p, lam, qacc=r.constrained_acceleration)
            grad = (
                direct.position_tangent + mass.position_tangent_bar - inverse.position_tangent_bar
            )
            if tape.inputs.command.mode == "position_pd":
                n = self.engine.layout.hand_dofs
                grad[:n] -= tape.inputs.command.kp * lam[:n]
            rows = self.constraint.row_vjp(
                table, r.newton.evaluation, r.constrained_acceleration, lam
            )
            selected = np.flatnonzero(table.kind == 1)
            if len(selected):
                ids = table.a[selected].astype(int)
                sign = table.b[selected].astype(int)
                mat = self.engine.tables.joints.material[ids]
                q = self.engine.numeric.finite(s.qpos[: self.engine.layout.hand_dofs], "关节位置")
                pos = np.where(sign == 1, q[ids] - mat[:, 0], mat[:, 1] - q[ids])
                soft = self.constraint.distance_derivative(
                    mat[:, 5:7], mat[:, 7:12], pos, mat[:, 4], h
                )
                # J在固定限位侧分支为常量；干摩擦pos恒为0，没有位置参数贡献。
                row_q = (
                    rows.D[selected] * soft.D + rows.effective_aref[selected] * soft.aref
                ) * sign
                np.add.at(grad, ids, -row_q)
            grad = self.engine.numeric.finite(grad, "关节约束单步位置梯度")
            grad.setflags(write=False)
            return StepStateVJP(
                grad, pv.qvel, pv.action, "joint_constraints_step_local_right_rotation_tangent"
            )


class CudaContactStateAdjoint(CudaJointStateAdjoint):
    """基本形状碰撞的完整单步状态/动作VJP；mixed64解析正确性入口。

    使用原几何分支的精确导数；规定手掌/材料/PD增益固定。位置采用局部扰动。
    基方向收集几何JVP；胶囊—盒子使用设备侧逐运算AD并保留退化守卫。
    默认不是最终融合批量后端；JAX通过jax_adapter调用本接口。
    geometry显式传入时可使用实验碰撞桥及其数值JVP；此时不声明几何导数为解析。
    condim4含绕法向角运动及其几何导数，不对材料/规定手掌/增益求导。
    """

    def __init__(self, engine, *, geometry=None):
        from .collision import CudaPrimitiveContacts
        from .collision_adjoint import CudaCollisionGeometryJVP
        from .contact_position import CudaContactPositionJVP

        if engine.numeric.mode != "mixed64":
            raise ValueError("完整接触状态反向当前只验证mixed64")
        if geometry is None and not isinstance(engine.contact_provider, CudaPrimitiveContacts):
            raise TypeError("需要原基本形状GPU碰撞provider")
        if engine.contact_provider.layout is not engine.layout:
            raise ValueError("碰撞与动力学layout必须相同")
        super().__init__(engine)
        if (
            geometry is not None
            and getattr(geometry, "provider", None) is not engine.contact_provider
        ):
            raise ValueError("外部几何导数必须绑定同一个碰撞provider")
        self.geometry = (
            CudaCollisionGeometryJVP(engine.contact_provider) if geometry is None else geometry
        )
        self.contact_convention = (
            "primitive_contact_step_local_right_rotation_tangent"
            if geometry is None
            else "external_geometry_contact_step_local_right_rotation_tangent"
        )
        self.projection = CudaContactPositionJVP(engine.dynamics.motion)

    @staticmethod
    def _check(table):
        if not np.isin(table.kind, [0, 1, 2, 3]).all():
            raise ValueError("不支持的约束行类型")

    def vjp(self, tape, qpos_next_bar, qvel_next_bar):
        # 父类已经包含一般H!=M的动力学、积分、PD及关节限位贡献。
        with self.engine._lock:
            base = super().vjp(tape, qpos_next_bar, qvel_next_bar)
            r = tape.forward
            table = r.constraints
            s = tape.inputs.state
            p = tape.inputs.palm
            nv = self.engine.layout.nv
            baseline = self.geometry.evaluate(s, p, np.zeros(nv))
            # 重建行并严格核对当前provider/模型与此tape，避免拼接无关几何。
            fresh = self.engine.tables.assemble(s, p, baseline.contacts, baseline.pair_ids)
            for key in (
                "jacobian",
                "base_jacobian",
                "D",
                "effective_aref",
                "kind",
                "a",
                "b",
                "contact_row",
                "contact_nrow",
            ):
                if not np.array_equal(getattr(fresh, key), getattr(table, key)):
                    raise ValueError("碰撞重放与前向tape约束行不一致")
            if not len(baseline.contacts):
                return StepStateVJP(
                    base.position_tangent,
                    base.qvel,
                    base.action,
                    self.contact_convention,
                )
            direct = self.velocity.action.integration.evaluate(
                s, r.advancement_acceleration, self.engine.h, qpos_next_bar, qvel_next_bar
            )
            y = self.engine.solver.linear_solve(tape.implicit_matrix, direct.qacc)
            astar_bar = self.velocity.action._kernel(
                1, np.r_[direct.qacc, r.velocity_coefficient, y]
            )
            lam = self.engine.solver.linear_solve(r.newton.evaluation.hessian, astar_bar)
            rows = self.constraint.row_vjp(
                table, r.newton.evaluation, r.constrained_acceleration, lam
            )
            # 静态材料及invweight逐接触装配一次；摩擦行保留原金字塔缩放。
            material = self.engine.tables
            ids = baseline.pair_ids
            iw = np.array(
                [material.invw[c.body1] + material.invw[c.body2] for c in baseline.contacts]
            )
            mu = material.mu[ids]
            dim = material.dim[ids]
            friction = dim >= 3
            iw[friction] *= (
                (1 + mu[friction] ** 2)
                * 2
                * mu[friction] ** 2
                / self.engine.plan.physics["impratio"]
            )
            soft = self.constraint.distance_derivative(
                material.sr[ids],
                material.si[ids],
                np.array([c.distance for c in baseline.contacts]),
                iw,
                self.engine.h,
            )
            grad = base.position_tangent.copy()
            for i in range(nv):
                direction = np.zeros(nv)
                direction[i] = 1
                g = self.geometry.evaluate(s, p, direction)
                if not all(
                    np.array_equal(getattr(g, k), getattr(baseline, k))
                    for k in ("pair_ids", "slots", "branch")
                ):
                    raise RuntimeError("同一输入的碰撞分支重放不一致")
                j = self.projection.evaluate(
                    s,
                    p,
                    g.contacts,
                    direction,
                    g.position,
                    g.frame,
                    include_torsion=bool(np.any(dim == 4)),
                )
                total = 0.0
                for c in range(len(g.contacts)):
                    start = int(table.contact_row[c])
                    size = int(table.contact_nrow[c])
                    for edge in range(size):
                        row = start + edge
                        dJ = j.jacobian[c, 0].copy()
                        dv = j.velocity[c, 0]
                        dpa = j.prescribed_acceleration[c, 0]
                        if size in (4, 6):
                            t = 1 + edge // 2
                            coefficient = material.torsion_mu[ids[c]] if t == 3 else mu[c]
                            weight = (-1 if edge % 2 else 1) * coefficient
                            dJ += weight * j.jacobian[c, t]
                            dv += weight * j.velocity[c, t]
                            dpa += weight * j.prescribed_acceleration[c, t]
                        dD = soft.D[c] * g.distance[c]
                        daref = (
                            soft.aref[c] * g.distance[c] - table.velocity_damping[row] * dv - dpa
                        )
                        total += float(
                            rows.jacobian[row] @ dJ
                            + rows.D[row] * dD
                            + rows.effective_aref[row] * daref
                        )
                grad[i] -= total
            grad = self.engine.numeric.finite(grad, "完整接触单步位置梯度")
            grad.setflags(write=False)
            return StepStateVJP(grad, base.qvel, base.action, self.contact_convention)
