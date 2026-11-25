"""正式 hand 后端的物理模式检查、容量预算与工作区声明。

make_backend_plan 输入模型、HandLayout及可选容量，输出只读参数快照与拓扑计划。
支持显式pair的condim=1/3/4；condim4预留每接触6行、4个接触方向，其余模型保留原预算。
不执行碰撞或动力学求解。完整批量生产后端的ready标记，与显式单步验证入口分别管理。
Newton、pyramidal和implicitfast规则保持原约定，不能静默退回教学Euler。
"""

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping
import numpy as np
import mujoco
from .layout import build_topology


@dataclass(frozen=True)
class HandCapacity:
    max_contacts: int = 128
    max_rows: int | None = None

    def __post_init__(self):
        for name in ("max_contacts", "max_rows"):
            value = getattr(self, name)
            if value is None and name == "max_rows":
                continue
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not 1 <= value < 65535
            ):
                raise ValueError(f"{name}必须是1..65534内的整数")


@dataclass(frozen=True)
class HandBackendPlan:
    layout: object
    topology: object
    capacity: HandCapacity
    max_rows: int
    contact_support: int
    physics: Mapping

    def check_counts(self, contacts, rows):
        for value in (contacts, rows):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError("实际接触/约束行计数必须是非负整数")
        if contacts > self.capacity.max_contacts or rows > self.max_rows:
            raise OverflowError("hand接触/约束工作区溢出；必须扩容或拒绝该步，禁止静默丢弃约束")

    def require_ready(self):
        raise NotImplementedError(
            "完整批量hand后端尚未接通；可显式使用外部几何implicit_step验证入口，禁止自动退回教学Euler步进器"
        )

    def storage_header(self):
        """hand独立声明，无G1 root6/body1/3子节点假设。尚不是原求解器的结构体替换。"""
        nv = self.layout.nv
        nb = self.layout.body_count
        ntri = nv * (nv + 1) // 2
        directions = 4 if np.any(self.physics["pair_dim"] == 4) else 3
        return f"""#include <stdint.h>
#define H_NBODY {nb}
#define H_NQ {self.layout.nq}
#define H_NV {nv}
#define H_NTRI {ntri}
#define H_MAXCON {self.capacity.max_contacts}
#define H_MAXROW {self.max_rows}
#define H_KDOF {self.contact_support}
enum HandRowKind : uint8_t {{ H_FRICTIONLOSS=0, H_LIMIT=1, H_PYRAMID=2, H_NORMAL=3 }};
// 大接触表驻留global，每环境独立；row_a允许超过255，kind保留原0/1/2/3语义。
struct HandContactStorage {{
  float J[H_MAXCON][{directions}][H_KDOF];
  float Jp[H_MAXCON][{directions}][6];
  uint16_t dof[H_MAXCON][H_KDOF], ndof[H_MAXCON];
  uint64_t mask[H_MAXCON];
  uint16_t ncon, nrow;
  uint8_t row_kind[H_MAXROW];
  uint16_t row_a[H_MAXROW];
  int8_t row_b[H_MAXROW];
  float D[H_MAXROW], aref[H_MAXROW];
}};
// 这是声明阶段的Newton scratch预算，实际移植需逐字段映射/核对生命周期。
struct HandNewtonScratch {{
  float M[H_NTRI], L[H_NTRI];
  float vectors[8][H_NV];
  float rows[3][H_MAXROW];
  float cbuf[72];
}};
"""

    def summary(self):
        nv = self.layout.nv
        ntri = nv * (nv + 1) // 2
        # 全float声明，无内部对齐空隙；只覆盖该scratch，不代表整套后端或最终占用率。
        scratch = 4 * (2 * ntri + 8 * nv + 3 * self.max_rows + 72)
        return dict(
            application="hand_only",
            nq=self.layout.nq,
            nv=nv,
            bodies=self.layout.body_count,
            free_objects=self.layout.object_count,
            prescribed_palms=1,
            max_contacts=self.capacity.max_contacts,
            max_rows=self.max_rows,
            contact_support=self.contact_support,
            body_mask_bits=64,
            dof_mask_bits=64,
            row_count_bits=16,
            contact_index_bits=16,
            child_storage="CSR",
            declared_newton_scratch_bytes=scratch,
            contact_storage_location="global",
            integrator="implicitfast",
            solver="Newton",
            cone="pyramidal",
            # 这里描述完整批量后端；单步碰撞/完整状态VJP通过显式正确性入口调用。
            runtime_contact_kernel_ready=False,
            gradient_ready=False,
        )


def make_backend_plan(model, layout, capacity=None):
    """只做正式入口的初步物理模式检查，不声称完成全部MuJoCo字段能力审计。

    不修改model，不将Euler模型偷偷改成implicitfast，也不改变原材料参数。
    """
    required = [
        ("integrator", mujoco.mjtIntegrator.mjINT_IMPLICITFAST),
        ("solver", mujoco.mjtSolver.mjSOL_NEWTON),
        ("cone", mujoco.mjtCone.mjCONE_PYRAMIDAL),
    ]
    for name, value in required:
        if getattr(model.opt, name) != value:
            raise ValueError(f"正式hand后端要求{name}={value.name}；教学模型请使用独立教学入口")
    if not np.isfinite(model.opt.timestep) or model.opt.timestep <= 0:
        raise ValueError("timestep必须是有限正数")
    if any(int(d) not in (1, 3, 4) for d in model.pair_dim):
        raise ValueError("正式基准仅保留condim=1/3/4支持范围")
    if np.any(model.pair_margin) or np.any(model.pair_gap):
        raise ValueError("正式基准保留原显式pair margin=gap=0约定")
    if np.any(model.pair_friction[:, 0] != model.pair_friction[:, 1]):
        raise ValueError("正式基准要求两个滑动摩擦系数相同")
    topology = build_topology(model, layout)
    capacity = capacity or HandCapacity()
    # condim4每接触预留6条锥行；仅旧模型时保持原4条预算，禁止悄悄截断。
    rows_per_contact = 6 if np.any(model.pair_dim == 4) else 4
    required_rows = rows_per_contact * capacity.max_contacts + 2 * layout.hand_dofs
    max_rows = capacity.max_rows or ((required_rows + 31) // 32) * 32
    if max_rows < required_rows or max_rows >= 65535:
        raise ValueError(f"max_rows至少为{required_rows}且必须小于65535，不能静默截断")
    support = max(
        (
            topology.contact_mask(int(model.geom_bodyid[a]), int(model.geom_bodyid[b])).bit_count()
            for a, b in zip(model.pair_geom1, model.pair_geom2)
        ),
        default=0,
    )
    support = max(4, ((support + 3) // 4) * 4)
    # 保留源参数独立快照，后续loader直接使用，禁止以教学默认值替换。
    physics = {
        name: np.array(getattr(model, name), copy=True)
        for name in (
            "pair_dim",
            "pair_solref",
            "pair_solimp",
            "pair_friction",
            "jnt_range",
            "jnt_solref",
            "jnt_solimp",
            "dof_armature",
        )
    }
    if not np.isfinite(model.opt.impratio) or model.opt.impratio <= 0:
        raise ValueError("impratio必须是有限正数")
    if not np.isfinite(model.stat.meaninertia) or model.stat.meaninertia <= 0:
        raise ValueError("meaninertia必须是有限正数")
    physics.update(
        meaninertia=float(model.stat.meaninertia),
        impratio=float(model.opt.impratio),
        gravity=np.array(model.opt.gravity, copy=True),
        timestep=float(model.opt.timestep),
        iterations=int(model.opt.iterations),
        tolerance=float(model.opt.tolerance),
    )
    for value in physics.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return HandBackendPlan(layout, topology, capacity, max_rows, support, MappingProxyType(physics))
