"""根据 MuJoCo 模型生成的 G1 专用 CUDA 动力学源码。

本文件提供无接触部分的基础运算：正向运动学、速度与加速度递推、递归牛顿－欧拉
逆动力学（RNE），以及复合刚体算法（CRB）构造的关节空间质量矩阵。
设备函数操作共享内存中的状态；每个环境由一个 warp 处理，lane b 对应刚体 b，
按树深度逐层递推。约定与 MuJoCo 一致：自由根的线速度在世界系、角速度在局部系；
hinge 位于刚体原点；递推空间量在世界系表达。质量矩阵以根刚体原点为参考，
包含 armature 附加惯性项，对应 mj_crb 的 qM。

constants(m) 按 CONST_LAYOUT 顺序打包模型数组，并检查专用内核的结构前提：
刚体 1 为自由根，之后每体一个原点 hinge，最多 32 个刚体，nv >= 32；
位置执行器按关节顺序排列（执行器 a 驱动自由度 6+a），且存在 torso_link。
学习时先读 constants/header，再读运动学、力与惯量，最后读 Cholesky 求解。
这些限制属于当前实现，不能通过删除断言获得通用模型支持。
"""

import mujoco
import numpy as np

CONST_LAYOUT = (  # （设备符号名，类型 i/f，数量单位 nb/nv/one，每项宽度）；顺序必须与 parts 和加载器一致
    ("c_parent", "i", "nb", 1), ("c_depth", "i", "nb", 1), ("c_qadr", "i", "nb", 1), ("c_dadr", "i", "nb", 1),
    ("c_nchild", "i", "nb", 1), ("c_child", "i", "nb", 3), ("c_pos", "f", "nb", 3), ("c_quat", "f", "nb", 4),
    ("c_axis", "f", "nb", 3), ("c_ipos", "f", "nb", 3), ("c_iquat", "f", "nb", 4), ("c_mass", "f", "nb", 1),
    ("c_inert", "f", "nb", 3), ("c_grav", "f", "one", 3), ("c_armature", "f", "nv", 1), ("c_nanc", "i", "nb", 1),
    ("c_anc", "i", "nb", 12), ("c_kp", "f", "nv", 1), ("c_kv", "f", "nv", 1), ("c_frclim", "f", "nv", 1),
    ("c_misc", "f", "one", 2),  # 物理时间步、躯干刚体编号
)


def constants(m):
    """返回按 CONST_LAYOUT 排列的 float32 模型常量和树的最大深度；不推进仿真。"""
    # 主机端准备：nb 包含编号 0 的世界刚体；编译后的父节点编号先于子节点。
    nb = m.nbody
    depth, children = [0] * nb, [[] for _ in range(nb)]
    for b in range(1, nb):
        depth[b] = depth[m.body_parentid[b]] + 1
        children[m.body_parentid[b]].append(b)
    # 这些断言对应设备端固定容量和专用分支，不是可以直接移除的形式检查。
    assert nb <= 32 and max(len(c) for c in children[1:]) <= 3
    assert m.jnt_type[m.body_jntadr[1]] == mujoco.mjtJoint.mjJNT_FREE and m.body_jntnum[1] == 1
    # 按刚体保存位置/速度地址和局部转轴；自由根的地址由 fwd_body 专门处理。
    qadr, dadr, axis = np.zeros(nb), np.zeros(nb), np.zeros((nb, 3))
    for b in range(2, nb):
        j = m.body_jntadr[b]
        assert m.body_jntnum[b] == 1 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and not m.jnt_pos[j].any()
        qadr[b], dadr[b], axis[b] = m.jnt_qposadr[j], m.jnt_dofadr[j], m.jnt_axis[j]
    assert m.nv >= 32, (
        f"nv = {m.nv}: the engine needs nv >= 32 (at least 26 hinges). The Cholesky and triangular solves "
        "(chol_solve_compact / chol_solve_reg here, adjoint.py tri_solve_packed) give lane l the dof row l, so with "
        "fewer dofs they read and write past their arrays (an illegal address, NaN or silently wrong results)")
    # 每体最多 3 个子节点；完整五指模型需要先核对手掌的实际分支数。
    child = np.zeros((nb, 3))
    for b in range(1, nb):
        child[b, :len(children[b])] = children[b]
    nchild = np.array([len(children[b]) if b else 0 for b in range(nb)])
    anc, nanc = np.zeros((nb, 12)), np.zeros(nb)  # 各刚体的 hinge 祖先表（不含自由根），从近到远排列
    for b in range(2, nb):
        k, lst = m.body_parentid[b], []
        while k >= 2:
            lst.append(k)
            k = m.body_parentid[k]
        assert len(lst) <= 12
        anc[b, :len(lst)], nanc[b] = lst, len(lst)
    kp, kv, frclim = np.zeros(m.nv), np.zeros(m.nv), np.full(m.nv, 1e30)
    for a in range(m.nu):  # hinge 位置执行器，传动比为 1：力矩 = kp*(ctrl-q)-kv*qdot
        j = m.actuator_trnid[a, 0]
        assert m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT, f"actuator {a}: joint transmissions only"
        assert m.jnt_dofadr[j] == 6 + a, (
            f"actuator {a} drives dof {m.jnt_dofadr[j]}: the kernels read the setpoint of hinge dof i from "
            "ctrl[i - 6], so actuator a must drive dof 6 + a (actuators in joint order)")
        assert m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE and m.actuator_gear[a, 0] == 1.0
        assert m.actuator_gainprm[a, 0] == -m.actuator_biasprm[a, 1] and not m.actuator_forcelimited[a]
        dof = m.jnt_dofadr[j]
        kp[dof], kv[dof] = m.actuator_gainprm[a, 0], -m.actuator_biasprm[a, 2]
        if m.jnt_actfrclimited[j]:
            frclim[dof] = m.jnt_actfrcrange[j, 1]
            assert m.jnt_actfrcrange[j, 0] == -m.jnt_actfrcrange[j, 1]
    # G1 专用域随机化入口：未来手模型应将此项改为可选或显式指定，不能仅改名字。
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    assert torso >= 1, 'no body named "torso_link": the per-world COM offset (com) is applied to that body'
    # 所有段必须与 CONST_LAYOUT 一一对应。整数先用 float32 传输，再由 host.loader 转成 int。
    parts = [m.body_parentid, depth, qadr, dadr, nchild, child, m.body_pos, m.body_quat, axis, m.body_ipos,
             m.body_iquat, m.body_mass, m.body_inertia, m.opt.gravity, m.dof_armature, nanc, anc, kp, kv, frclim,
             [m.opt.timestep, torso]]
    return np.concatenate([np.asarray(x, np.float64).ravel() for x in parts]).astype(np.float32), max(depth)


def header(m, maxd, wpb):
    """生成模型维度宏和设备常量声明；维度在 CUDA 编译时固定。"""
    return f"""
#include <cuda_runtime.h>
#define NB {m.nbody}
#define NQ {m.nq}
#define NV {m.nv}
#define MAXD {maxd}
#define WPB {wpb}
#define NTRI (NV * (NV + 1) / 2)
#define TRI(i, j) ((i) * ((i) + 1) / 2 + (j))  // 下三角压缩索引，要求 j<=i；第 i 行从 i*(i+1)/2 开始
""" + r"""
#ifndef DKC
#define DKC __device__  // 模型数组位于全局内存（使用 L1 缓存）；各 lane 按接触对、刚体或
                       // 自由度访问不同地址，若放常量内存会串行；可用 -DDKC=__constant__ 对比
#endif
DKC int c_parent[NB], c_depth[NB], c_qadr[NB], c_dadr[NB], c_nchild[NB], c_child[NB * 3], c_nanc[NB],
    c_anc[NB * 12];
DKC float c_pos[NB * 3], c_quat[NB * 4], c_axis[NB * 3], c_ipos[NB * 3], c_iquat[NB * 4], c_mass[NB],
    c_inert[NB * 3], c_grav[3], c_armature[NV], c_kp[NV], c_kv[NV], c_frclim[NV], c_misc[2];
"""


DEVICE = r"""
// 四元数采用 (w,x,y,z) 顺序；乘积顺序决定旋转复合顺序，不可交换。
__device__ __forceinline__ void qmul(const float* a, const float* b, float* o) {
  float w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  float x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  float y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  float z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
  o[0] = w; o[1] = x; o[2] = y; o[3] = z;
}
__device__ __forceinline__ void qrot(const float* q, const float* v, float* o) {  // 四元数旋转向量：o = q*v*q^{-1}
  float tx = 2.f * (q[2] * v[2] - q[3] * v[1]), ty = 2.f * (q[3] * v[0] - q[1] * v[2]), tz = 2.f * (q[1] * v[1] - q[2] * v[0]);
  o[0] = v[0] + q[0] * tx + q[2] * tz - q[3] * ty;
  o[1] = v[1] + q[0] * ty + q[3] * tx - q[1] * tz;
  o[2] = v[2] + q[0] * tz + q[1] * ty - q[2] * tx;
}
__device__ __forceinline__ void qrot_inv(const float* q, const float* v, float* o) {
  float qc[4] = {q[0], -q[1], -q[2], -q[3]};
  qrot(qc, v, o);
}
__device__ __forceinline__ void cross(const float* a, const float* b, float* o) {
  o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0];
}
__device__ __forceinline__ float dot3(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// 显式控制浮点舍入：伴随在另一个 kernel（JAX 中甚至另一编译单元）重放前向的
// 刚体位姿、碰撞几何位姿及接触几何 dist/pos/frame，需要逐位一致（adj_poses/adj_contacts）。
// 普通 C 表达式的乘加可能被 NVVM/ptxas 融合成 FMA；融合受寄存器压力、上下文、
// -fmad 和 GPU 架构影响，同一公式在两个 kernel 中可能相差一个末位单位（ulp）。
// 下列内建指令固定舍入位置，__fmaf_rn 也不会被拆开，从而保持重放的一致性。
// 在同一 GPU 上，rsqrtf/sincosf 仍采用该架构实现；默认 -prec-div/-prec-sqrt 的
// IEEE 除法与 sqrtf 已正确舍入，无需额外包装。
// 重放涉及本文件 *_rn/fwd_body、step.py 的 geom_pose、collide_pair 与伴随共用的
// make_frame/sphere_sphere/plane_point/plane_capsule_frame/capsule_end/segment_point/capcap_*，
// 以及包围测试、limit_dists、kbimp/imp_raw/contact_invw/efc_D 中相关运算。
// 此运算顺序沿用 sm_70 上 ptxas 对原始表达式的选择，保留 V100 原来的逐位结果：
// a*b+c*d 用 fma(a,b,c*d)，a*b-c*d 用 fma(a,b,-(c*d))；累加从左到右。
// 被复用的乘积先舍入，例如 W=Wp+round(qd*aw)，以及胶囊两端的 p±round(l*a)。
// 不需要重放的动力学、求解器及伴随 VJP 仍使用上面的普通 C 辅助函数。
// 原代码验证方式：前向或伴随编译单元采用 -fmad=false，重放仍与前向逐位一致。
#define RMUL(a, b) __fmul_rn((a), (b))
#define RADD(a, b) __fadd_rn((a), (b))
#define RSUB(a, b) __fsub_rn((a), (b))
#define RFMA(a, b, c) __fmaf_rn((a), (b), (c))   // a*b+c，仅舍入一次
#define RFMS(a, b, c) __fmaf_rn(-(a), (b), (c))  // c-a*b，仅舍入一次

__device__ __forceinline__ void qmul_rn(const float* a, const float* b, float* o) {
  float w = RFMS(a[3], b[3], RFMS(a[2], b[2], RFMA(a[0], b[0], -RMUL(a[1], b[1]))));
  float x = RFMS(a[3], b[2], RFMA(a[2], b[3], RFMA(a[0], b[1], RMUL(a[1], b[0]))));
  float y = RFMA(a[3], b[1], RFMA(a[2], b[0], RFMA(a[0], b[2], -RMUL(a[1], b[3]))));
  float z = RFMA(a[3], b[0], RFMS(a[2], b[1], RFMA(a[0], b[3], RMUL(a[1], b[2]))));
  o[0] = w; o[1] = x; o[2] = y; o[3] = z;
}
// 四元数旋转的向量展开：t=2*(qv×v)，o=v+q0*t+qv×t
__device__ __forceinline__ void qrot_rn(const float* q, const float* v, float* o) {
  float tx = RMUL(2.f, RFMA(q[2], v[2], -RMUL(q[3], v[1])));
  float ty = RMUL(2.f, RFMA(q[3], v[0], -RMUL(q[1], v[2])));
  float tz = RMUL(2.f, RFMA(q[1], v[1], -RMUL(q[2], v[0])));
  o[0] = RFMS(q[3], ty, RFMA(q[2], tz, RFMA(q[0], tx, v[0])));
  o[1] = RFMS(q[1], tz, RFMA(q[3], tx, RFMA(q[0], ty, v[1])));
  o[2] = RFMS(q[2], tx, RFMA(q[1], ty, RFMA(q[0], tz, v[2])));
}
__device__ __forceinline__ void cross_rn(const float* a, const float* b, float* o) {
  o[0] = RFMA(a[1], b[2], -RMUL(a[2], b[1]));
  o[1] = RFMA(a[2], b[0], -RMUL(a[0], b[2]));
  o[2] = RFMA(a[0], b[1], -RMUL(a[1], b[0]));
}
__device__ __forceinline__ float dot3_rn(const float* a, const float* b) {
  return RFMA(a[2], b[2], RFMA(a[0], b[0], RMUL(a[1], b[1])));
}
__device__ __forceinline__ void zaxis_rn(const float* q, float* a) {  // 旋转局部 z 轴，得到平面法向或胶囊轴向
  const float ez[3] = {0.f, 0.f, 1.f};
  qrot_rn(q, ez, a);
}
// warp 含 32 个 lane；用 XOR shuffle 在寄存器间归约，最终各 lane 都得到总和。
// 全掩码要求相关 lane 一致参与，不能随意让部分线程提前退出。
__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// hinge 刚体（b>=2）的前向计算拆成三步：
// 1. 用父位姿和固定局部变换求原点 X、初始旋转 q0（qb=c_quat+4*b，pb=c_pos+3*b）；
// 2. 用关节角和轴 ax=c_axis+3*b 求 hinge 旋转 ha，此步不依赖父节点；
// 3. 合成 Q=q0*ha，并递推速度与加速度。
// 拆分允许 kinematics 模块预先计算各体 ha、加载常量，同时保持与 fwd_body 逐位一致。
__device__ __forceinline__ void fwd_hinge_pos(const float* Qp, const float* Xp, const float* qb, const float* pb,
                                              float* q0, float* X) {
  float t[3];
  qmul_rn(Qp, qb, q0);
  qrot_rn(Qp, pb, t);
  for (int k = 0; k < 3; k++) X[k] = RADD(Xp[k], t[k]);
}
__device__ __forceinline__ void fwd_hinge_rot(float ang, const float* ax, float* ha) {
  float s, c;
  sincosf(RMUL(0.5f, ang), &s, &c);  // 共用一次参数规约，结果与 sinf/cosf 逐位一致；原作者 V100 前向快 4%
  ha[0] = c; ha[1] = RMUL(s, ax[0]); ha[2] = RMUL(s, ax[1]); ha[3] = RMUL(s, ax[2]);
}
__device__ __forceinline__ void fwd_hinge_rest(const float* q0, const float* ha, const float* ax, float qd, float qdd,
                                               const float* Xp, const float* Wp, const float* Vp, const float* ALp,
                                               const float* Ap, float* Q, const float* X, float* W, float* V,
                                               float* AL, float* A) {
  // W/V 是世界系角速度/原点线速度；AL/A 是角加速度/原点递推加速度。
  qmul_rn(q0, ha, Q);
  float aw[3], awd[3], r[3], u[3], u2[3], wr[3];
  qrot_rn(Q, ax, aw);
  for (int k = 0; k < 3; k++) { r[k] = RSUB(X[k], Xp[k]); awd[k] = RMUL(aw[k], qd); W[k] = RADD(Wp[k], awd[k]); }
  cross_rn(Wp, r, u);
  for (int k = 0; k < 3; k++) V[k] = RADD(Vp[k], u[k]);
  // AL = ALp + aw*qdd + Wp×(aw*qd)，包含转轴随父体旋转的耦合项。
  cross_rn(Wp, awd, u);
  for (int k = 0; k < 3; k++) AL[k] = RADD(RFMA(aw[k], qdd, ALp[k]), u[k]);
  // A = Ap + ALp×r + Wp×(Wp×r)：切向与向心项；hinge 位于子体原点。
  cross_rn(ALp, r, u);
  cross_rn(Wp, r, wr);
  cross_rn(Wp, wr, u2);
  for (int k = 0; k < 3; k++) A[k] = RADD(RADD(Ap[k], u[k]), u2[k]);
}

// 通用的单 hinge 递推入口：不依赖 b==1 的自由根假设。
// 父节点可以是固定世界；当前只支持关节位于刚体原点、零参考角的 hinge。
// 此处仅拆出已有运算，不代表完整求解器已支持固定根或小自由度。
__device__ __forceinline__ void fwd_hinge_body(int b, const float* Qp, const float* Xp, const float* Wp,
                                               const float* Vp, const float* ALp, const float* Ap,
                                               const float* qpos, const float* qvel, const float* qacc,
                                               float* Q, float* X, float* W, float* V, float* AL, float* A) {
  float q0[4], ha[4];
  fwd_hinge_pos(Qp, Xp, c_quat + 4 * b, c_pos + 3 * b, q0, X);
  float ang = qpos[c_qadr[b]], qd = qvel[c_dadr[b]], qdd = qacc[c_dadr[b]];
  fwd_hinge_rot(ang, c_axis + 3 * b, ha);
  fwd_hinge_rest(q0, ha, c_axis + 3 * b, qd, qdd, Xp, Wp, Vp, ALp, Ap, Q, X, W, V, AL, A);
}

// 从父节点状态递推刚体 b 的位姿、速度和加速度。
// 显式舍入使前向 substep_pre（qacc=0）与伴随 adj_poses（qacc=a*）的重放位姿逐位一致。
__device__ __forceinline__ void fwd_body(int b, const float* Qp, const float* Xp, const float* Wp, const float* Vp,
                                         const float* ALp, const float* Ap, const float* qpos, const float* qvel,
                                         const float* qacc, float* Q, float* X, float* W, float* V, float* AL,
                                         float* A) {
  if (b == 1) {  // 自由根：世界位置、四元数；世界系线速度、局部系角速度
    // 先归一化根四元数；线加速度减重力，把重力统一纳入逆动力学递推。
    float n = rsqrtf(RFMA(qpos[6], qpos[6], RFMA(qpos[5], qpos[5], RFMA(qpos[3], qpos[3], RMUL(qpos[4], qpos[4])))));
    for (int k = 0; k < 4; k++) Q[k] = RMUL(qpos[3 + k], n);
    for (int k = 0; k < 3; k++) { X[k] = qpos[k]; V[k] = qvel[k]; A[k] = RSUB(qacc[k], c_grav[k]); }
    qrot_rn(Q, qvel + 3, W);
    qrot_rn(Q, qacc + 3, AL);
    return;
  }
  fwd_hinge_body(b, Qp, Xp, Wp, Vp, ALp, Ap, qpos, qvel, qacc, Q, X, W, V, AL, A);
}

// 刚体 b 自身所需的力 F 和关于刚体原点的力矩 N；在质心应用牛顿－欧拉方程
__device__ __forceinline__ void body_force(int b, const float* Q, const float* W, const float* AL, const float* A,
                                           const float* ipos, float* F, float* N) {
  // cvec 为原点到质心的世界系向量；质心加速度为 A+AL×c+W×(W×c)。
  float cvec[3], u[3], u2[3], wc[3];
  qrot(Q, ipos, cvec);
  cross(AL, cvec, u);
  cross(W, cvec, wc);
  cross(W, wc, u2);
  float m = c_mass[b];
  for (int k = 0; k < 3; k++) F[k] = m * (A[k] + u[k] + u2[k]);
  // 在惯性主轴系逐分量乘惯量，再旋回世界系，避免显式构造 3×3 惯量矩阵。
  float qi[4], l[3], Iw[3], Ia[3], cf[3];
  qmul(Q, c_iquat + 4 * b, qi);
  qrot_inv(qi, W, l);
  for (int k = 0; k < 3; k++) l[k] *= c_inert[3 * b + k];
  qrot(qi, l, Iw);
  qrot_inv(qi, AL, l);
  for (int k = 0; k < 3; k++) l[k] *= c_inert[3 * b + k];
  qrot(qi, l, Ia);
  // N = I*AL + W×(I*W) + c×F；最后一项将质心力矩平移到刚体原点。
  cross(W, Iw, u);
  cross(cvec, F, cf);
  for (int k = 0; k < 3; k++) N[k] = Ia[k] + u[k] + cf[k];
}

// RNE 从叶到根累加：子体的力直接相加，力矩还需加力臂 r×Fc。
__device__ __forceinline__ void accumulate(const float* Xb, const float* Xc, const float* Fc, const float* Nc, float* F,
                                           float* N) {
  float r[3] = {Xc[0] - Xb[0], Xc[1] - Xb[1], Xc[2] - Xb[2]}, u[3];
  cross(r, Fc, u);
  for (int k = 0; k < 3; k++) { F[k] += Fc[k]; N[k] += Nc[k] + u[k]; }
}

// 单 hinge 的广义力投影：从原 torques() 的 hinge 分支抽出，计算内容不变。
// 固定基座的第一个 hinge 也可以使用此入口，不进入 b==1 的自由根分支。
__device__ __forceinline__ void hinge_torque(int b, const float* Q, const float* N, float* qfrc) {
  float aw[3];
  qrot(Q, c_axis + 3 * b, aw);
  qfrc[c_dadr[b]] = dot3(aw, N);
}

// 将空间力投影到广义自由度：hinge 力矩为世界关节轴与原点力矩的点积。
// 自由根平移力在世界系，旋转力矩需变回与局部角速度共轭的局部系。
__device__ __forceinline__ void torques(int b, const float* Q, const float* F, const float* N, float* qfrc) {
  if (b == 1) {
    float l[3];
    qrot_inv(Q, N, l);
    for (int k = 0; k < 3; k++) { qfrc[k] = F[k]; qfrc[3 + k] = l[k]; }
  } else {
    hinge_torque(b, Q, N, qfrc);
  }
}

// 刚体 b 关于 O 的空间惯量：J[0:6]=(xx,yy,zz,xy,xz,yz)，J[6:9]=m*c，J[9]=m；c=质心-O
__device__ __forceinline__ void body_inertia(int b, const float* Q, const float* X, const float* O, const float* ipos,
                                             float* J) {
  float qi[4], R[9], c[3], t[3];
  qmul(Q, c_iquat + 4 * b, qi);
  for (int k = 0; k < 3; k++) {
    float e[3] = {0.f, 0.f, 0.f};
    e[k] = 1.f;
    qrot(qi, e, R + 3 * k);  // 惯性坐标系旋转矩阵的第 k 列
  }
  const float* I = c_inert + 3 * b;
  J[0] = I[0] * R[0] * R[0] + I[1] * R[3] * R[3] + I[2] * R[6] * R[6];
  J[1] = I[0] * R[1] * R[1] + I[1] * R[4] * R[4] + I[2] * R[7] * R[7];
  J[2] = I[0] * R[2] * R[2] + I[1] * R[5] * R[5] + I[2] * R[8] * R[8];
  J[3] = I[0] * R[0] * R[1] + I[1] * R[3] * R[4] + I[2] * R[6] * R[7];
  J[4] = I[0] * R[0] * R[2] + I[1] * R[3] * R[5] + I[2] * R[6] * R[8];
  J[5] = I[0] * R[1] * R[2] + I[1] * R[4] * R[5] + I[2] * R[7] * R[8];
  qrot(Q, ipos, t);
  for (int k = 0; k < 3; k++) c[k] = X[k] + t[k] - O[k];
  // 平行轴定理：I_O = I_COM + m*((c·c)*单位矩阵 - c*c^T)。
  float m = c_mass[b], cc = dot3(c, c);
  J[0] += m * (cc - c[0] * c[0]); J[1] += m * (cc - c[1] * c[1]); J[2] += m * (cc - c[2] * c[2]);
  J[3] -= m * c[0] * c[1]; J[4] -= m * c[0] * c[2]; J[5] -= m * c[1] * c[2];
  J[6] = m * c[0]; J[7] = m * c[1]; J[8] = m * c[2]; J[9] = m;
}

// 关于 O 的空间惯量乘运动向量：(n,f)=I*(w,v)，n=J*w+h×v，f=m*v-h×w
__device__ __forceinline__ void apply_inertia(const float* J, const float* w, const float* v, float* n, float* f) {
  float hv[3], hw[3];
  cross(J + 6, v, hv);
  cross(J + 6, w, hw);
  n[0] = J[0] * w[0] + J[3] * w[1] + J[4] * w[2] + hv[0];
  n[1] = J[3] * w[0] + J[1] * w[1] + J[5] * w[2] + hv[1];
  n[2] = J[4] * w[0] + J[5] * w[1] + J[2] * w[2] + hv[2];
  for (int k = 0; k < 3; k++) f[k] = J[9] * v[k] - hw[k];
}

// 刚体 k 的第 t 个自由度相对 O 的运动子空间：w 为角分量，v 为 O 点线速度分量
__device__ __forceinline__ void dof_motion(int k, int t, const float* Q, const float* X, const float* O, float* w,
                                           float* v) {
  if (k == 1) {
    float e[3] = {0.f, 0.f, 0.f};
    if (t < 3) { e[t] = 1.f; w[0] = w[1] = w[2] = 0.f; v[0] = e[0]; v[1] = e[1]; v[2] = e[2]; return; }
    e[t - 3] = 1.f;
    qrot(Q, e, w);  // 将局部角速度基轴旋转到世界系；此处 O 为根原点
    v[0] = v[1] = v[2] = 0.f;
    return;
  }
  // 单位关节角速度在 O 点产生的线速度为 w×(O-X)，供 CRB 质量矩阵投影使用。
  float r[3] = {O[0] - X[0], O[1] - X[1], O[2] - X[2]};
  qrot(Q, c_axis + 3 * k, w);
  cross(w, r, v);
}

// 刚体系内的质心位置；仅对指定躯干叠加每环境的质心偏移（域随机化）
__device__ __forceinline__ void body_ipos(int b, const float* com, float* out) {
  for (int k = 0; k < 3; k++) out[k] = c_ipos[3 * b + k] + (b == (int)c_misc[1] ? com[k] : 0.f);
}

// 一个 warp 原地分解 A=L*L^T：A 是共享内存中 NV×NV 稠密矩阵，使用其下三角。
// 采用左看式（Crout）更新：第 k 列依赖左侧已完成列；lane l 计算 k+1+l、k+33+l 等行。
// G1 的行跨度 NV=35 使各 lane 分布在不同共享内存 bank；第 k 行读取通过广播完成。
__device__ __forceinline__ void chol_warp(float* A, int lane) {
  for (int k = 0; k < NV; k++) {
    const float* Ak = A + k * NV;
    float part = 0.f;
    for (int j = lane; j < k; j += 32) part += Ak[j] * Ak[j];
    float d = sqrtf(fmaxf(Ak[k] - warp_sum(part), 1e-30f)), inv = 1.f / d;
    for (int i = k + 1 + lane; i < NV; i += 32) {
      const float* Ai = A + i * NV;
      float a0 = Ai[k], a1 = 0.f, a2 = 0.f, a3 = 0.f;
      int j = 0;
      for (; j + 4 <= k; j += 4) {
        a0 -= Ai[j] * Ak[j]; a1 -= Ai[j + 1] * Ak[j + 1]; a2 -= Ai[j + 2] * Ak[j + 2]; a3 -= Ai[j + 3] * Ak[j + 3];
      }
      for (; j < k; j++) a0 -= Ai[j] * Ak[j];
      A[i * NV + k] = ((a0 + a1) + (a2 + a3)) * inv;
    }
    __syncwarp();
    if (lane == 0) A[k * NV + k] = d;
    __syncwarp();
  }
}

// 在寄存器中分解并求解：先求 A=L*L^T，再原地将共享向量 b 替换为 A^{-1}b。
// A/L 是按 TRI(i,j) 存储的下三角；先读完 A 再写 L，因此两者允许共用存储。
// lane l 持有第 l 和 l+32 行（NV<=64）；右看式分解用 __shfl_sync 广播主元列，
// 循环在编译时展开。输出 L 保留真实对角元，供回代和后续复用。
// 注意：r0=lane 没有 r0<NV 保护，所以当前 NV>=32 限制不可直接删除。
#ifndef CHOL_INLINE
#define CHOL_INLINE __noinline__  // 保留一份较大的完全展开代码；-DCHOL_INLINE=__forceinline__ 可强制内联
#endif
__device__ CHOL_INLINE void chol_solve_reg(const float* A, float* L, float* b, int lane) {
  static_assert(NV <= 64, "two rows per lane");
  const int r0 = lane, r1 = lane + 32;
  const bool has1 = r1 < NV;
  float a0[NV], a1[NV], dinv[NV];
#pragma unroll
  for (int j = 0; j < NV; j++) { a0[j] = j <= r0 ? A[TRI(r0, j)] : 0.f; a1[j] = has1 && j <= r1 ? A[TRI(r1, j)] : 0.f; }
  __syncwarp();
#pragma unroll
  for (int k = 0; k < NV; k++) {
    float dkk = fmaxf(__shfl_sync(0xffffffffu, k < 32 ? a0[k] : a1[k], k & 31), 1e-30f);
    float inv = rsqrtf(dkk);
    inv = inv * (1.5f - 0.5f * dkk * inv * inv);  // 对倒平方根做一次 Newton 精化，接近 float 完整精度
    float d = dkk * inv;
    dinv[k] = inv;
    if (r0 > k) a0[k] *= inv; else if (r0 == k) a0[k] = d;
    if (r1 > k) a1[k] *= inv; else if (r1 == k) a1[k] = d;
#pragma unroll
    for (int j = k + 1; j < NV; j++) {  // 不加条件屏蔽：对角线上方也更新对称 Schur 补
      float ljk = __shfl_sync(0xffffffffu, j < 32 ? a0[k] : a1[k], j & 31);  // 数值有界，非下三角部分不会作为 L 读取
      a0[j] -= a0[k] * ljk;
      a1[j] -= a1[k] * ljk;
    }
  }
#pragma unroll
  for (int j = 0; j < NV; j++) { if (j <= r0) L[TRI(r0, j)] = a0[j]; if (has1 && j <= r1) L[TRI(r1, j)] = a1[j]; }
  float y0 = b[r0], y1 = has1 ? b[r1] : 0.f;
#pragma unroll
  for (int k = 0; k < NV; k++) {  // 逐列前代：L*y=b
    float yk = __shfl_sync(0xffffffffu, k < 32 ? y0 : y1, k & 31) * dinv[k];
    if (r0 == k) y0 = yk; else if (r0 > k) y0 -= a0[k] * yk;
    if (r1 == k) y1 = yk; else if (r1 > k) y1 -= a1[k] * yk;
  }
  __syncwarp();
  // 回代 L^T*x=y；lane i 保存 y_i，连续读取 L 的第 k 行并更新 i<k 的元素
#pragma unroll
  for (int k = NV - 1; k >= 0; k--) {
    float xk = __shfl_sync(0xffffffffu, k < 32 ? y0 : y1, k & 31) * dinv[k];
    if (r0 == k) y0 = xk; else if (r0 < k) y0 -= L[TRI(k, r0)] * xk;
    if (r1 == k) y1 = xk; else if (r1 < k) y1 -= L[TRI(k, r1)] * xk;
  }
  b[r0] = y0;
  if (has1) b[r1] = y1;
  __syncwarp();
}

// chol_solve_reg 的紧凑版本，输入输出约定相同，但主元列循环不完全展开。
// 寄存器 a0/a1 保存 lane/lane+32 行的剩余部分，每步左移一个元素；
// 第 k 步 a0[m] 对应该行第 k+m 列，使寄存器索引保持编译期常量。
// 完成的 L 列立即写回共享内存；剩余更新按每组 8 列处理，无列时停止。
// 代码较小、减少取指停顿，但执行指令数多于完全展开版本。
__device__ __noinline__ void chol_solve_compact(const float* A, float* L, float* b, int lane) {
  static_assert(NV <= 64, "two rows per lane");
  const int r0 = lane, r1 = lane + 32;
  const bool has1 = r1 < NV;
  float a0[NV], a1[NV];
#pragma unroll
  for (int j = 0; j < NV; j++) { a0[j] = j <= r0 ? A[TRI(r0, j)] : 0.f; a1[j] = has1 && j <= r1 ? A[TRI(r1, j)] : 0.f; }
  __syncwarp();
  for (int k = 0; k < NV; k++) {
    float dkk = fmaxf(__shfl_sync(0xffffffffu, k < 32 ? a0[0] : a1[0], k & 31), 1e-30f);
    float inv = rsqrtf(dkk);
    inv = inv * (1.5f - 0.5f * dkk * inv * inv);
    float d = dkk * inv;
    float l0 = r0 > k ? a0[0] * inv : (r0 == k ? d : 0.f), l1 = r1 > k ? a1[0] * inv : (r1 == k ? d : 0.f);
    if (r0 >= k) L[TRI(r0, k)] = l0;
    if (has1 && r1 >= k) L[TRI(r1, k)] = l1;
    if (r0 == k) l0 = 0.f;  // 主元行不参与剩余子矩阵更新
    if (r1 == k) l1 = 0.f;
#pragma unroll
    for (int g = 1; g < NV; g += 8) {
      if (k + g >= NV) break;  // 所有 lane 一致退出：本组已无剩余列
#pragma unroll
      for (int m = g; m < g + 8 && m < NV; m++) {
        int j = k + m;
        float ljk = __shfl_sync(0xffffffffu, j < 32 ? l0 : l1, j & 31);
        a0[m - 1] = a0[m] - l0 * ljk;
        a1[m - 1] = a1[m] - l1 * ljk;
      }
    }
  }
  __syncwarp();
  float y0 = b[r0], y1 = has1 ? b[r1] : 0.f;
  for (int k = 0; k < NV; k++) {  // 逐列前代：L*y=b
    float yk = __shfl_sync(0xffffffffu, k < 32 ? y0 : y1, k & 31) / L[TRI(k, k)];
    if (r0 == k) y0 = yk; else if (r0 > k) y0 -= L[TRI(r0, k)] * yk;
    if (r1 == k) y1 = yk; else if (has1 && r1 > k) y1 -= L[TRI(r1, k)] * yk;
  }
  for (int k = NV - 1; k >= 0; k--) {  // 回代：L^T*x=y
    float xk = __shfl_sync(0xffffffffu, k < 32 ? y0 : y1, k & 31) / L[TRI(k, k)];
    if (r0 == k) y0 = xk; else if (r0 < k) y0 -= L[TRI(k, r0)] * xk;
    if (r1 == k) y1 = xk; else if (has1 && r1 < k) y1 -= L[TRI(k, r1)] * xk;
  }
  b[r0] = y0;
  if (has1) b[r1] = y1;
  __syncwarp();
}

// 广播优化版本，适用于 32<NV<=35；额外需要 cb：72 个 float、16 字节对齐的共享暂存区。
// reg/compact 版本在剩余更新中逐元素 shuffle 广播列，开销较大；本版本先将第 k 列
// 每 lane 写一次共享内存，再用 float4 广播读取，一次取四项。
// 前 32 行在寄存器中逐步左移；尾部 NT=NV-32 行按转置方式存储（lane j 保存 A[32+t][j]），
// 每个 lane 还冗余保存 NT×NT 尾块，因此只需更新本行第 32 列左侧的元素。
// 前代融合在分解中：lane k 将 y_k 和第 k 列一起广播；lane k+1 用自身数据准备下一主元。
// 回代按行读取 L。L 与 compact 版本更新顺序相同，求解乘以精化后的对角倒数。
// 原作者的独立微基准（35×35、4096 环境，每次分解加求解周期数）：
// V100 每 SM 3 环境为 8.2k（compact 31.6k、reg 16.2k）；
// RTX 3090 每 SM 2 环境为 8.9k（compact 51.9k、reg 19.3k）。
// 引擎中 V100 的 qacc_smooth 阶段由 44.8k 降至 10.8k 周期；这些不是当前机器的实测。
#if NV > 32 && NV <= 35
#define CHOL_NT (NV - 32)  // 尾部行数
// 下列 FMS（一次 FMA 求 c-a*b）和 MUL 固定舍入，避免 ptxas 随调用 kernel 改变乘加融合。
// k_substep、k_pre、k_post 的结果必须逐位一致。
#define FMS(a, b, c) __fmaf_rn(-(a), (b), (c))
#define MUL(a, b) __fmul_rn((a), (b))
// 正规数 x>=2^-126 的倒平方根：直接使用 MUFU.RSQ，省去非正规数缩放，结果相同
__device__ __forceinline__ float rsqrt_normal(float x) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
// 对前 32 行分块回代 L^T*x=y，每块 8 行；lane r 输入 y_r、输出 x_r，尾行贡献已扣除。
// 每个 lane 冗余求解块内三角系统（广播读取 L；rinv 为共享内存中的对角倒数），
// 再更新本行：每块进行一轮 shuffle，而非每行一轮。
__device__ __forceinline__ float chol_back32(const float* L, const float* rinv, float y, int lane) {
#pragma unroll
  for (int kb = 24; kb >= 0; kb -= 8) {
    float xb[8];
#pragma unroll
    for (int j = 0; j < 8; j++) xb[j] = __shfl_sync(0xffffffffu, y, kb + j);
#pragma unroll
    for (int j = 7; j >= 0; j--) {
      float t = xb[j];
#pragma unroll
      for (int i = j + 1; i < 8; i++) t = FMS(L[TRI(kb + i, kb + j)], xb[i], t);
      xb[j] = MUL(t, rinv[kb + j]);
    }
#pragma unroll
    for (int j = 0; j < 8; j++)
      y = lane < kb ? FMS(L[TRI(kb + j, min(lane, kb + j))], xb[j], y) : (lane == kb + j ? xb[j] : y);
  }
  return y;
}

__device__ __noinline__ void chol_solve_bcast(const float* A, float* L, float* b, int lane, float* cb) {
  static_assert(NV > 32 && NV <= 35, "a tail of 1..3 rows");
  const int r = lane;
  float a[32];  // 第 k 步 a[m]=A[r][k+m]；对角线右侧是有界占位值，不作为 L 读取
#pragma unroll
  for (int j = 0; j < 32; j++) a[j] = j <= r ? A[TRI(r, j)] : 0.f;
  float T[CHOL_NT], tb[CHOL_NT * (CHOL_NT + 1) / 2], bt[CHOL_NT];  // 分别保存 A[32+t][r]、压缩尾块、右端向量尾部
#pragma unroll
  for (int t = 0; t < CHOL_NT; t++) {
    T[t] = A[TRI(32 + t, r)];
    bt[t] = b[32 + t];
#pragma unroll
    for (int u = 0; u <= t; u++) tb[t * (t + 1) / 2 + u] = A[TRI(32 + t, 32 + u)];
  }
  float y = b[r], myinv = 0.f;
  __syncwarp();
  float dn = a[0];  // 下一主元（第 k 步由 lane k 持有）：已完成先前更新的对角元
  // 原作者测试选择展开因子 2。展开因子 1：V100 每控制步 13.7 对 12.7 ms，
  // RTX 3090 为 7.59 对 7.46 ms；逐组判断或按剩余列数分阶段曾使 V100 达到 13.0/15.2 ms。
#pragma unroll 2
  for (int k = 0; k < 32; k++) {
    float* buf = cb + (k & 1) * 36;  // 双缓冲：[0,4) 放尾行列值及 y_k；4+(j-k-1) 放 L[j][k]，k<j<32
    float dkk = fmaxf(__shfl_sync(0xffffffffu, dn, k), 1e-30f);
    float inv = rsqrt_normal(dkk);
    inv = MUL(inv, FMS(MUL(MUL(0.5f, dkk), inv), inv, 1.5f));  // 对倒平方根做一次 Newton 精化，接近 float 完整精度
    float d = MUL(dkk, inv);
    float l = r > k ? MUL(a[0], inv) : (r == k ? d : 0.f);
    if (r >= k) L[TRI(r, k)] = l;
    if (r == k) {
      myinv = inv;
      y = MUL(y, inv);
      float tw[4] = {0.f, 0.f, 0.f, y};
#pragma unroll
      for (int t = 0; t < CHOL_NT; t++) { T[t] = MUL(T[t], inv); L[TRI(32 + t, k)] = T[t]; tw[t] = T[t]; }
      *reinterpret_cast<float4*>(buf) = make_float4(tw[0], tw[1], tw[2], tw[3]);
      l = 0.f;  // 主元行不参与剩余子矩阵更新
    }
    if (r > k) buf[4 + r - k - 1] = l;
    dn = FMS(l, l, a[1]);  // lane k+1 的下一对角元，等于下面左移后的 a[0]
    __syncwarp();
    float4 tv = *reinterpret_cast<const float4*>(buf);
    float tl[3] = {tv.x, tv.y, tv.z};
    y = FMS(l, tv.w, y);
#pragma unroll
    for (int t = 0; t < CHOL_NT; t++) {
      T[t] = FMS(tl[t], l, T[t]);
      bt[t] = FMS(tl[t], tv.w, bt[t]);
#pragma unroll
      for (int u = 0; u <= t; u++) tb[t * (t + 1) / 2 + u] = FMS(tl[t], tl[u], tb[t * (t + 1) / 2 + u]);
    }
#pragma unroll
    for (int g = 1; g < 32; g += 4) {  // 每步处理全部分组；曾尝试 k>=16 时用一致分支跳过后半组，
      float4 c = *reinterpret_cast<const float4*>(buf + 4 + g - 1);  // 但更慢：原作者 RTX 3090 为 9.9k 对 8.9k 周期
      a[g - 1] = FMS(l, c.x, a[g]);
      if (g + 1 < 32) a[g] = FMS(l, c.y, a[g + 1]);
      if (g + 2 < 32) a[g + 1] = FMS(l, c.z, a[g + 2]);
      if (g + 3 < 32) a[g + 2] = FMS(l, c.w, a[g + 3]);
    }
  }
  float dt[CHOL_NT], it[CHOL_NT], xt[CHOL_NT];  // 每个 lane 冗余完成尾块分解与求解
#pragma unroll
  for (int t = 0; t < CHOL_NT; t++) {
    float s = tb[t * (t + 1) / 2 + t];
#pragma unroll
    for (int u = 0; u < t; u++) s = FMS(tb[t * (t + 1) / 2 + u], tb[t * (t + 1) / 2 + u], s);
    float dkk = fmaxf(s, 1e-30f), inv = rsqrt_normal(dkk);
    inv = MUL(inv, FMS(MUL(MUL(0.5f, dkk), inv), inv, 1.5f));
    dt[t] = MUL(dkk, inv); it[t] = inv;
#pragma unroll
    for (int v = t + 1; v < CHOL_NT; v++) {
      float e = tb[v * (v + 1) / 2 + t];
#pragma unroll
      for (int u = 0; u < t; u++) e = FMS(tb[v * (v + 1) / 2 + u], tb[t * (t + 1) / 2 + u], e);
      tb[v * (v + 1) / 2 + t] = MUL(e, inv);
    }
    float yt = bt[t];
#pragma unroll
    for (int u = 0; u < t; u++) yt = FMS(tb[t * (t + 1) / 2 + u], xt[u], yt);
    xt[t] = MUL(yt, inv);  // 前代
  }
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < CHOL_NT; t++) {
#pragma unroll
      for (int u = 0; u < t; u++) L[TRI(32 + t, 32 + u)] = tb[t * (t + 1) / 2 + u];
      L[TRI(32 + t, 32 + t)] = dt[t];
    }
  }
#pragma unroll
  for (int t = CHOL_NT - 1; t >= 0; t--) {  // 尾部回代
    float x = xt[t];
#pragma unroll
    for (int v = t + 1; v < CHOL_NT; v++) x = FMS(tb[v * (v + 1) / 2 + t], xt[v], x);
    xt[t] = MUL(x, it[t]);
  }
#pragma unroll
  for (int t = 0; t < CHOL_NT; t++) y = FMS(T[t], xt[t], y);
  __syncwarp();
  // 对前 32 行逐列回代 L^T*x=y，展开因子 4。完全展开时 V100 为 12.73 对 12.44 ms，
  // RTX 3090 无差别；此处改用 chol_back32 反而更慢（原作者测试）。
#pragma unroll 4
  for (int k = 31; k >= 0; k--) {
    float xk = __shfl_sync(0xffffffffu, MUL(y, myinv), k);
    float lk = r < k ? L[TRI(k, r)] : 0.f;
    y = r == k ? xk : FMS(lk, xk, y);
  }
  b[r] = y;
  if (lane == 0) {  // 若让 lane t 写第 t 项，会产生动态索引的局部数组
#pragma unroll
    for (int t = 0; t < CHOL_NT; t++) b[32 + t] = xt[t];
  }
  __syncwarp();
}

// 已有 CHOL_SOLVE 的分解 L 时，原地求 x=(L*L^T)^{-1}b；b 为共享内存 NV 向量。
// 用于 Newton 迭代中 Hessian 未变化的情况，避免重复分解；cb 与 bcast 版本约定相同。
// 前代/回代均采用 8 行块，尾行由每个 lane 冗余处理。
// 原作者 V100 测试为 3.0k 周期，逐行广播版本为 6.3k 周期。
__device__ __noinline__ void chol_resolve(const float* L, float* b, int lane, float* cb) {
  const int r = lane;
  float* rinv = cb;
  rinv[r] = 1.f / L[TRI(r, r)];
  if (r < CHOL_NT) rinv[32 + r] = 1.f / L[TRI(32 + r, 32 + r)];
  const int r1 = min(32 + r, NV - 1);  // 前 CHOL_NT 个 lane 对应尾行 32+r（y1），其他 lane 使用合法占位行
  float y = b[r], y1 = b[r1];
  const int tr = TRI(r, 0), tr1 = TRI(r1, 0);
  __syncwarp();
#pragma unroll
  for (int kb = 0; kb < 32; kb += 8) {  // 对前 32 行前代：L*y=b
    float yb[8];
#pragma unroll
    for (int j = 0; j < 8; j++) yb[j] = __shfl_sync(0xffffffffu, y, kb + j);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float t = yb[j];
#pragma unroll
      for (int i = 0; i < j; i++) t = FMS(L[TRI(kb + j, kb + i)], yb[i], t);
      yb[j] = MUL(t, rinv[kb + j]);
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
      y = r >= kb + 8 ? FMS(L[tr + min(kb + j, r)], yb[j], y) : (r == kb + j ? yb[j] : y);
      y1 = FMS(L[tr1 + kb + j], yb[j], y1);
    }
  }
  float yt[CHOL_NT];  // 每个 lane 保存尾行解：先前代、再回代，最后扣除它们对前 32 行的贡献
#pragma unroll
  for (int t = 0; t < CHOL_NT; t++) {
    float u = __shfl_sync(0xffffffffu, y1, t);
#pragma unroll
    for (int i = 0; i < t; i++) u = FMS(L[TRI(32 + t, 32 + i)], yt[i], u);
    yt[t] = MUL(u, rinv[32 + t]);
  }
#pragma unroll
  for (int t = CHOL_NT - 1; t >= 0; t--) {
    float u = yt[t];
#pragma unroll
    for (int i = t + 1; i < CHOL_NT; i++) u = FMS(L[TRI(32 + i, 32 + t)], yt[i], u);
    yt[t] = MUL(u, rinv[32 + t]);
  }
#pragma unroll
  for (int t = 0; t < CHOL_NT; t++) y = FMS(L[TRI(32 + t, r)], yt[t], y);
  y = chol_back32(L, rinv, y, lane);
  b[r] = y;
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < CHOL_NT; t++) b[32 + t] = yt[t];
  }
  __syncwarp();
}
#undef FMS
#undef MUL
#endif

// 选择实际使用的 Cholesky：统一入口 CHOL_SOLVE(A,L,b,lane,cb)，cb 供 bcast 暂存。
// 维度允许时默认 CHOL_BCAST=1；配合 NEWTON_REUSE，原作者每控制步 10 次迭代测试：
// V100 20.5→12.4 ms，RTX 3090 11.1→7.5 ms。
// CHOL_BCAST=0 时，CHOL_COMPACT=1/0 选择紧凑/展开版本；Volta 默认紧凑（V100 26 对 34 ms），
// Ampere 及以后默认展开（RTX 3090 14.7 对 17.9 ms）。
#ifndef CHOL_BCAST
#define CHOL_BCAST (NV > 32 && NV <= 35)
#endif
#ifndef CHOL_COMPACT
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define CHOL_COMPACT 1
#else
#define CHOL_COMPACT 0
#endif
#endif
#if CHOL_BCAST
#define CHOL_SOLVE(A, L, b, lane, cb) chol_solve_bcast(A, L, b, lane, cb)
#elif CHOL_COMPACT
#define CHOL_SOLVE(A, L, b, lane, cb) chol_solve_compact(A, L, b, lane)
#else
#define CHOL_SOLVE(A, L, b, lane, cb) chol_solve_reg(A, L, b, lane)
#endif

// 使用 chol_warp 的分解，原地求 L*L^T*x=b；共享内存向量 b 最终存放解 x
__device__ __forceinline__ void chol_solve_warp(const float* L, float* b, int lane) {
  for (int k = 0; k < NV; k++) {  // 前代：L*y=b
    if (lane == 0) b[k] /= L[k * NV + k];
    __syncwarp();
    for (int i = k + 1 + lane; i < NV; i += 32) b[i] -= L[i * NV + k] * b[k];
    __syncwarp();
  }
  for (int k = NV - 1; k >= 0; k--) {  // 回代：L^T*x=y
    if (lane == 0) b[k] /= L[k * NV + k];
    __syncwarp();
    for (int i = lane; i < k; i += 32) b[i] -= L[k * NV + i] * b[k];
    __syncwarp();
  }
}
"""
