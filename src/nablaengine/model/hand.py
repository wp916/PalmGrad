"""手动力学的数据接口：定义输入结构、检查数据、转换状态数组的排列。

这个文件做什么
    把“手指关节 + 被操作物体”的状态组织成明确的数据结构，并记录每段数组的地址。
    它不检测碰撞、不求接触力，也不推进仿真时间；实际步进在 hand_engine/implicit_step.py。
    MuJoCo 在这里用于读取已编译模型的结构与执行器参数，没有调用其动力学求解器。

主要接口与数据流
    layout = from_mujoco(model, palm='palm', objects=('object0',))
        输入：MuJoCo 模型、手掌名称、物体名称，以及可选的手指关节顺序。
        输出：HandLayout，保存源模型与手引擎之间的地址映射。
    state = layout.pack(source_qpos, source_qvel)
        输入：源模型顺序的一维状态数组；输出：手引擎顺序的 HandState。
    inputs = HandInput(state=state, palm=PalmMotion.fixed(), command=command)
        将当前状态、规定的手掌运动和关节控制命令放在一个容器中。
    torque = layout.validate_input(inputs)
        检查输入并返回 (n,) 的期望关节力矩，单位 N·m；不进行动力学步进。
    source_qpos, source_qvel = layout.unpack(state)
        将状态转换回源模型顺序，返回两个独立的可写数组。

数组约定（单环境，不含批量维度）
    设手指有 n 个单自由度关节，被操作物体有 k 个，0 <= k <= 4。
    qpos: (nq,)，nq=n+7*k，排列为 [全部手指角度, 物体0的xyz+wxyz, 物体1的..., ...]。
    qvel: (nv,)，nv=n+6*k，排列为 [全部手指角速度, 物体0的线速度+角速度, ...]。
    手指角度用 rad、角速度用 rad/s；位置用 m、线速度用 m/s。
    物体位置和线速度在世界系，物体角速度在物体局部系；姿态四元数顺序为 wxyz。
    手掌位姿、速度、加速度独立放在 PalmMotion 中，不占上述状态数组的位置。
    手掌线/角速度及线/角加速度均在世界系；四元数表示局部坐标系相对世界系的姿态。

支持范围与数据保护
    本布局接口面向规定运动的手掌、直驱 hinge，以及最多四个独立单刚体 free 物体。
    通过这里的结构检查，不代表几何类型、物理选项和后端容量均已支持，后端还需独立检查。
    数据类的 frozen=True 禁止普通字段重新赋值；初始化中用 object.__setattr__ 写入校验结果。
    _array 会复制为有限的 float64 数组并设为只读，不修改或共享调用方输入数组。
    输入非法时抛出 ValueError 或 TypeError，不静默调整长度或归一化四元数。
"""
from dataclasses import dataclass

import mujoco
import numpy as np


def _array(value, shape, name):
    """复制为指定形状的有限 float64 数组，并禁止普通元素写入。"""
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} 必须是有限实数，形状为 {shape}，实际为 {result.shape}')
    result.setflags(write=False)
    return result


def _quaternion(q, name):
    """检查四元数长度是否接近 1；通过时返回 None，不修改 q。形状/有限性由调用方检查。"""
    if not np.isclose(np.linalg.norm(q), 1.0, rtol=0, atol=1e-6):
        raise ValueError(f'{name} 必须是单位四元数 wxyz；接口不自动归一化')


@dataclass(frozen=True)
class PalmMotion:
    """外部规定轨迹；加速度是物理加速度，不包含人为减去的重力。

    输入为同一时刻的六个向量，创建后均保存为独立、只读的 float64 数组。
    手掌运动是外部给定的，不作为自由根由本接口求解；固定手掌用 fixed() 创建。
    不从相邻帧猜测速度或加速度。调用方保证轨迹的时间一致性。
    """
    position: np.ndarray              # (3,)，世界系位置，m
    quaternion: np.ndarray            # (4,)，姿态四元数 wxyz，必须为单位长度
    linear_velocity: np.ndarray       # (3,)，世界系线速度，m/s
    angular_velocity: np.ndarray      # (3,)，世界系角速度，rad/s
    linear_acceleration: np.ndarray   # (3,)，世界系线加速度，m/s²
    angular_acceleration: np.ndarray  # (3,)，世界系角加速度，rad/s²

    def __post_init__(self):
        # 逐项列出字段，避免通过字符串反射查找成员。
        # frozen=True 禁止 self.position = ...，这里只在初始化时调用底层赋值。
        position = _array(self.position, (3,), 'palm.position')
        object.__setattr__(self, 'position', position)
        quaternion = _array(self.quaternion, (4,), 'palm.quaternion')
        object.__setattr__(self, 'quaternion', quaternion)
        linear_velocity = _array(self.linear_velocity, (3,), 'palm.linear_velocity')
        object.__setattr__(self, 'linear_velocity', linear_velocity)
        angular_velocity = _array(self.angular_velocity, (3,), 'palm.angular_velocity')
        object.__setattr__(self, 'angular_velocity', angular_velocity)
        linear_acceleration = _array(self.linear_acceleration, (3,), 'palm.linear_acceleration')
        object.__setattr__(self, 'linear_acceleration', linear_acceleration)
        angular_acceleration = _array(self.angular_acceleration, (3,), 'palm.angular_acceleration')
        object.__setattr__(self, 'angular_acceleration', angular_acceleration)
        _quaternion(self.quaternion, 'palm.quaternion')

    @classmethod
    def fixed(cls, position=(0, 0, 0), quaternion=(1, 0, 0, 0)):
        """固定是位姿恒定、速度和加速度全部为零，位姿不必位于世界原点。"""
        return cls(
            position=position,
            quaternion=quaternion,
            linear_velocity=(0, 0, 0),
            angular_velocity=(0, 0, 0),
            linear_acceleration=(0, 0, 0),
            angular_acceleration=(0, 0, 0),
        )


@dataclass(frozen=True)
class StateBlock:
    """一个关节或物体的数组地址表。

    qpos_start/qvel_start 是手引擎内的起始地址；source_* 是源模型地址。
    hinge 占 nq=1、nv=1；free_object 占 nq=7、nv=6。
    """
    name: str                   # hinge 的关节名，或 free_object 的刚体名
    kind: str                   # 'hinge' 或 'free_object'
    body_id: int                # 源模型中的刚体编号，不是状态数组地址
    joint_id: int               # 源模型中的关节编号
    qpos_start: int             # 内部 qpos 的起始地址，从 0 开始
    qvel_start: int             # 内部 qvel 的起始地址，从 0 开始
    nq: int                     # 本块占用的位置分量数
    nv: int                     # 本块占用的速度分量数
    source_qpos_start: int      # 源模型 qpos 中的起始地址
    source_qvel_start: int      # 源模型 qvel 中的起始地址

    # @property 让调用方用 block.qpos_slice 读取下列函数的返回值，无需加括号。
    @property
    def qpos_slice(self):
        """返回内部位置块的切片 [起点, 终点)，终点不包含在内。"""
        return slice(self.qpos_start, self.qpos_start + self.nq)

    @property
    def qvel_slice(self):
        """返回内部速度块的切片；物体的位置块为 7 项，速度块为 6 项。"""
        return slice(self.qvel_start, self.qvel_start + self.nv)


@dataclass(frozen=True)
class DirectDrive:
    """一个手指执行器的源地址、内部自由度编号和源模型 PD 参数。"""
    joint_name: str
    dof: int                     # 内部手指自由度编号，也是力矩输出的下标
    source_actuator_id: int       # 源模型执行器编号
    source_control_address: int   # 源模型 ctrl 数组中的地址，不假定等于执行器编号
    source_kind: str              # 源模型驱动类型：'torque' 或 'position_pd'
    kp: float                    # 源模型位置增益，N·m/rad；力矩驱动时记为 0
    kv: float                    # 源模型速度增益，N·m·s/rad；力矩驱动时记为 0


@dataclass(frozen=True)
class HandState:
    """单环境状态：qpos 为 (nq,)，qvel 为 (nv,)；具体长度由 HandLayout 校验。"""
    qpos: np.ndarray
    qvel: np.ndarray

    def __post_init__(self):
        qpos = np.asarray(self.qpos)
        if qpos.ndim != 1:
            raise ValueError('qpos 当前接口仅接受单环境一维数组')
        qpos = _array(qpos, qpos.shape, 'qpos')
        object.__setattr__(self, 'qpos', qpos)

        qvel = np.asarray(self.qvel)
        if qvel.ndim != 1:
            raise ValueError('qvel 当前接口仅接受单环境一维数组')
        qvel = _array(qvel, qvel.shape, 'qvel')
        object.__setattr__(self, 'qvel', qvel)


@dataclass(frozen=True)
class DirectDriveCommand:
    """torque: values 单位Nm；position_pd: values 单位rad，kp/kv单位Nm/rad、Nm·s/rad。

    values 为 (n,) 数组，按内部手指关节顺序排列，不能直接假定等于源模型执行器顺序。
    torque 模式：values 是关节力矩，kp/kv 必须为 None。
    position_pd 模式：values 是目标关节角，kp/kv 必须为同形状的非负数组。
    PD输出为 kp*(target-q)-kv*v；它是无饱和的期望关节力矩，未应用电机限幅。
    本结构保存的是本次命令；DirectDrive 则记录从源模型读取的执行器配置。
    """
    mode: str
    values: np.ndarray
    kp: np.ndarray | None = None
    kv: np.ndarray | None = None

    def __post_init__(self):
        if self.mode not in ('torque', 'position_pd'):
            raise ValueError('mode 必须是 torque 或 position_pd')
        values = np.asarray(self.values)
        if values.ndim != 1:
            raise ValueError('values 必须是一维手指控制数组')
        object.__setattr__(self, 'values', _array(values, values.shape, 'command.values'))
        if self.mode == 'torque':
            if self.kp is not None or self.kv is not None:
                raise ValueError('torque 模式不接受 kp/kv')
        else:
            kp = _array(self.kp, values.shape, 'command.kp')
            if np.any(kp < 0):
                raise ValueError('PD增益必须非负')
            object.__setattr__(self, 'kp', kp)

            kv = _array(self.kv, values.shape, 'command.kv')
            if np.any(kv < 0):
                raise ValueError('PD增益必须非负')
            object.__setattr__(self, 'kv', kv)


@dataclass(frozen=True)
class HandInput:
    """单环境、同一时刻的完整输入；手掌轨迹不可省略。

    state：内部排列的运动状态；palm：外部规定的手掌运动；command：手指驱动命令。
    创建时只检查容器类型，状态长度与控制长度需由 layout.validate_input() 进一步检查。
    """
    state: HandState
    palm: PalmMotion
    command: DirectDriveCommand

    def __post_init__(self):
        if not isinstance(self.state, HandState) or not isinstance(self.palm, PalmMotion):
            raise TypeError('必须提供HandState和PalmMotion，不能隐式假定基座静止')
        if not isinstance(self.command, DirectDriveCommand):
            raise TypeError('必须提供DirectDriveCommand')


@dataclass(frozen=True)
class HandLayout:
    """源模型与手引擎之间的地址映射，不保存当前时刻的运动状态。

    blocks 按“全部手指关节、全部物体”排序；actuators 按内部手指自由度排序。
    source_nq/source_nv 为源模型数组长度，nq/nv 属性为内部数组长度。
    """
    palm_name: str
    palm_body_id: int
    blocks: tuple[StateBlock, ...]
    actuators: tuple[DirectDrive, ...]
    source_nq: int
    source_nv: int
    body_count: int
    max_children: int             # 非 world 刚体中，直接子体数量的最大值
    max_hinge_ancestors: int       # 一个手指刚体上方的 hinge 祖先数最大值，不含自身

    @property
    def hand_dofs(self):
        """手指自由度数 n；每个 hinge 恰好对应一个直驱执行器。"""
        return len(self.actuators)

    @property
    def object_count(self):
        """独立自由物体数 k，不包括手掌和手指刚体。"""
        count = 0
        for block in self.blocks:
            if block.kind == 'free_object':
                count += 1
        return count

    @property
    def nq(self):
        """内部 qpos 长度；四元数占四项，所以不一定等于 nv。"""
        return self.hand_dofs + 7 * self.object_count

    @property
    def nv(self):
        """内部 qvel 长度；每个自由物体占六个速度自由度。"""
        return self.hand_dofs + 6 * self.object_count

    def validate_state(self, state):
        """检查 HandState 的长度、有限性和物体单位四元数；成功返回 None，不修改输入。"""
        _array(state.qpos, (self.nq,), 'state.qpos')
        _array(state.qvel, (self.nv,), 'state.qvel')
        for block in self.blocks:
            if block.kind == 'free_object':
                _quaternion(state.qpos[block.qpos_start + 3:block.qpos_start + 7], block.name)

    def pack(self, source_qpos, source_qvel):
        """源模型顺序 → 内部顺序，只复制和重排，不进行运动学或动力学计算。

        输入：source_qpos 为 (source_nq,)，source_qvel 为 (source_nv,)。
        输出：HandState，其中数组为 (nq,)/(nv,) 的只读 float64 独立副本。
        地址来自 blocks，不要求源模型先放手再放物体，也不依赖执行器排列。
        """
        source_qpos = _array(source_qpos, (self.source_nq,), 'source_qpos')
        source_qvel = _array(source_qvel, (self.source_nv,), 'source_qvel')
        qpos = np.empty(self.nq)
        qvel = np.empty(self.nv)
        for block in self.blocks:
            source_qpos_end = block.source_qpos_start + block.nq
            source_qvel_end = block.source_qvel_start + block.nv
            qpos[block.qpos_slice] = source_qpos[block.source_qpos_start:source_qpos_end]
            qvel[block.qvel_slice] = source_qvel[block.source_qvel_start:source_qvel_end]
        state = HandState(qpos=qpos, qvel=qvel)
        self.validate_state(state)
        return state

    def unpack(self, state):
        """内部顺序 → 源模型顺序，是 pack 的反向地址转换。

        输入：与本布局匹配的 HandState。
        输出：(source_qpos, source_qvel)，长度为 source_nq/source_nv，可写且不共享输入。
        手掌无动力学关节，其规定运动单独存放，不会写入这两个数组。
        """
        self.validate_state(state)
        source_qpos = np.empty(self.source_nq)
        source_qvel = np.empty(self.source_nv)
        for block in self.blocks:
            source_qpos_end = block.source_qpos_start + block.nq
            source_qvel_end = block.source_qvel_start + block.nv
            source_qpos[block.source_qpos_start:source_qpos_end] = state.qpos[block.qpos_slice]
            source_qvel[block.source_qvel_start:source_qvel_end] = state.qvel[block.qvel_slice]
        return source_qpos, source_qvel

    def desired_torque(self, state, command):
        """由 HandState 和 DirectDriveCommand 返回内部关节顺序的力矩 (n,)，单位 N·m。

        力矩模式直接复制输入；位置模式按本次 command 的 kp/kv 计算 PD。
        不修改状态、不施加限幅、不计算接触反力，输出中也不包含物体自由度。
        """
        self.validate_state(state)
        if command.values.shape != (self.hand_dofs,):
            raise ValueError('每个手指hinge必须恰好一个控制输入；物体不接收电机命令')
        if command.mode == 'torque':
            return command.values.copy()
        hand_dofs = self.hand_dofs
        joint_position = state.qpos[:hand_dofs]
        joint_velocity = state.qvel[:hand_dofs]
        position_error = command.values - joint_position
        desired_torque = command.kp * position_error - command.kv * joint_velocity
        return desired_torque

    def validate_input(self, inputs):
        """验证完整输入，返回期望关节力矩；此函数不推进时间。"""
        if not isinstance(inputs, HandInput):
            raise TypeError('完整输入必须是HandInput')
        return self.desired_torque(inputs.state, inputs.command)

    def capacity(self):
        """返回布局规模字典，用于查看尺寸，不作为物理后端可用性的最终判断。

        packed_matrix_float32_bytes 是单个 float32 对称矩阵的三角压缩存储估计，
        不是整个求解器的显存需求。cuda_backend_ready=False 是此旧入口的保留标记，
        不表示工程没有 CUDA 步进器；实际支持范围与容量由 hand_engine.backend 检查。
        """
        return {
            'nq': self.nq,
            'nv': self.nv,
            'objects': self.object_count,
            'bodies_including_world': self.body_count,
            'max_children': self.max_children,
            'max_hinge_ancestors': self.max_hinge_ancestors,
            'packed_matrix_float32_bytes': 4 * self.nv * (self.nv + 1) // 2,
            'index_backend': 'hand_only',
            'hand_index_widths_exceeded': self.body_count > 64 or self.nv > 64,
            'cuda_backend_ready': False,
        }

    def require_cuda_backend(self):
        """保留的禁用入口，始终抛出 NotImplementedError，避免误用自动回退路径。

        实际手后端需显式通过 hand_engine.backend 检查，并使用 implicit_step 创建步进器。
        """
        raise NotImplementedError('灵巧手完整批量接触CUDA后端尚未实现；外部几何单步可显式使用hand_engine.implicit_step。'
                                  '正式入口使用hand_engine.backend；不能自动回退到教学步进器。')


def from_mujoco(model, *, palm, objects=(), joints=None):
    """从编译模型建立目标布局。只审计结构/直驱执行器，不证明所有物理选项已实现。

    输入：
        model：已经编译的 mujoco.MjModel；只读取，不修改模型。
        palm：手掌刚体名称。
        objects：物体刚体名称序列，其顺序就是内部状态中的物体顺序。
        joints：可选的手指关节名称序列；省略时使用模型刚体遍历得到的关节顺序。
    输出：HandLayout；这里只建立静态地址表，不读取某一时刻的 MjData 状态。
    后续用 layout.pack(data.qpos, data.qvel) 取得内部状态。

    palm必须是world下无关节刚体（可用mocap表示）；它的后代各含一个原点hinge。
    objects按传入顺序排列，每个是world下的独立单刚体free根。模型中的其他动态树会拒绝。
    joints可显式指定手指状态顺序，但必须完整覆盖手掌后代的hinge。
    """
    # 1. 检查整体结构，并查找规定运动的手掌。
    objects = tuple(objects)
    if len(objects) > 4 or len(set(objects)) != len(objects) or palm in objects:
        raise ValueError('物体必须是最多4个不同的自由刚体，且不能包含手掌')
    if model.ntendon or model.neq or model.nflex:
        raise ValueError('当前目标接口不支持tendon、equality或flex')
    palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, palm)
    if palm_body_id < 1 or model.body_parentid[palm_body_id] != 0 or model.body_jntnum[palm_body_id] != 0:
        raise ValueError('手掌必须是world直接子体且无动力学关节；运动由PalmMotion规定')
    # 2. 沿父体索引查找手掌后代；每个后代必须只有一个原点 hinge。
    descendants = []
    for body in range(1, model.nbody):
        parent = int(model.body_parentid[body])
        while parent not in (0, palm_body_id):
            parent = int(model.body_parentid[parent])
        if parent == palm_body_id:
            descendants.append(body)
    if not descendants:
        raise ValueError('手掌下至少需要一个直驱hinge')
    hand_joints = []
    for body in descendants:
        joint_id = int(model.body_jntadr[body])
        if model.body_jntnum[body] != 1 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError('手掌后代必须每体一个hinge，不接受焊接体、多关节、ball或slide')
        if np.any(model.jnt_pos[joint_id]) or model.qpos0[model.jnt_qposadr[joint_id]] != 0:
            raise ValueError('hinge锚点和参考角必须为零')
        hand_joints.append(joint_id)
    if joints is not None:
        ordered = []
        for name in joints:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ordered.append(joint_id)
        if len(ordered) != len(set(ordered)) or set(ordered) != set(hand_joints):
            raise ValueError('joints必须无重复且完整覆盖手掌下所有hinge')
        hand_joints = ordered
    # 3. 检查物体：每个物体是 world 下独立的单刚体 free 根。
    object_bodies = []
    for name in objects:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 1 or model.body_parentid[body] != 0 or model.body_jntnum[body] != 1:
            raise ValueError(f'{name} 必须是world下的单刚体free根')
        joint_id = int(model.body_jntadr[body])
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE or np.any(model.body_parentid[1:] == body):
            raise ValueError(f'{name} 必须只有一个free joint且没有子体')
        object_bodies.append(body)
    allowed = {0, palm_body_id}
    allowed.update(descendants)
    allowed.update(object_bodies)
    if allowed != set(range(model.nbody)):
        raise ValueError('存在未声明的刚体树；静态场景几何请挂在world上')
    # 4. 按指定关节顺序建立状态地址表和直驱执行器表。
    blocks = []
    drives = []
    for dof, joint_id in enumerate(hand_joints):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            raise ValueError('每个hinge必须命名')
        # 先找传动类型为 joint、且目标为当前关节的执行器，再要求它唯一。
        actuator_ids = np.flatnonzero((model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                             & (model.actuator_trnid[:, 0] == joint_id))
        if len(actuator_ids) != 1:
            raise ValueError(f'{name} 必须有且仅有一个直接关节执行器')
        actuator_id = int(actuator_ids[0])
        if (model.actuator_gear[actuator_id, 0] != 1 or np.any(model.actuator_gear[actuator_id, 1:])
                or model.actuator_dyntype[actuator_id] != mujoco.mjtDyn.mjDYN_NONE
                or model.actuator_gaintype[actuator_id] != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_ctrlnum[actuator_id] != 1 or model.actuator_outnum[actuator_id] != 1):
            raise ValueError(f'{name} 需要gear=1、无激活动力学的单输入直驱执行器')
        gain = model.actuator_gainprm[actuator_id]
        bias = model.actuator_biasprm[actuator_id]
        if np.any(gain[1:]) or np.any(bias[3:]):
            raise ValueError(f'{name} 存在未支持的执行器参数')
        # MuJoCo 的标准位置驱动可写成：gain*ctrl + bias[1]*q + bias[2]*v。
        # 要匹配 kp*(target-q)-kv*v，必须 gain=kp、bias[1]=-kp、bias[2]=-kv。
        if model.actuator_biastype[actuator_id] == mujoco.mjtBias.mjBIAS_NONE and gain[0] == 1:
            kind = 'torque'
            kp = 0.0
            kv = 0.0
        elif (model.actuator_biastype[actuator_id] == mujoco.mjtBias.mjBIAS_AFFINE and bias[0] == 0
              and gain[0] == -bias[1] and gain[0] >= 0 and bias[2] <= 0):
            kind = 'position_pd'
            kp = float(gain[0])
            kv = float(-bias[2])
        else:
            raise ValueError(f'{name} 执行器必须是单位力矩输入或标准位置PD')
        blocks.append(StateBlock(
            name=name,
            kind='hinge',
            body_id=int(model.jnt_bodyid[joint_id]),
            joint_id=joint_id,
            qpos_start=dof,
            qvel_start=dof,
            nq=1,
            nv=1,
            source_qpos_start=int(model.jnt_qposadr[joint_id]),
            source_qvel_start=int(model.jnt_dofadr[joint_id]),
        ))
        drives.append(DirectDrive(
            joint_name=name,
            dof=dof,
            source_actuator_id=actuator_id,
            source_control_address=int(model.actuator_ctrladr[actuator_id]),
            source_kind=kind,
            kp=kp,
            kv=kv,
        ))
    if model.nactuator != len(drives):
        raise ValueError('物体或未声明关节不能带执行器，不支持额外传动')
    # 5. 物体状态接在全部手指状态之后；位置和速度分别维护偏移。
    qpos_offset = len(hand_joints)
    qvel_offset = len(hand_joints)
    for name, body in zip(objects, object_bodies):
        joint_id = int(model.body_jntadr[body])
        blocks.append(StateBlock(
            name=name,
            kind='free_object',
            body_id=body,
            joint_id=joint_id,
            qpos_start=qpos_offset,
            qvel_start=qvel_offset,
            nq=7,
            nv=6,
            source_qpos_start=int(model.jnt_qposadr[joint_id]),
            source_qvel_start=int(model.jnt_dofadr[joint_id]),
        ))
        qpos_offset += 7
        qvel_offset += 6
    # 6. 统计树结构规模，供后端检查容量。
    children = np.bincount(model.body_parentid[1:], minlength=model.nbody)
    max_hinge_ancestors = 0
    for body in descendants:
        parent = int(model.body_parentid[body])
        count = 0
        while parent != palm_body_id:
            count += 1
            parent = int(model.body_parentid[parent])
        max_hinge_ancestors = max(max_hinge_ancestors, count)
    return HandLayout(
        palm_name=palm,
        palm_body_id=palm_body_id,
        blocks=tuple(blocks),
        actuators=tuple(drives),
        source_nq=int(model.nq),
        source_nv=int(model.nv),
        body_count=int(model.nbody),
        max_children=int(children[1:].max(initial=0)),
        max_hinge_ancestors=max_hinge_ancestors,
    )
