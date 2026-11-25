"""JAX 调用手动力学的入口：连接训练代码与已有 CUDA 前向、反向计算。

用途与调用链
    训练中的物理 step → 本文件 JaxHandPhysics.step → 主机回调 _forward
    → _run_world → engine.step → CUDA 动力学求解与状态积分。
    损失反传时走 _backward：重算前向、记录各子步，再逆序调用 adjoint.vjp。
    本文件负责接口与调度；力学公式、接触求解和积分在其他物理模块中。

主要接口
    physics = JaxHandPhysics(engine, nworld, ..., substeps=2)
    next_qpos, next_qvel, next_warm = physics.step(qpos, qvel, warm, ctrl, friction, com)
    engine 是已创建的 CudaImplicitFastStepper；nworld 是环境数，不是刚体数。
    每次 step 执行 substeps 个物理小步，同一 ctrl 在这些小步中保持不变。

输入输出（B=nworld，n=手指关节数，k=自由物体数）
    qpos:     (B, nq)，nq=n+7*k；手指角度 rad，物体世界位置 m、姿态 wxyz。
    qvel:     (B, nv)，nv=n+6*k；手指角速度、物体世界线速度/物体局部角速度。
    warm:     (B, nv)，求解加速度的初始猜测，用于提高收敛速度。
    ctrl:     (B, n)，torque 模式为关节力矩 N·m，position_pd 模式为目标角 rad。
    friction: (B,)，目前只接受全1；com: (B, 3)，目前只接受全0。
    所有输入为 float64；返回的位置、速度、暖启动与输入对应形状相同。
    状态和控制采用 hand 内部顺序；源模型顺序由 mimic_interface.py 在外层转换。

求导与使用边界
    向 JAX 返回位置、速度、控制的梯度；warm/friction/com 的输入梯度约定为0。
    下一步 warm 的输出梯度也忽略。规定手掌、材料和 PD 增益保持固定。
    grad_decay 只缩放传回上一步的位置/速度梯度，不缩放控制梯度，也不改变前向。
    这是 CPU JAX 调度、主机搬运、逐环境执行 CUDA 的正确性桥，不是融合批量 GPU FFI。
    不负责关闭 engine；调用方必须等异步 JAX 计算完成后再释放引擎资源。
"""

import numpy as np
from nablaengine.model.hand import HandState, HandInput, DirectDriveCommand, PalmMotion
from .state_adjoint import CudaContactStateAdjoint


class JaxHandPhysics:
    """把六输入、三输出的物理步进注册为 JAX 可求导函数。

    公开 step 是在构造时通过 custom_vjp 注册并绑定的函数，定义位于 __init__ 中。
    engine/layout/nworld 等已有属性名保留，便于现有任务和验证脚本继续使用。
    """

    def __init__(
        self,
        engine,
        nworld,
        *,
        palm=None,
        command_mode="torque",
        kp=None,
        kv=None,
        substeps=1,
        grad_decay=1.0,
        geometry=None,
    ):
        """检查后端及固定参数，建立数组规格，并注册前向/反向接口。

        palm 是每次调用使用的规定手掌状态，省略时为固定原点手掌。
        kp/kv 是内部手指顺序的 PD 增益数组；力矩模式不接受增益。
        geometry省略时沿用默认CUDA解析/AD几何导数；显式传入实验对象时使用其JVP，
        导数精度和碰撞实现应在实验配置中另行记录，不能据此宣称纯GPU后端。
        substeps 决定一次控制覆盖几个物理小步；控制周期为 engine.h * substeps。
        """
        import jax
        import jax.numpy as jnp

        if jax.default_backend() != "cpu":
            raise ValueError(
                "主机正确性桥要求JAX_PLATFORMS=cpu；动力学仍在CUDA执行，GPU JAX FFI另行接入"
            )
        if not jax.config.x64_enabled:
            raise ValueError("mixed64正确性桥要求显式启用jax_enable_x64")
        if isinstance(nworld, bool) or not isinstance(nworld, (int, np.integer)) or nworld < 1:
            raise ValueError("nworld必须为正整数")
        if (
            isinstance(substeps, bool)
            or not isinstance(substeps, (int, np.integer))
            or substeps < 1
        ):
            raise ValueError("substeps必须为正整数")
        if not np.isfinite(grad_decay):
            raise ValueError("grad_decay必须有限")

        self.engine = engine
        self.layout = engine.layout
        self.nworld = int(nworld)
        self.substeps = int(substeps)
        self.palm = PalmMotion.fixed() if palm is None else palm
        if not isinstance(self.palm, PalmMotion):
            raise TypeError("需要显式PalmMotion")
        self.mode = command_mode
        self.kp = None if kp is None else np.array(kp, dtype=np.float64, copy=True)
        self.kv = None if kv is None else np.array(kv, dtype=np.float64, copy=True)
        # 用零命令提前检查控制模式、增益形状和增益合法性，不执行仿真。
        self._command(np.zeros(self.layout.hand_dofs))
        for gain_array in (self.kp, self.kv):
            if gain_array is not None:
                gain_array.setflags(write=False)
        self.adjoint = CudaContactStateAdjoint(engine, geometry=geometry)
        self.verified_cost_stop_calls = 0

        batch_size = self.nworld
        hand_layout = self.layout
        # 顺序固定对应 qpos、qvel、warm、ctrl、friction、com。
        self.shapes = (
            (batch_size, hand_layout.nq),
            (batch_size, hand_layout.nv),
            (batch_size, hand_layout.nv),
            (batch_size, hand_layout.hand_dofs),
            (batch_size,),
            (batch_size, 3),
        )
        # 回调在实际运行前，JAX 就需要知道返回数组的形状与类型。
        forward_output_specs = tuple(
            jax.ShapeDtypeStruct(shape, np.float64) for shape in self.shapes[:3]
        )
        gradient_output_specs = tuple(
            jax.ShapeDtypeStruct(self.shapes[input_index], np.float64) for input_index in (0, 1, 3)
        )

        def check_jax_inputs(physics_inputs):
            """先检查框架能看到的形状/类型；数值有限性在主机回调中检查。"""
            if len(physics_inputs) != 6:
                raise ValueError("需要六个输入")
            for input_array, expected_shape in zip(physics_inputs, self.shapes):
                if input_array.shape != expected_shape or input_array.dtype != jnp.float64:
                    raise ValueError(f"输入需要float64形状{expected_shape}")

        def call_forward_callback(*physics_inputs):
            check_jax_inputs(physics_inputs)
            return jax.pure_callback(
                self._forward, forward_output_specs, *physics_inputs, vmap_method="sequential"
            )

        @jax.custom_vjp
        def step(*physics_inputs):
            # 普通前向从这里进入；反向规则由下方 defvjp 注册。
            return call_forward_callback(*physics_inputs)

        def forward_for_autodiff(*physics_inputs):
            # 第二项为反向保留的纯数组输入，不缓存 Python tape ID。
            # 这样 scan/checkpoint 重复执行时可以从相同输入重新计算物理轨迹。
            return call_forward_callback(*physics_inputs), physics_inputs

        def backward_for_autodiff(physics_inputs, output_cotangents):
            # cotangent（余切）可理解为损失对输出的梯度。
            # 输出有 qpos/qvel/warm 三项，这里只使用前两项梯度。
            qpos_gradient, qvel_gradient, control_gradient = jax.pure_callback(
                self._backward,
                gradient_output_specs,
                *physics_inputs,
                output_cotangents[0],
                output_cotangents[1],
                vmap_method="sequential",
            )
            # 必须按六个前向输入的原顺序返回梯度。
            return (
                qpos_gradient * grad_decay,
                qvel_gradient * grad_decay,
                jnp.zeros_like(physics_inputs[2]),
                control_gradient,
                jnp.zeros_like(physics_inputs[4]),
                jnp.zeros_like(physics_inputs[5]),
            )

        step.defvjp(forward_for_autodiff, backward_for_autodiff)
        self.step = step
        self.info = dict(
            backend="hand_jax_host_correctness_bridge",
            dtype="float64",
            state_order="hand_canonical",
            substeps=self.substeps,
            control_dt=engine.h * self.substeps,
            parallel_batch=False,
            kinematics=False,
            fixed_context=True,
            grad_decay=float(grad_decay),
        )

    def _command(self, control):
        """把一环境的控制数组包装为直驱命令；此处不计算动力学。"""
        return DirectDriveCommand(self.mode, control, self.kp, self.kv)

    def _arrays(self, physics_inputs):
        """主机侧转为 NumPy 并检查数值；返回六个输入，顺序不变。"""
        input_arrays = tuple(np.asarray(value, dtype=np.float64) for value in physics_inputs)
        for input_array, expected_shape in zip(input_arrays, self.shapes):
            if input_array.shape != expected_shape or not np.isfinite(input_array).all():
                raise ValueError("输入形状不符或包含非有限值")
        friction_scale = input_arrays[4]
        center_of_mass_offset = input_arrays[5]
        if not np.all(friction_scale == 1.0) or not np.all(center_of_mass_offset == 0.0):
            raise ValueError("当前hand桥固定材料与质心，仅接受friction=1和com=0；未接入随机化")
        return input_arrays

    @staticmethod
    def _converged(step_result):
        """要求 Newton 达到指定收敛精度，避免返回未经验证的隐式梯度。"""
        solved = step_result.newton
        residual_ok = np.isfinite(solved.scaled_gradient) and solved.scaled_gradient < 1e-10
        cost_stop_ok = False
        if solved.reason == "cost_improvement" and residual_ok:
            # 目标函数接近浮点舍入水平时，可能先触发代价停止。
            # 不降低残差门槛；额外核对当前Hessian对应的剩余加速度修正。
            try:
                correction = np.linalg.solve(solved.evaluation.hessian, solved.evaluation.gradient)
                cost_stop_ok = bool(
                    np.isfinite(correction).all() and np.max(abs(correction)) < 1e-8
                )
            except np.linalg.LinAlgError:
                pass
        if not residual_ok or (solved.reason != "gradient" and not cost_stop_ok):
            raise RuntimeError(
                "正确性桥要求每个子步充分收敛，不输出未经验证的隐式梯度；"
                f"reason={step_result.newton.reason}, "
                f"scaled_gradient={step_result.newton.scaled_gradient:.17g}"
            )
        return cost_stop_ok

    def _run_world(self, qpos, qvel, warm_start, control, record):
        """单环境执行一个控制周期，返回末态、暖启动及可选的子步记录。

        record=False 直接执行 engine.step；True 用 adjoint.record 执行并记录前向。
        control 在全部子步保持不变，暖启动则由上一个子步传给下一个子步。
        """
        current_state = HandState(qpos, qvel)
        substep_tapes = []
        for substep_index in range(self.substeps):
            # 沿用已有几何评估/边界检查。零向量用于此阶段评估，不是把系统速度设为零。
            self.adjoint.geometry.evaluate(current_state, self.palm, np.zeros(self.layout.nv))
            physics_input = HandInput(current_state, self.palm, self._command(control))
            if record:
                substep_tape = self.adjoint.record(
                    physics_input, warm_start=warm_start, time=substep_index * self.engine.h
                )
                step_result = substep_tape.forward
                substep_tapes.append(substep_tape)
            else:
                # 真正进入动力学步进器：接触/动力学/约束求解/积分在下层完成。
                step_result = self.engine.step(
                    physics_input, warm_start=warm_start, time=substep_index * self.engine.h
                )
            try:
                if self._converged(step_result):
                    self.verified_cost_stop_calls += 1
            except RuntimeError:
                # 只补失败现场，不修改收敛门槛、不替换求解结果。
                diagnostics_dir = getattr(self.adjoint.geometry, "diagnostics_dir", None)
                if diagnostics_dir is not None:
                    from pathlib import Path
                    import json

                    path = Path(diagnostics_dir)
                    path.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        path / "newton_failure.npz",
                        qpos=current_state.qpos,
                        qvel=current_state.qvel,
                        warm_start=warm_start,
                        control=control,
                        substep=substep_index,
                    )
                    (path / "newton_failure.json").write_text(
                        json.dumps(
                            dict(
                                reason=step_result.newton.reason,
                                scaled_gradient=step_result.newton.scaled_gradient,
                                history=step_result.newton.trace,
                            ),
                            indent=2,
                        )
                    )
                raise
            current_state = step_result.state
            warm_start = step_result.warm_start
        return current_state, warm_start, substep_tapes

    def _forward(self, *physics_inputs):
        """JAX 的前向主机回调：逐环境运行 CUDA，返回三个批量 NumPy 数组。"""
        qpos, qvel, warm_start, control, _, _ = self._arrays(physics_inputs)
        next_qpos = np.empty_like(qpos)
        next_qvel = np.empty_like(qvel)
        next_warm_start = np.empty_like(warm_start)
        # 引擎资源共用，锁内串行使用；批量维存在不代表此处并行运行环境。
        with self.engine._lock:
            for environment_index in range(self.nworld):
                next_state, next_warm, _ = self._run_world(
                    qpos[environment_index],
                    qvel[environment_index],
                    warm_start[environment_index],
                    control[environment_index],
                    False,
                )
                next_qpos[environment_index] = next_state.qpos
                next_qvel[environment_index] = next_state.qvel
                next_warm_start[environment_index] = next_warm
        return next_qpos, next_qvel, next_warm_start

    def _backward(self, *callback_inputs):
        """输入六个原始物理数组及两个输出梯度，返回步初位置/速度/控制梯度。

        每个环境先重算并记录子步，再从最后一个子步向前传递梯度。
        这里只反传一个控制周期；多个控制周期由外层 JAX 的 scan 链式连接。
        """
        qpos, qvel, warm_start, control, _, _ = self._arrays(callback_inputs[:6])
        next_qpos_gradient, next_qvel_gradient = callback_inputs[6:]
        if (
            next_qpos_gradient.shape != qpos.shape
            or next_qvel_gradient.shape != qvel.shape
            or not np.isfinite(next_qpos_gradient).all()
            or not np.isfinite(next_qvel_gradient).all()
        ):
            raise ValueError("非法输出余切")
        initial_qpos_gradient = np.empty_like(qpos)
        initial_qvel_gradient = np.empty_like(qvel)
        control_gradient = np.zeros_like(control)
        with self.engine._lock:
            for environment_index in range(self.nworld):
                _, _, substep_tapes = self._run_world(
                    qpos[environment_index],
                    qvel[environment_index],
                    warm_start[environment_index],
                    control[environment_index],
                    True,
                )
                propagated_qpos_gradient = next_qpos_gradient[environment_index]
                propagated_qvel_gradient = next_qvel_gradient[environment_index]
                for substep_tape in reversed(substep_tapes):
                    substep_gradient = self.adjoint.vjp(
                        substep_tape, propagated_qpos_gradient, propagated_qvel_gradient
                    )
                    # 所有子步共用同一个控制，必须把它们对该控制的梯度相加。
                    control_gradient[environment_index] += substep_gradient.action
                    # 将旋转扰动的梯度转换成 qpos 数组梯度，再交给前一个子步。
                    propagated_qpos_gradient = substep_gradient.qpos_cotangent(
                        substep_tape.inputs.state, self.layout
                    )
                    propagated_qvel_gradient = substep_gradient.qvel
                initial_qpos_gradient[environment_index] = propagated_qpos_gradient
                initial_qvel_gradient[environment_index] = propagated_qvel_gradient
        return initial_qpos_gradient, initial_qvel_gradient, control_gradient
