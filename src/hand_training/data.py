"""DexTrack闭环实验的模型/参考输入；仅服务验证，不改变原数据。

使用原Allegro运动学与原5cm方块，保留关节限位。物体按塑料密度1050计算。
实验控制采用PD目标kp20/kv1；URDF摩擦项跨模拟器含义未标定，因此本实验明确
设frictionloss=0、armature=1e-4，模型内饱和关闭，目标角在任务接口按原限位裁剪。
输出模型、HandLayout、600帧参考；这些实验参数不声称复现原PhysX或硬件。
固定掌心朝上的实验：掌面外法向为+X，重力明确取-X，不能沿用XML默认-Z。
这是选定的托持任务朝向，并非还原原始运动手掌坐标系下随时间变化的重力。
"""

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import mujoco
from nablaengine.model.hand import from_mujoco


def load_experiment_model(output_directory, *, gravity=(-9.81, 0.0, 0.0), physics_dt=0.0166 / 4):
    if not np.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("物理子步必须是有限正数，单位为秒")
    model = mujoco.MjModel.from_xml_path(
        str(
            (ROOT / "assets/allegro_plastic_cube.xml")
        )
    )
    # 先设置重力再创建layout/引擎：CUDA动力学在构造时会复制模型参数。
    # Allegro掌面在局部YZ平面、物体位于+X侧，-X才是落向手掌。
    model.opt.gravity[:] = gravity
    model.opt.timestep = float(physics_dt)
    model.opt.iterations = 200
    model.opt.tolerance = 1e-12
    model.actuator_ctrllimited[:] = 0
    model.actuator_forcelimited[:] = 0
    model.jnt_actfrclimited[:] = 0
    model.dof_frictionloss[:16] = 0.0
    model.dof_armature[:16] = 0.0001
    model.actuator_gainprm[:, 0] = 20.0
    model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_biasprm[:, :3] = [0.0, -20.0, -1.0]
    layout = from_mujoco(
        model, palm="palm_link", objects=["object0"], joints=[f"joint_{i}" for i in range(16)]
    )
    # mj_saveLastXML跟踪原XML，不能保存这些数组修改；完整参数另存NPZ。
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        Path(output_directory) / "physical_parameters.npz",
        timestep=model.opt.timestep,
        gravity=model.opt.gravity,
        palm_outward_normal=np.array([1.0, 0.0, 0.0]),
        damping=model.dof_damping,
        frictionloss=model.dof_frictionloss,
        armature=model.dof_armature,
        joint_range=model.jnt_range,
        gain=model.actuator_gainprm,
        bias=model.actuator_biasprm,
        body_mass=model.body_mass,
        body_inertia=model.body_inertia,
    )
    positions = []
    velocities = []
    for tag in ("inspect_1", "lift"):
        with np.load(
            ROOT / f"datasets/dextrack/{tag}_bounded.npz", allow_pickle=False
        ) as candidate:
            q = candidate["qpos"].copy()
            hand_velocity = candidate["hand_qvel"].copy()
        with np.load(
            ROOT / f"datasets/dextrack/{tag}_source.npz", allow_pickle=False
        ) as original:
            original_positions = original["qpos"]
            object_velocity = original["qvel"][:, 16:].copy()
            assert np.array_equal(q[:, 16:], original_positions[:, 16:])
        positions.append(q)
        velocities.append(np.c_[hand_velocity, object_velocity])
    return model, layout, np.concatenate(positions), np.concatenate(velocities)
