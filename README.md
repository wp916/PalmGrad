# PalmGrad

**面向掌内操作的可微动力学与强化学习工程。**

PalmGrad 将手指、被操作物体、接触力和摩擦扭矩放进同一个动力学系统，计算下一时刻的状态，并将梯度沿物理步进传回控制策略。

**当前状态：训练、物理前向、多步反传、策略更新、评估和断点续训已接通；现有策略尚未稳定完成长时操作。** 底层状态接口允许最多 4 个自由物体，随附训练任务仍是 Allegro 手与单个方块。

## 对 SuperMimic 世界模型的改进

[SuperMimic](https://github.com/HybridRobotics/supermimic) 使用学习得到的动力学模型预测状态，并通过短时展开对策略求导。模型需要根据采集到的状态转移数据持续训练，其预测误差也会影响策略获得的梯度。

PalmGrad 采用 **显式物理动力学承担世界模型中的状态转移功能**。

| 环节 | 学习动力学的路线 | PalmGrad 当前路线 |
| --- | --- | --- |
| 下一状态 | 神经网络 `s_next = f_phi(s, a)` | 物理步进 `s_next = F_phys(s, a; p)` |
| 动力学来源 | 从交互数据拟合模型参数 | 质量、惯量、关节、接触与摩擦等物理参数 |
| 策略梯度 | 经过学习模型反向传播 | 经过 PD 控制、约束求解和积分反向传播 |
| 训练组成 | 需要额外训练状态转移模型 | 当前不训练状态转移网络，仍训练 actor 和 critic |

对手操作而言，本工程将这一训练路线接到手专用动力学上：手掌作为规定运动的基座，手指为独立直驱关节，被操作物体作为自由刚体；补充状态排列、动作映射、接触及扭转摩擦、导数接口、参考跟踪任务和数据加载。

这样可以避免学习状态转移模型产生的那部分近似误差，也便于直接检查重力、关节限位、接触力等物理量。接触切换的不光滑、物理参数偏差和数值误差仍然存在。当前没有足够实验支持训练速度或成功率优于 SuperMimic 的结论。

## 整个系统怎样运行

```text
参考动作 + 当前仿真状态
          ↓
actor 输出手指动作残差
          ↓
参考关节角 + 残差 → 关节目标角 → PD 控制力矩
          ↓
动力学：质量与偏置力 → 接触/摩擦/限位约束 → 求解 → 积分
          ↓
下一状态 → 跟踪奖励、终止判定、critic
          ↓
沿短时轨迹反向传播 → 直接策略梯度或 PODS 更新
```

`train.py` 中的 `run_round()` 每轮完成状态采集、从状态池选择反传初态、actor/PODS 更新、critic 更新和指标记录。物体仅在 reset 时赋初值，随后由动力学推进，不会在每步被强行写成参考位置。

PODS 将策略梯度形成的动作目标用于回归更新策略；不加 `--pods` 时使用直接策略梯度。更新函数集中在 `updates.py`，便于与 NablaMimic 对照阅读。

## 动力学输入与输出

### 模型范围与状态排列

设手指共有 `n` 个直驱转动关节，被操作物体数量为 `k`，其中 **`0 <= k <= 4`**。每个物体是一个独立自由刚体。手掌位姿单独输入，不占用动态状态向量前几项。

```text
qpos = [n 个手指关节角,
        物体0的位置 xyz、姿态四元数 wxyz,
        物体1的位置 xyz、姿态四元数 wxyz, ...]

qvel = [n 个手指关节角速度,
        物体0的线速度 xyz、角速度 xyz,
        物体1的线速度 xyz、角速度 xyz, ...]

nq = n + 7*k
nv = n + 6*k
```

物体位置、线速度使用世界坐标系，角速度使用物体局部坐标系。四元数顺序是 **`w, x, y, z`**。关节角和角速度的单位为 rad、rad/s，长度单位为 m。

例如，16 关节手加 1 个物体时，`nq=23, nv=22`；加 4 个物体时，`nq=44, nv=40`。物体运动数据统一排在手指之后。`HandLayout.pack()` / `unpack()` 根据模型地址转换源模型排列与上述内部排列，不要求刚体编号等于自由度编号。

### 单环境接口

接口定义见 [hand.py](src/nablaengine/model/hand.py)。`HandInput` 将三部分组织在一起：

| 数据 | 含义 |
| --- | --- |
| `HandState(qpos, qvel)` | 手指和全部自由物体的当前状态 |
| `PalmMotion` | 手掌的位置、四元数、世界线速度/角速度、世界线加速度/角加速度 |
| `DirectDriveCommand` | 长度为 `n` 的关节力矩，或关节位置目标及 PD 增益 |

固定手掌通过规定恒定位姿、速度和加速度均为零实现。底层保留规定运动基座输入；当前训练桥接在创建时固定手掌参数，没有将可变手掌轨迹作为策略动作或可微输入。

位置 PD 模式的基本力矩为 `kp * (target - q) - kv * qvel`。动作只控制手指关节，物体没有直接驱动器。

### 训练使用的批量接口

[jax_adapter.py](src/nablaengine/hand_engine/jax_adapter.py) 提供：

```python
next_qpos, next_qvel, next_warm = physics.step(
    qpos, qvel, warm, ctrl, friction, com
)
```

| 输入 | 形状 | 作用 |
| --- | --- | --- |
| `qpos` | `(B, n+7*k)` | 当前关节角与物体位姿 |
| `qvel` | `(B, n+6*k)` | 当前速度 |
| `warm` | `(B, n+6*k)` | 求解器加速度热启动值 |
| `ctrl` | `(B, n)` | 关节位置目标或力矩，模式在创建接口时确定 |
| `friction` | `(B,)` | 兼容训练调用的参数，当前只接受全 1 |
| `com` | `(B, 3)` | 兼容训练调用的参数，当前只接受全 0 |

`B` 是环境数量，数组使用 float64。三个输出形状分别与 `qpos`、`qvel`、`warm` 相同；`step()` 可配置多个物理子步。

接口通过 JAX `custom_vjp` 返回状态和控制输入的向量雅可比积，使训练能够反传到动作及前序状态；默认不显式构造整个雅可比矩阵。`warm`、`friction`、`com` 当前不参与可微优化。

**最多 4 个物体是底层模型与动力学接口的范围。** 随附任务的观测、奖励、reset 和参考数据按单物体设计，多物体任务仍需要扩展任务层代码并单独验证性能。

## 动力学实现与当前边界

- 采用质量矩阵、偏置力、软接触约束、Newton 求解和 `implicitfast` 积分。
- 支持当前手任务所需的法向接触、滑动摩擦、关节限位及 `condim=4` 扭转摩擦。扭转摩擦限制绕接触法向的相对转动，保留独立参数设置；当前不使用 `noslip` 后处理。
- MuJoCo 提供模型解析、碰撞和相关几何量；动力学约束求解与积分由本工程复用、适配的 CUDA 代码完成。几何导数仍含数值差分，不能理解为全部碰撞过程均为解析求导。
- 当前由 **CPU JAX 逐环境调度 CUDA 动力学**，尚非整个训练过程都在 GPU 上并行。批量增大不等于已有融合的 GPU 批量求解。
- 主要模型范围是直驱转动关节、规定手掌和独立自由物体。当前保留手专用求解结构，尚未迁移到新版官方 forest/网格实现。

## 数据与默认任务

随附 DexTrack 的两条处理后参考，各 300 帧：`inspect_1` 用于训练，`lift` 留作评估。它们是小型验证子集，不是完整数据集。

`*_bounded.npz` 是按当前关节限位和接触要求处理的训练参考；`*_source.npz` 保留前一阶段参考。两段轨迹边界独立，不将第一段末帧与第二段首帧拼成连续动作。来源记录中的历史绝对路径仅用于追溯，运行不读取这些路径。

每个控制步包含 4 个 0.00415 s 的物理子步，控制间隔为 0.0166 s。该间隔是当前任务配置，未核验为原始采集时间戳。动作是在下一参考关节角上叠加 `0.1 * action` 的残差，再按关节限位裁剪；PD 增益为 `kp=20, kv=1`，当前没有力矩饱和和目标角变化限速。

## 安装与启动

在工程目录执行：

```bash
conda create -n palmgrad python=3.12
conda activate palmgrad
pip install -r requirements.txt

bash scripts/train.sh --pods --rounds 20 --num-envs 2 \
    --output-dir outputs/pods
```

本机也可以沿用已经配置好的 `nablamimic` 环境。启动脚本会设置源码路径、CPU JAX 和 float64，不需要安装原 NablaEngine / NablaMimic 项目。

先运行一轮检查完整训练路径：

```bash
bash scripts/train.sh --pods --rounds 1 --output-dir outputs/first_run
```

不加 `--pods` 即运行直接策略梯度，两个实验应使用不同输出目录。脚本默认反传窗口为 8 步、训练批量为 2；`--num-envs` 允许 2、4、8，其中 8 尚未完成实际训练验证。评估使用固定的两个环境。

继续同一实验时，`--rounds` 表示续训后的总轮数：

```bash
bash scripts/train.sh --pods --rounds 40 --num-envs 2 \
    --output-dir outputs/pods --resume
```

续训会核对算法、数据、模型、源码和参数指纹；源码注释变化也会影响指纹。除增加总轮数外，应保持原配置。新实验不会直接覆盖已有实验目录。

## 输出与验证

输出目录包含 `history.json`、`results.json`、评估轨迹 NPZ、每轮原子保存的 `checkpoint_latest.pkl` 和最终 checkpoint。训练状态包含网络、优化器、状态池、随机数以及 PODS 缓存，供中断后继续运行。

评估记录位姿误差、失败时刻、末帧成功率和持续成功率。失败后的冻结尾帧由 `valid_step` 标记，统计时不计入有效轨迹；持续成功要求整段控制步都达标。每 10 轮及训练结束还会运行连续段评估。一轮训练通过只说明计算链路可执行，不表示学会了操作任务。

运行回归测试：

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
    python -m unittest discover -s tests -v
```

工程保留正常训练、评估、数值校验和回归测试；独立诊断脚本、历史实验报告、教学脚本和训练输出不作为源码发行内容。

## 许可与致谢

代码采用 Apache-2.0，详见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和 [第三方声明](THIRD_PARTY_NOTICES.md)。NablaMimic 的许可证副本位于 [docs/NablaMimic_LICENSE](docs/NablaMimic_LICENSE)。模型和参考数据的来源条款应分别遵守。

感谢 [SuperMimic](https://github.com/HybridRobotics/supermimic) 与 [NablaMimic](https://github.com/sailab-ai/NablaMimic) 提供的研究思路和开源代码。

特别感谢 **[NablaEngine](https://github.com/sailab-ai/NablaEngine) 团队的技术开源**，使本工程能够在已有可微多体动力学、接触求解和反向传播实现上，继续探索灵巧手掌内操作。
