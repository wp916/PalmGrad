"""灵巧手模仿训练主入口：读取参考动作，实际训练策略并评估物体跟踪。

运行：bash scripts/train.sh --pods --rounds 80（工程根目录）。
输入：assets中的Allegro/方块模型、datasets/dextrack两条参考、命令行训练参数。
输出：output-dir中的训练日志、评估轨迹、完整checkpoint和results.json。

main：建立模型、可微物理、任务环境、网络、优化器、重置分布和状态池。
run_round（main内）：采集两步 → 从状态池取反传初态 → 更新actor/PODS → 更新critic。
evaluate（main内）：固定初态运行物理、计算误差；不更新网络。
每轮原子保存完整训练状态，供中断后继续训练。

数据链：策略动作 → reference_tracking任务 → jax_adapter前向/VJP → implicit_step CUDA。
训练inspect_1，lift仅用于留出评估。物体只在reset赋初值，后续由动力学积分。
本入口是验证规模训练，不代表已经掌握完整操作；目前尚未集成目标变化限速。
"""

from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from .data import load_experiment_model
from .evaluation import evaluate_policy
from .checkpoint import run_signature, validate_run_directory, save_latest, load_latest
from .tree_utils import (
    add_device_axis,
    remove_device_axis,
    assert_trees_equal,
    all_arrays_finite,
)
from nablaengine.hand_engine.mujoco_collision_bridge import (
    MujocoCollisionBridge,
    FiniteDifferenceCollisionJVP,
)
from nablaengine.hand_engine.implicit_step import CudaImplicitFastStepper
from nablaengine.hand_engine.jax_adapter import JaxHandPhysics
from nablaengine.hand_engine.mimic_interface import HandActionMap
from nablaengine.hand_engine.reference_tracking import HandReferenceTracking


def main():
    # 1. 读取启动参数。args只是一个保存命令行选项的对象，例如args.pods表示是否启用PODS。
    # rounds：更新网络多少轮；horizon：每次反向传播向未来展开多少个控制步。
    # 二者都不是数据集帧数。脚本train.sh把horizon设为8，直接运行本模块默认是4。
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument(
        "--num-envs",
        type=int,
        choices=(2, 4, 8),
        default=2,
        help="训练环境数量；评估固定为两个环境，避免改变对照样本",
    )
    parser.add_argument("--output-dir", default="outputs/training")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--feasible-only", action="store_true")
    parser.add_argument("--velocity-feedforward", action="store_true")
    parser.add_argument("--pods", action="store_true")
    parser.add_argument("--feasibility-report", help="当前重力下的承重必要条件报告；不用旧方向区间")
    args = parser.parse_args()
    if args.rounds <= 0 or args.horizon <= 0:
        parser.error("rounds和horizon必须为正整数")
    if not jax.config.x64_enabled:
        parser.error("当前物理接口要求JAX_ENABLE_X64=true，请使用scripts/train.sh")
    output = Path(args.output_dir)
    signature = run_signature(args, Path(__file__).resolve().parents[2])
    validate_run_directory(output, signature, resume=args.resume)
    resumed_state = None
    resumed_history = []
    if args.resume:
        resumed_state, resumed_history = load_latest(output / "checkpoint_latest.pkl")
        if args.rounds <= len(resumed_history):
            parser.error("rounds应大于checkpoint已完成轮数；rounds表示续训后的总轮数")
    from . import updates as train, sim, nets

    # 保存算法文件的“指纹”，训练结束再核对，防止运行期间源码被改动。
    algorithm_source_hashes = {
        module.__file__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (train, sim, nets)
    }
    # 2. 模型是质量、惯量、关节、几何等常量；layout说明每个物理量在数组中的位置。
    # 两条参考拼成600帧：inspect_1为0..299，lift为300..599。
    # reference_qpos形状(600,23)：16个关节角 + 物体xyz + 物体四元数wxyz。
    # reference_qvel形状(600,22)：16个关节角速度 + 物体6维速度。
    model, layout, reference_qpos, reference_qvel = load_experiment_model(output)
    # 每个训练环境都是一只手和一个物体；环境数不等于物体数或GPU数量。
    num_envs = args.num_envs
    start_time = time.monotonic()
    history = []
    # 3. 建立真实物理计算。with退出时释放底层资源，可理解为C++的作用域资源管理。
    # provider只提供碰撞几何；接触力、动力学求解和积分由Nabla CUDA执行。
    # geometry负责碰撞几何变化的数值导数；引擎其余导数由对应反向代码计算。
    with MujocoCollisionBridge(model, layout) as provider:
        geometry = FiniteDifferenceCollisionJVP(provider, diagnostics_dir=output)
        with CudaImplicitFastStepper(model, layout, provider, numerics="mixed64") as engine:
            # 把CUDA求解器包装成JAX可以求导的接口。
            # 一次策略动作保持4个物理子步：4×0.00415s = 0.0166s。
            # kp/kv分别是位置误差/速度误差的PD增益，这里的16项对应16个直驱关节。
            physics = JaxHandPhysics(
                engine,
                num_envs,
                command_mode="position_pd",
                kp=np.full(16, 20.0),
                kv=np.ones(16),
                substeps=4,
                geometry=geometry,
            )
            # 动作映射给出16维动作的尺度和关节目标上下限。
            # 当前reference_tracking使用“下一参考帧关节角 + 0.1×网络动作”作为PD目标，
            # 再按原关节限位裁剪。网络动作不是物体位移，也不是直接输出电机力矩。
            action_map = HandActionMap(
                reference_qpos[0, :16],
                np.full(16, 0.1),
                model.jnt_range[:16, 0],
                model.jnt_range[:16, 1],
            )
            # 任务层负责reset、观测、奖励、终止和动作转换。
            # reference_last为每一帧记录所属轨迹的最后一帧，避免从inspect跳到lift。
            # 参考给出“希望怎样运动”；物体下一状态仍由实际动力学算出。
            env = HandReferenceTracking(
                physics,
                action_map,
                reference_qpos=reference_qpos,
                reference_qvel=reference_qvel,
                reference_last=np.r_[np.full(300, 299), np.full(300, 599)],
                control_dt=0.0166,
                velocity_feedforward=args.velocity_feedforward,
            )
            # 评估始终使用原来的两个环境。增加训练批量时不复制评估样本，
            # 也不改动其起点、时长或成功标准。训练/评估共享模型和CUDA引擎，顺序调用。
            evaluation_env = env
            if num_envs != 2:
                evaluation_physics = JaxHandPhysics(
                    engine,
                    2,
                    command_mode="position_pd",
                    kp=np.full(16, 20.0),
                    kv=np.ones(16),
                    substeps=4,
                    geometry=geometry,
                )
                evaluation_env = HandReferenceTracking(
                    evaluation_physics,
                    action_map,
                    reference_qpos=reference_qpos,
                    reference_qvel=reference_qvel,
                    reference_last=np.r_[np.full(300, 299), np.full(300, 599)],
                    control_dt=0.0166,
                    velocity_feedforward=args.velocity_feedforward,
                )
            # 4. 算法配置。SimpleNamespace像一个简单结构体，字段在这里明确列出。
            # collect_noise：采集时给动作加噪声；max_episode_steps：训练episode最长16步。
            # gamma：未来奖励折扣；lam：critic的TD(lambda)目标混合系数。
            # critic_pre_state=True：用转移开始时的状态拟合该状态对应的回报。
            # action_target=(8,1,0.03)：PODS回归8次、缓存1个窗口、目标改进RMS约0.03rad。
            # act_cot_clip裁剪反向的动作梯度；它不限制相邻时刻手指张开的速度。
            config = SimpleNamespace(
                collect_noise=0.01,
                max_episode_steps=16,
                horizon=args.horizon,
                act_cot_clip=3.0 if args.pods else None,
                shac_strict=False,
                critic_pre_state=True,
                critic_minibatches=1,
                critic_epochs=5,
                gamma=0.99,
                lam=0.95,
                action_target=(8, 1, 0.03) if args.pods else None,
                action_target_delta_floor=1.0,
                lr_decay_steps=0,
            )
            # 优化器对象定义“怎样改参数”，init产生的optimizer_state保存Adam历史记忆。
            # actor直接梯度路径、critic路径和PODS回归路径各用自己的优化器。
            # 梯度裁剪限制一次参数更新的梯度幅度，不是物理力矩饱和。
            opt_actor = optax.chain(
                optax.clip_by_global_norm(1.0), optax.adam(3e-4, b1=0.9, b2=0.95)
            )
            opt_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4))
            opt_regression = optax.chain(
                optax.clip_by_global_norm(1.0), optax.clip(0.1), optax.adam(6e-4, b1=0.9, b2=0.95)
            )
            # rollout把“网络→动作→物理步→奖励”连续展开horizon次。
            # 窗口末尾由critic估计更远的回报（bootstrap），再沿展开轨迹反向求导。
            # remat=True表示反向时可重新计算部分中间量，节省保存中间结果的内存。
            # mask_after_done使episode结束后窗口剩余部分不再贡献该段的策略目标。
            rollout = train.make_rollout(
                env,
                args.horizon,
                0.99,
                16,
                True,
                mask_after_done=True,
                record_pre=True,
                record_obs=args.pods,
                act_clip=config.act_cot_clip,
                act_noise=0.05 if args.pods else 0.0,
            )
            # 将环境、优化器等显式交给算法模块；返回一个“函数名→可调用函数”的字典。
            # 对照NablaMimic时，看updates.py中的同名函数即可。
            functions = train.build_training_functions(
                env=env,
                args=config,
                n=num_envs,
                nc=num_envs,
                use_replay=True,
                use_at=args.pods,
                opt_r=opt_regression,
                opt_a=opt_actor,
                opt_c=opt_critic,
                rollout=rollout,
            )
            # pmap增加设备维，并提供算法中名为d的设备间统计轴。
            # 当前只有一个CPU JAX设备；这不表示底层CUDA已实现批量并行训练。
            # 后面的add_device_axis/remove_device_axis只负责加/去掉长度为1的设备维。
            collect = jax.pmap(functions["collect_step"], axis_name="d")
            actor_update = jax.pmap(functions["actor_step"], axis_name="d")
            critic_update = jax.pmap(functions["critic_step"], axis_name="d")
            # 只从inspect手内操作段采样，lift从不写入训练状态池或归一化统计。
            training_frames = np.arange(100, 208 if args.feasible_only else 240, dtype=np.int32)
            heldout_starts = [400, 405] if args.feasible_only else [420, 500]
            # 5. 选择reset起点；每次episode不必从整条轨迹第一帧开始。
            # 承重报告只用于筛选可能可行的连续区间，不保证软接触一定能实现参考动作。
            # 报告中的0.7Nm是可行性诊断条件；本训练实际力矩仍未施加该饱和。
            if args.feasibility_report:
                feasibility = json.loads(Path(args.feasibility_report).read_text())
                assert np.array_equal(
                    feasibility["gravity_m_s2"], model.opt.gravity
                ), "承重报告重力不一致"
                feasible = {
                    r["frame"]
                    for r in feasibility["records"]
                    if r.get("reference_instantaneous_dynamic_with_07Nm", False)
                }
                # reset之后的16步episode和8步反传窗口都不跨出已检查的连续区间。
                margin = config.max_episode_steps + args.horizon
                training_frames = np.array(
                    [
                        f
                        for f in range(100, 240 - margin)
                        if all(t in feasible for t in range(f, f + margin + 1))
                    ],
                    np.int32,
                )
                heldout_candidates = [
                    f for f in range(400, 540 - 12) if all(t in feasible for t in range(f, f + 13))
                ]
                assert len(training_frames) >= 2 and len(heldout_candidates) >= 2
                heldout_starts = [heldout_candidates[0], heldout_candidates[-1]]
            # 600帧的采样概率表：允许训练的帧等概率，其余（包括整条lift）为0。
            reset_probabilities = np.zeros(600)
            reset_probabilities[training_frames] = 1 / len(training_frames)
            reset_probabilities = jnp.asarray(reset_probabilities)
            # JAX随机数需要显式传key；split生成后续独立使用的key，而不是依赖全局随机状态。
            key = jax.random.PRNGKey(76)
            collector_state = functions["sample_resets"](key, reset_probabilities, None)
            assert_trees_equal(jax.vmap(env.restore)(env.compact(collector_state)), collector_state)
            # 6. 状态池保存策略实际走到的状态，不是示范动作数据集。
            # compact将状态压成可保存的数组集合，restore还原；上面的assert验证二者互逆。
            # 先分配64行容量，初始只有num_envs行有效；其余重复值只是占位。
            replay_capacity = 64
            replay_pool = jax.tree.map(
                lambda x: jnp.tile(x, (replay_capacity // num_envs,) + (1,) * (x.ndim - 1)),
                env.compact(collector_state),
            )
            # replay_pointer=[下一次写入的行号, 当前有效行数]，写满后循环覆盖旧状态。
            replay_pointer = jnp.array([num_envs, num_envs], jnp.int32)
            # 7. actor是策略网络参数，输入观测、输出16维动作；这里用小网络做验证。
            # critic是价值网络参数，输入状态观测、输出一个预期未来回报。
            # 下面仅用训练帧的reset观测初始化均值/方差，lift不参与这些统计。
            actor = nets.actor_init(jax.random.PRNGKey(1), env, hidden=(32, 32), latent=16)
            reference_states = jax.vmap(env.reset)(
                jax.random.split(jax.random.PRNGKey(2), len(training_frames)),
                jnp.asarray(training_frames),
            )
            policy_state_observation, policy_goal_observation = jax.vmap(env.policy_obs)(
                reference_states, jax.random.split(jax.random.PRNGKey(3), len(training_frames))
            )
            observations = jnp.concatenate(
                (policy_state_observation, policy_goal_observation), axis=-1
            )
            actor["obs_norm"] = (observations.mean(0), jnp.maximum(observations.var(0), 1e-4))
            critic = nets.mlp_init(
                jax.random.PRNGKey(4), [env.critic_dim, 32, 32, 1], out_scale=0.001
            )
            critic_observation_stats = (
                observations.mean(0),
                jnp.maximum(observations.var(0), 1e-4),
            )
            actor_optimizer_state = opt_actor.init(actor)
            critic_optimizer_state = opt_critic.init(critic)
            # 完整训练状态按固定顺序放在tuple中，类似需要整体存盘的C++结构体。
            # 下标：0策略参数，1策略优化器，2价值参数，3价值优化器，4价值观测均值/方差，
            #       5采集器状态，6状态池，7写指针，8随机key；启用PODS时9为动作目标缓存。
            # 这些顺序也用于run_round解包和checkpoint，不能只改其中一处。
            training_state = (
                actor,
                actor_optimizer_state,
                critic,
                critic_optimizer_state,
                critic_observation_stats,
                collector_state,
                replay_pool,
                replay_pointer,
                key,
            )
            if args.pods:
                # PODS临时保存窗口观测、改进后的动作目标、有效权重及回归优化器状态。
                # 数组前四维的含义按字段不同为：[缓存窗口数, 时间步数, 环境数, 特征/动作数]。
                # hist是任务观测部分的历史命名，不应据此假设这里缓存了多帧历史。
                action_target_buffer = dict(
                    hist=jnp.zeros((1, args.horizon, num_envs, env.hist_dim), jnp.bfloat16),
                    goal=jnp.zeros((1, args.horizon, num_envs, env.goal_dim), jnp.bfloat16),
                    tgt=jnp.zeros((1, args.horizon, num_envs, env.nu)),
                    w=jnp.zeros((1, args.horizon, num_envs)),
                    count=jnp.int32(0),
                    it=jnp.int32(0),
                    opt=opt_regression.init(actor),
                )
                training_state = (*training_state, action_target_buffer)
            batched = sim.vmapped(evaluation_env, None)

            # 8. 评估单步：只前向执行策略和实际物理，不更新网络或归一化统计。
            # parameters=None时网络残差为0，仍跟踪参考PD目标，不是把PD目标设成0。
            def eval_step(parameters, state):
                policy_state_observation, policy_goal_observation = batched["policy_obs"](
                    state, jax.random.split(jax.random.PRNGKey(0), 2)
                )
                actions = (
                    jnp.zeros((2, env.nu))
                    if parameters is None
                    else nets.actor_apply(
                        parameters, policy_state_observation, policy_goal_observation
                    )
                )
                after = batched["step"](state, actions)
                reward, _ = batched["reward"](after, actions, state.prev_a, prev=state)
                return after, reward, actions

            # jit编译JAX流程以减少反复解释开销；实际CUDA调用由physics的桥接接口完成。
            eval_step = jax.jit(eval_step)

            # 从两个指定参考帧开始，记录真实仿真轨迹和误差，写成NPZ数组文件。
            # starts始终是两个评估起点；max_steps是控制步数，不是物理子步数。
            def evaluate(parameters, starts, max_steps, filename):
                return evaluate_policy(
                    parameters,
                    starts,
                    max_steps,
                    filename,
                    env=evaluation_env,
                    eval_step=eval_step,
                    reference_qpos=reference_qpos,
                    model=model,
                    output=output,
                    num_envs=2,
                )

            # 先保存训练前表现，便于与训练后对照；resume时读取原基线，避免重新定义起点。
            if args.resume:
                before = json.loads((output / "before.json").read_text())
            else:
                print("evaluating before training", flush=True)
                before = dict(
                    train_contact=evaluate(actor, [120, 180], 12, "train_before.npz"),
                    heldout_contact=evaluate(actor, heldout_starts, 12, "heldout_before.npz"),
                )
                (output / "before.json").write_text(json.dumps(before, indent=2))

            # 9. 一轮训练的核心入口。输入完整训练状态，返回新状态和这一轮的日志。
            # 函数返回新数组；调用者必须接收返回值，才能继续下一轮。
            def run_round(current_training_state):
                (
                    actor,
                    actor_optimizer_state,
                    critic,
                    critic_optimizer_state,
                    critic_observation_stats,
                    collector_state,
                    replay_pool,
                    replay_pointer,
                    key,
                ) = current_training_state[:9]
                action_target_buffer = current_training_state[9] if args.pods else {}
                # A. 当前策略真实运行两次控制步，将所得状态写入循环状态池。
                # 失败或episode结束时，collect内部会从允许的训练参考帧重新reset。
                collection_failures = []
                for _ in range(2):
                    key, collection_key = jax.random.split(key)
                    collector_state, replay_pool, replay_pointer, failed = remove_device_axis(
                        jax.block_until_ready(
                            collect(
                                *add_device_axis(
                                    (
                                        actor,
                                        collector_state,
                                        replay_pool,
                                        replay_pointer,
                                        reset_probabilities,
                                        collection_key,
                                        None,
                                    )
                                )
                            )
                        )
                    )
                    collection_failures.append(float(failed))
                # B. 从状态池抽取反传起点，展开horizon步并更新策略。
                # PODS：求动作梯度→形成目标动作→回归策略；另一分支直接更新策略参数。
                # window是反传窗口的末状态；rollout_records保存窗口每步的奖励、观测等。
                # 返回的占位池被忽略：actor_update只读状态池，写入由上面的采集器负责。
                key, actor_key, critic_key = jax.random.split(key, 3)
                (
                    actor,
                    actor_optimizer_state,
                    window,
                    _,
                    replay_pointer,
                    rollout_records,
                    loss,
                    actor_update_statistic,
                    action_target_buffer,
                ) = remove_device_axis(
                    jax.block_until_ready(
                        actor_update(
                            *add_device_axis(
                                (
                                    actor,
                                    actor_optimizer_state,
                                    critic,
                                    critic_observation_stats,
                                    collector_state,
                                    replay_pool,
                                    replay_pointer,
                                    reset_probabilities,
                                    actor_key,
                                    None,
                                    action_target_buffer,
                                )
                            )
                        )
                    )
                )
                # C. 利用刚才窗口的数据构造TD(lambda)目标，拟合价值网络。
                # targets是每个状态的目标回报；critic_fit_statistics是拟合前后误差等诊断。
                critic, critic_optimizer_state, critic_loss, targets, critic_fit_statistics = (
                    remove_device_axis(
                        jax.block_until_ready(
                            critic_update(
                                *add_device_axis(
                                    (
                                        critic,
                                        critic_optimizer_state,
                                        critic_observation_stats,
                                        rollout_records,
                                        critic_key,
                                    )
                                )
                            )
                        )
                    )
                )
                # D. 用当前训练窗口观测缓慢更新价值输入的均值/方差（99%旧值+1%新值）。
                # 归一化只是调整网络输入尺度，不修改物理状态或参考数据。
                critic_observations = rollout_records["cobs"].reshape(-1, env.critic_dim)
                critic_observation_stats = (
                    0.99 * critic_observation_stats[0] + 0.01 * critic_observations.mean(0),
                    0.99 * critic_observation_stats[1]
                    + 0.01 * jnp.maximum(critic_observations.var(0), 1e-6),
                )
                result = (
                    actor,
                    actor_optimizer_state,
                    critic,
                    critic_optimizer_state,
                    critic_observation_stats,
                    collector_state,
                    replay_pool,
                    replay_pointer,
                    key,
                )
                if args.pods:
                    result = (*result, action_target_buffer)
                # E. 检查数值有限，且训练池没有混入帧号>=300的留出轨迹。
                # actor_update_statistic在PODS分支是回归损失，直接路径才是梯度范数。
                assert all_arrays_finite(
                    (result, rollout_records, loss, actor_update_statistic, critic_loss)
                )
                # 零梯度/零回归损失也可能合法，不能把它误判成计算崩溃。
                assert float(actor_update_statistic) >= 0
                assert np.all(
                    np.asarray(replay_pool["frame"])[: int(replay_pointer[1])] < 300
                ), "heldout数据进入训练池"
                metrics = dict(
                    actor_loss=float(loss),
                    actor_gradient_norm=None if args.pods else float(actor_update_statistic),
                    actor_update_stat=float(actor_update_statistic),
                    actor_update_stat_kind=(
                        "PODS regression loss" if args.pods else "direct gradient norm"
                    ),
                    action_target_count=int(action_target_buffer["count"]) if args.pods else 0,
                    critic_loss=float(critic_loss),
                    critic_fit=np.asarray(critic_fit_statistics).tolist(),
                    pool_filled=int(replay_pointer[1]),
                    collection_failure_fraction=collection_failures,
                    elapsed_seconds=time.monotonic() - start_time,
                )
                return result, metrics

            # 10. 开始循环。checkpoint不仅保存网络，也保存优化器、池、随机数和PODS缓存。
            # 缺少这些状态，“恢复训练”就不能精确延续原来的下一轮。
            start_round = 0
            if args.resume:
                training_state = jax.tree.map(jnp.asarray, resumed_state)
                history = resumed_history
                start_round = len(history)
                if args.pods and int(training_state[9]["count"]) != start_round:
                    raise ValueError("PODS缓存计数与checkpoint轮数不一致")
            # resume时应与恢复瞬间比较，而不是与新初始化的随机网络比较。
            initial_actor = jax.tree.map(np.asarray, training_state[0])
            initial_critic = jax.tree.map(np.asarray, training_state[2])
            for round_index in range(start_round, args.rounds):
                # 每轮只执行一次采集和网络更新；保存后即可从该状态续训。
                training_state, metrics = run_round(training_state)
                history.append(dict(round=round_index + 1, **metrics))
                save_latest(
                    output / "checkpoint_latest.pkl",
                    jax.tree.map(np.asarray, training_state),
                    history,
                )
                print(history[-1], flush=True)
                (output / "history.json").write_text(json.dumps(history, indent=2))
                # 每10轮独立评估训练段和留出段；评估不用于更新网络和训练统计。
                if (round_index + 1) % 10 == 0:
                    interim = dict(
                        round=round_index + 1,
                        train=evaluate(
                            training_state[0], [120, 180], 12, f"train_round{round_index+1}.npz"
                        ),
                        heldout=evaluate(
                            training_state[0],
                            heldout_starts,
                            12,
                            f"heldout_round{round_index+1}.npz",
                        ),
                        # 较长的训练段对照，避免12步窗口掩盖后续失稳。
                        train_continuous=evaluate(
                            training_state[0],
                            [100, 120],
                            100,
                            f"train_continuous_round{round_index+1}.npz",
                        ),
                    )
                    (output / f"evaluation_round{round_index+1}.json").write_text(
                        json.dumps(interim, indent=2)
                    )
                    print("evaluation", interim, flush=True)
            # 11. 训练后确认网络确实发生更新，运行最终评估并保存报告。
            # 权重变化只作诊断，零更新也可能合法；任务是否学会应看真实物理评估。
            actor = training_state[0]
            actor_weight_change = max(
                float(np.max(abs(np.asarray(a) - np.asarray(b))))
                for a, b in zip(
                    jax.tree.leaves({k: v for k, v in actor.items() if k != "obs_norm"}),
                    jax.tree.leaves({k: v for k, v in initial_actor.items() if k != "obs_norm"}),
                )
            )
            assert np.isfinite(actor_weight_change)
            critic_changed = max(
                float(np.max(abs(np.asarray(a) - np.asarray(b))))
                for a, b in zip(jax.tree.leaves(training_state[2]), jax.tree.leaves(initial_critic))
            )
            assert np.isfinite(critic_changed)
            after = dict(
                train_contact=evaluate(actor, [120, 180], 12, "train_after.npz"),
                heldout_contact=evaluate(actor, heldout_starts, 12, "heldout_after.npz"),
                train_continuous=evaluate(actor, [100, 120], 100, "train_continuous_after.npz"),
            )
            with (output / "final_checkpoint.pkl").open("wb") as stream:
                pickle.dump(jax.tree.map(np.asarray, training_state), stream)
            for path, digest in algorithm_source_hashes.items():
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
            if args.pods:
                assert int(training_state[9]["count"]) == args.rounds
            # results.json记录运行配置、训练前后表现和数值求解统计。
            # original_mimic_source_sha256沿用旧字段名，记录此独立副本的算法文件指纹。
            report = dict(
                gravity_m_s2=model.opt.gravity.tolist(),
                feasibility_report=args.feasibility_report,
                rounds=args.rounds,
                method=(
                    "NablaMimic PODS action-target"
                    if args.pods
                    else "direct-policy-gradient ablation"
                ),
                action_target=config.action_target,
                action_gradient_clip=config.act_cot_clip,
                action_noise=0.05 if args.pods else 0.0,
                action_target_buffer_count=int(training_state[9]["count"]) if args.pods else 0,
                feasible_reset_filter=bool(args.feasible_only or args.feasibility_report),
                training_reset_frames=training_frames.tolist(),
                velocity_feedforward=args.velocity_feedforward,
                heldout_evaluation_starts=heldout_starts,
                batch=num_envs,
                evaluation_batch=2,
                horizon=args.horizon,
                substeps=4,
                control_dt=0.0166,
                training_trajectory=f"DexTrack s2 cubesmall inspect_1, contact reset starts{training_frames[0]}..{training_frames[-1]}",
                heldout_trajectory="DexTrack s2 cubesmall lift",
                before=before,
                after=after,
                history=history,
                actor_max_weight_change=actor_weight_change,
                critic_max_weight_change=critic_changed,
                original_mimic_source_sha256=algorithm_source_hashes,
                geometry_max_scaled_disagreement=geometry.max_scaled_disagreement,
                verified_cost_stop_callback_calls=physics.verified_cost_stop_calls,
                geometry_derivative_failures=geometry.failures,
                geometry_adaptive_reductions=geometry.adaptive_reductions,
                geometry_minimum_epsilon=geometry.minimum_epsilon,
                scope="Full actual collector/replay/actor+critic/checkpoint/evaluation loop; Nabla CUDA dynamics, MuJoCo collision only, numerical geometry JVP; not pure GPU or PhysX/hardware reproduction",
                physical_profile="PD20/1, passive damping from URDF, dry friction0, armature1e-4, unsaturated torque; target angles clipped to original limits; plastic density1050, fixed palm, gravity enabled, no hand self-contact pairs",
                elapsed_seconds=time.monotonic() - start_time,
            )
            (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print("completed training experiment", flush=True)


if __name__ == "__main__":
    main()
