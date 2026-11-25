"""独立评估：每个环境首次失败即停止计分，并区分末帧达标与全程达标。

输入为网络参数、reset帧及真实物理eval_step；不执行网络更新。
NPZ保留批量矩形数组，但valid_step=False的冻结尾帧不能计入误差或回报。
"""

import jax
import jax.numpy as jnp
import numpy as np
from .tree_utils import all_arrays_finite


def evaluate_policy(
    parameters,
    starts,
    max_steps,
    filename,
    *,
    env,
    eval_step,
    reference_qpos,
    model,
    output,
    num_envs
):
    if len(starts) != num_envs or max_steps <= 0:
        raise ValueError("评估起点数必须等于环境数，步数必须为正")
    state = jax.vmap(env.reset)(
        jax.random.split(jax.random.PRNGKey(7600), num_envs), jnp.asarray(starts, jnp.int32)
    )
    records = {
        name: []
        for name in (
            "qpos",
            "qvel",
            "actions",
            "reward",
            "position_error_m",
            "orientation_error_rad",
            "joint_rmse_rad",
            "valid_step",
            "frame",
        )
    }
    failures = np.zeros(num_envs, bool)
    first_failure = np.full(num_envs, -1)
    active = np.ones(num_envs, bool)
    for step in range(max_steps):
        stepped, reward, actions = jax.block_until_ready(eval_step(parameters, state))
        # 已结束的环境保留最后状态。物理批量调用仍存在，但其尾帧不计入任何指标。
        state = jax.tree.map(
            lambda new, old: jnp.where(
                jnp.asarray(active).reshape((num_envs,) + (1,) * (new.ndim - 1)), new, old
            ),
            stepped,
            state,
        )
        if not all_arrays_finite((state, reward, actions)):
            raise FloatingPointError("评估出现非有限状态、奖励或动作")
        qpos = np.asarray(state.dx.qpos)
        frames = np.asarray(state.frame)
        targets = reference_qpos[frames]
        position = np.linalg.norm(qpos[:, 16:19] - targets[:, 16:19], axis=1)
        dots = np.abs(np.sum(qpos[:, 19:23] * targets[:, 19:23], axis=1))
        rotation = 2 * np.arccos(np.clip(dots, 0, 1))
        joints = np.sqrt(np.mean((qpos[:, :16] - targets[:, :16]) ** 2, axis=1))
        failed = active & np.asarray(env_batch_terminated(env, state))
        first_failure = np.where(failed, step + 1, first_failure)
        failures |= failed
        values = dict(
            qpos=qpos,
            qvel=np.asarray(state.dx.qvel),
            actions=np.asarray(actions),
            reward=np.asarray(reward),
            position_error_m=position,
            orientation_error_rad=rotation,
            joint_rmse_rad=joints,
            valid_step=active.copy(),
            frame=frames,
        )
        for name, value in values.items():
            records[name].append(value)
        # 首次失败的那一步仍是有效观测；失败之后的冻结尾帧才无效。
        clip_end = np.asarray(jax.vmap(env.clip_ended)(state))
        active &= ~(failed | clip_end)
        if not np.any(active):
            break
    arrays = {name: np.asarray(value) for name, value in records.items()}
    np.savez_compressed(output / filename, **arrays, starts=starts, gravity=model.opt.gravity)
    valid = arrays["valid_step"]
    counts = valid.sum(0)
    final_index = counts - 1
    env_index = np.arange(num_envs)
    pos, rot = arrays["position_error_m"], arrays["orientation_error_rad"]
    final_pos, final_rot = pos[final_index, env_index], rot[final_index, env_index]
    completed = counts == max_steps
    endpoint_success = ~failures & completed & (final_pos < 0.02) & (final_rot < 0.35)
    sustained_success = (
        ~failures & completed & np.all(~valid | ((pos < 0.02) & (rot < 0.35)), axis=0)
    )
    return dict(
        starts=list(starts),
        requested_control_steps=max_steps,
        executed_control_steps=len(valid),
        executed_control_steps_per_env=counts.tolist(),
        return_mean=float(np.where(valid, arrays["reward"], 0).sum(0).mean()),
        position_error_mean_m=float(pos[valid].mean()),
        position_error_mean_per_env_m=[float(pos[:, i][valid[:, i]].mean()) for i in env_index],
        position_error_final_m=final_pos.tolist(),
        orientation_error_final_rad=final_rot.tolist(),
        joint_rmse_final_rad=arrays["joint_rmse_rad"][final_index, env_index].tolist(),
        failure=failures.tolist(),
        first_failure_control_step=first_failure.tolist(),
        success_fraction=float(endpoint_success.mean()),
        sustained_success_fraction=float(sustained_success.mean()),
        metric_scope="control-step samples; first failure included, frozen tail excluded; no physics-substep success claim",
    )


def env_batch_terminated(env, state):
    return jax.vmap(env.terminated)(state)
