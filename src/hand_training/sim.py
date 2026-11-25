"""Batched physics for TrackingEnv: NablaEngine (the fused differentiable engine) or patched MuJoCo Warp, between
vmapped per-env pre- and post-step functions, and the per-step reset select."""

import jax
import jax.numpy as jnp
import numpy as np

from .nets import tree_where

BIN = 50  # reset-sampling bins of 1 s

# Per State leaf, True where the reset select may keep the stepped value: leaves the physics step never writes and every
# reset state shares (MJX's constraint and contact buffers, which the engine and MuJoCo Warp paths never touch). Found at
# start-up from probe states (skip_mask); None selects every leaf.
STATE_SKIP = None


def select_reset(done, reset, s2):
    """tree_where(done, reset, s2), skipping the leaves STATE_SKIP marks (their two values are identical)."""
    if STATE_SKIP is None:
        return tree_where(done, reset, s2)
    return jax.tree_util.tree_map(
        lambda skip, x, y: y if skip else jnp.where(done.reshape(done.shape + (1,) * (x.ndim - done.ndim)), x, y),
        STATE_SKIP, reset, s2)


def skip_mask(x0, *others):
    """Per leaf of the batched state x0: True where it equals the same leaf of every state in others and is the same in
    every env (never for PRNG keys)."""
    def skippable(x, *ys):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            return False
        x, ys = np.asarray(x), [np.asarray(y) for y in ys]
        nan = x.dtype.kind == "f"
        return bool(all(np.array_equal(x, y, equal_nan=nan) for y in ys)
                    and np.array_equal(x, np.broadcast_to(x[:1], x.shape), equal_nan=nan))
    return jax.tree_util.tree_map(skippable, x0, *others)


def vmapped(env, tables):
    """The env functions vmapped over a batch; "step" runs env.warp ({batch size: physics} or one physics object) on the
    whole batch and finishes each env's state."""
    e = env.bind(tables)
    f = {k: jax.vmap(getattr(e, k)) for k in
         ("reset", "policy_obs", "critic_obs", "reward", "terminated", "clip_ended", "hazard_features")}

    def step_batched(state, action):
        wp_ = env.warp[state.frame.shape[0]] if isinstance(env.warp, dict) else env.warp
        out = wp_.step(state.dx.qpos, state.dx.qvel, state.dx.qacc_warmstart,
                       jax.vmap(e.ctrl_of)(action, state.frame), state.xi.friction, state.xi.com)
        if len(out) == 8:  # engine with kinematics: (qpos', qvel', warm', qpos_n, xpos, xquat, cvel, subtree_com)
            _, v, w, qn, xp, xq, cv, sc = out
            return jax.vmap(e.finish_physics_kin)(state, action, qn, v, w, xp, xq, cv, sc)
        q, v, w = out
        return jax.vmap(e.finish_physics)(state, action, q, v, w)

    f["step"] = step_batched
    return f
