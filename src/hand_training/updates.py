"""复用NablaMimic的训练算法，函数体保持原样。

make_rollout：展开物理轨迹、累计奖励并在窗口末尾用critic补估价值。
build_training_functions：显式接收环境、优化器等依赖，返回采集、actor、critic及PODS更新函数。
原工程把更新函数写在main内部；这里将它们移入普通工厂函数，便于阅读，不在运行时提取AST。
输入输出的数组格式由train.py建立；实际动力学由任务环境调用。
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from . import nets
from .nets import actor_apply, actor_decode, actor_encode, actor_init, clip_rows_to_median, critic_apply, normalize
from .sim import BIN, select_reset, vmapped
AT_BATCH, AT_EPS = 16384, 1e-8
SHAC_GUARD = 1e6

def termination_penalty(env, f, s2, term_penalty):
    """-W*dt*[(relu(dz-DZ0)/(h-DZ0))^2 + (relu(dtilt-DT0)/(t-DT0))^2] per env (see --term-penalty)."""
    w, dz0, dg0 = term_penalty
    hf = f["hazard_features"](s2)  # [h - dz, rate, t - dtilt, rate]
    h, tt = env.cfg.term_height, env.cfg.term_tilt
    return -w * 0.02 * ((jnp.maximum(h - hf[:, 0] - dz0, 0.0) / (h - dz0)) ** 2
                        + (jnp.maximum(tt - hf[:, 2] - dg0, 0.0) / (tt - dg0)) ** 2)

def make_rollout(env, horizon, gamma, max_ep, remat, act_clip=None, act_noise=0.0, mask_after_done=False,
                 term_penalty=None, record_pre=False, record_obs=False, enc_hoist=False, remat_policy=None,
                 v_bound=None, clip_end_pre=False):
    """The window objective over `horizon` steps; returns -J (mean over envs) and (final state, per-step records).

    act_noise=sigma executes a = pi(o) + sigma * N(0, I) (action units), so the objective is the return of the noisy
    policy; the pathwise gradient still flows through pi. act_clip=kappa clips, at every step, each env's action
    cotangent dJ/da to kappa x the batch median (per-sample clipping; the forward is unchanged); differentiating with
    respect to probe=zeros(2) yields the clipping statistics summed over the steps (nets.clip_rows_to_median), and
    with respect to delta=zeros((H, n, nu)) the per-step action gradients dJ/da_t. mask_after_done: after an episode end
    (termination, clip end or time limit, bootstrapped as usual) the rest of the window adds nothing to J; the env still
    restarts from reset_state, so the critic keeps training on those steps. enc_hoist (actors with the anchor in the
    decoder): the motion encoder's outputs for the whole window in two batched calls before the scan (the frames along
    the clip from the start state and from the reset state) instead of one call per step; the same policy. remat:
    jax.checkpoint of each step with remat_policy (the engine's saves only its step tapes). v_bound: the critic's
    values clipped to [-v_bound, v_bound] where they bootstrap (and so in the critic's targets). clip_end_pre: an
    episode end at a clip end bootstraps with V of the transition's start state (a fitted critic input) instead of V of
    the clip's last frame (goal window frozen at it, never a critic training input)."""
    def rollout(actor, critic, cnorm, state, reset_state, key, tables, probe=None, delta=None):
        f = vmapped(env, tables)
        v_pobs, v_cobs, v_step, v_rew = f["policy_obs"], f["critic_obs"], f["step"], f["reward"]
        v_term, v_end = f["terminated"], f["clip_ended"]
        n = state.frame.shape[0]
        hoist = enc_hoist and "anchor_dec" in actor
        if hoist:
            assert "obs_norm" not in actor, "--enc-hoist: not with --actor-obs-norm"
            k_anc = actor["anchor_dec"].shape[-1]
            e_ref = env.bind(tables)
            v_goal_ref = jax.vmap(e_ref.goal_ref)

            def enc_along(f0):  # (n,) start frames -> (horizon, n, latent): the encoder along the clip, one batched call
                fr = jnp.minimum(f0[None, :] + jnp.arange(horizon)[:, None], e_ref.ref_last[f0][None, :])
                return actor_encode(actor, v_goal_ref(fr.reshape(-1))).reshape(horizon, n, -1)

            z_cont, z_rst = enc_along(state.frame), enc_along(reset_state.frame)

        def body(c, xs):
            if hoist:  # z_t from the continuation table until the env's first reset, then from the reset table
                xs, zc = xs
                c, jr = c[:-1], c[-1]
            k, d = xs if delta is not None else (xs, None)
            state, acc, gam, total, surv = c
            hist, goal = v_pobs(state, jax.random.split(k, n))
            if hoist:
                zt = jnp.where((jr >= 0)[:, None], z_rst[jnp.maximum(jr, 0), jnp.arange(n)], zc)
                a = actor_decode(actor, zt, hist, goal[..., -k_anc:])
            else:
                a = actor_apply(actor, hist, goal)
            a_pi = a
            if act_noise > 0:
                a = a + act_noise * jax.random.normal(jax.random.fold_in(k, 7), a.shape)
            if act_clip is not None:
                a = clip_rows_to_median(a, act_clip, probe)
            if d is not None:  # zero offset whose gradient is dJ/da_t (unclipped)
                a = a + d
            s2 = v_step(state, a)
            r, terms = v_rew(s2, a, state.prev_a, prev=state)
            if term_penalty is not None:  # smooth penalty on the termination quantities (training objective only)
                pen = termination_penalty(env, f, s2, term_penalty)
                r = r + pen
                terms = {**terms, "term_pen": pen}
            term = v_term(s2)
            cend = v_end(s2)
            done = term | cend | (s2.ep >= max_ep)
            cob = v_cobs(s2)
            v2 = critic_apply(critic, normalize(cnorm, cob))[:, 0]
            if clip_end_pre:
                cob_pre = v_cobs(state)
                v_pre = critic_apply(critic, normalize(cnorm, cob_pre))[:, 0]
                v_gap = jax.lax.stop_gradient(v2 - v_pre)  # logged: the extrapolation step V(s_last) - V(s_t)
                v2 = jnp.where(cend, v_pre, v2)
            r = jnp.where(jnp.isfinite(r), r, 0.0)
            v2 = jnp.where(jnp.isfinite(v2), v2, 0.0)
            if v_bound is not None:
                v2 = jnp.clip(v2, -v_bound, v_bound)
            acc = acc + gam * surv * r  # surv: 1, or the alive mask with mask_after_done
            gam = gam * gamma
            total = total + jnp.where(done, acc + gam * surv * v2 * (1.0 - term), 0.0)
            if mask_after_done:
                surv = jnp.where(done, 0.0, surv)
            acc = jnp.where(done, 0.0, acc)
            gam = jnp.where(done, 1.0, gam)
            s_next = select_reset(done, reset_state, s2)
            out = dict(cobs=jax.lax.stop_gradient(cob), r=jax.lax.stop_gradient(r), term=term, done=done,
                       v2=jax.lax.stop_gradient(v2), frame=s2.frame, terms=jax.lax.stop_gradient(terms))
            if clip_end_pre:
                out["cend"], out["v_gap"] = cend & ~term, v_gap
            if record_pre:  # critic input of the state the transition starts from (the target's state)
                out["cobs_pre"] = jax.lax.stop_gradient(cob_pre if clip_end_pre else v_cobs(state))
            if record_obs:  # policy inputs and output, for --action-target
                out["hist"] = jax.lax.stop_gradient(hist).astype(jnp.bfloat16)
                out["goal"] = jax.lax.stop_gradient(goal).astype(jnp.bfloat16)
                out["a_pi"] = jax.lax.stop_gradient(a_pi)
            if hoist:  # steps since the env's last reset (-1: not reset in this window); a reset restarts at reset_state
                jr = jnp.where(done, 0, jnp.where(jr >= 0, jr + 1, -1))
                return (s_next, acc, gam, total, surv, jr), out
            return (s_next, acc, gam, total, surv), out

        step_fn = jax.checkpoint(body, policy=remat_policy) if remat else body
        init = (state, jnp.zeros(n), jnp.ones(n), jnp.zeros(n), jnp.ones(n))
        keys = jax.random.split(key, horizon)
        xs = (keys, delta) if delta is not None else keys
        if hoist:
            init, xs = init + (jnp.full(n, -1, jnp.int32),), (xs, z_cont)
        (state, acc, gam, total, surv, *_), outs = jax.lax.scan(step_fn, init, xs)
        v_h = critic_apply(critic, normalize(cnorm, v_cobs(state)))[:, 0]
        if v_bound is not None:
            v_h = jnp.clip(v_h, -v_bound, v_bound)
        total = total + acc + gam * surv * jnp.where(jnp.isfinite(v_h), v_h, 0.0)
        return -jnp.mean(total), (jax.lax.stop_gradient(state), outs)

    return rollout

def build_training_functions(*, env, args, n, nc, use_replay, use_at, opt_r, opt_a, opt_c, rollout):
    def sample_resets(key, probs, tables, size=None):
        size = size or n
        k1, k2 = jax.random.split(key)
        frames = jax.random.choice(k1, env.n_frames, (size,), p=probs)
        return vmapped(env, tables)["reset"](jax.random.split(k2, size), frames)

    def collect_step(actor, cstate, pool, ptr, probs, key, tables):
        """One step of the data collector: noisy policy, reset sampling, states into the replay."""
        f = vmapped(env, tables)
        k1, k2, k3 = jax.random.split(key, 3)
        hist, goal = f["policy_obs"](cstate, jax.random.split(k1, nc))
        a = actor_apply(actor, hist, goal) + args.collect_noise * jax.random.normal(k2, (nc, env.nu))
        s2 = f["step"](cstate, a)
        term = f["terminated"](s2)
        done = term | f["clip_ended"](s2) | (s2.ep >= args.max_episode_steps)
        s_next = select_reset(done, sample_resets(k3, probs, tables, nc), s2)
        psize = pool["frame"].shape[0]
        pos = (ptr[0] + jnp.arange(nc)) % psize
        row = env.compact(s_next)
        pool = jax.tree_util.tree_map(lambda p, v: p.at[pos].set(v), pool, row)
        ptr = jnp.stack([(ptr[0] + nc) % psize, jnp.minimum(ptr[1] + nc, psize)])
        return s_next, pool, ptr, jax.lax.pmean(jnp.mean(term), "d")

    def action_target_update(actor, buf, gd, outs, key):
        """--action-target: targets from this window's action gradients into the ring buffer, then M regression steps."""
        m_steps, k_win, delta = int(args.action_target[0]), int(args.action_target[1]), args.action_target[2]
        if args.action_target_delta_floor < 1.0 and args.lr_decay_steps > 0:  # cosine-annealed step
            frac = jnp.minimum(buf["it"].astype(jnp.float32) / args.lr_decay_steps, 1.0)
            fl = args.action_target_delta_floor
            delta = delta * (fl + (1.0 - fl) * 0.5 * (1.0 + jnp.cos(jnp.pi * frac)))
        g = -gd  # ascent direction per step and env (H, n, nu); exactly-zero rows are masked out of the objective
        nonfin = jnp.any(~jnp.isfinite(g), -1)
        g = jnp.where(jnp.isfinite(g), g, 0.0)
        nrm = jnp.sqrt(jnp.sum(g ** 2, -1))  # |dJ/da_t|: row norms of the window clip and the per-step profile
        live = nrm > 0
        sc = jnp.ones_like(nrm)
        if args.act_cot_clip is not None:
            nl = jnp.where(live, nrm, jnp.nan)
            thr = args.act_cot_clip * jnp.nanmedian(nl)
            thr = jnp.where(jnp.isfinite(thr), thr, jnp.inf)
            sc = jnp.minimum(1.0, thr / (nrm + 1e-30))
            g = g * sc[..., None]
        cl = live & (sc < 1.0)
        nlive = jnp.maximum(jnp.sum(live), 1)
        med_raw = jnp.nanmedian(jnp.where(live, nrm, jnp.nan))
        # robust to single huge rows (norm sums are not): [share of live rows clipped, mean shrink 1 - scale over live
        # rows, share of live rows above 100 x the median |dJ/da_t|, rows with non-finite entries (zeroed)]
        outs["at_clip"] = jnp.stack([jnp.sum(cl) / nlive, jnp.sum(jnp.where(live, 1.0 - sc, 0.0)) / nlive,
                                     jnp.sum(live & (nrm > 100.0 * med_raw)) / nlive,
                                     jnp.sum(nonfin).astype(g.dtype)])
        if args.horizon % 8 == 0:  # per eighth of the window: median |dJ/da_t| / overall median, share clipped
            lb = live.reshape(8, -1)
            mb = jnp.nanmedian(jnp.where(live, nrm, jnp.nan).reshape(8, -1), 1)
            outs["at_prof"] = jnp.stack([jnp.where(jnp.isfinite(mb), mb, 0.0) / (med_raw + 1e-30),
                                         jnp.sum(cl.reshape(8, -1), 1) / jnp.maximum(jnp.sum(lb, 1), 1)])
        s2 = env.action_scale ** 2  # the step is measured in joint-setpoint radians
        ss = jax.lax.pmean(jnp.sum(jnp.where(live[..., None], g ** 2 * s2, 0.0)), "d")
        cnt = jax.lax.pmean(jnp.maximum(jnp.sum(live), 1).astype(g.dtype) * env.nu, "d")
        eta = delta / jnp.sqrt(ss / cnt + 1e-30)
        tgt = outs.pop("a_pi") + eta * g
        j = buf["count"] % k_win
        buf = {**buf, "hist": buf["hist"].at[j].set(outs.pop("hist")), "goal": buf["goal"].at[j].set(outs.pop("goal")),
               "tgt": buf["tgt"].at[j].set(tgt), "w": buf["w"].at[j].set(live.astype(g.dtype)),
               "count": buf["count"] + 1, "it": buf["it"] + 1}
        nfill = jnp.minimum(buf["count"], k_win)
        w_err = env.action_scale  # regression error in setpoint radians

        def reg(carry, kk):
            p, st = carry
            ka, kb, kc = jax.random.split(kk, 3)
            si = jax.random.randint(ka, (AT_BATCH,), 0, nfill)
            ti = jax.random.randint(kb, (AT_BATCH,), 0, args.horizon)
            ei = jax.random.randint(kc, (AT_BATCH,), 0, n)
            hb, gb = buf["hist"][si, ti, ei].astype(jnp.float32), buf["goal"][si, ti, ei].astype(jnp.float32)
            yb, wb = buf["tgt"][si, ti, ei], buf["w"][si, ti, ei]

            def lf(q):
                err = (actor_apply(q, hb, gb) - yb) * w_err
                return jnp.sum(wb * jnp.mean(err ** 2, -1)) / jnp.maximum(jnp.sum(wb), 1.0)

            lb, gr = jax.value_and_grad(lf)(p)
            gr = jax.lax.pmean(gr, "d")
            u, st = opt_r.update(gr, st, p)
            p = optax.apply_updates(p, u)
            return (p, st), lb

        (actor, st_r), ls = jax.lax.scan(reg, (actor, buf["opt"]), jax.random.split(key, m_steps))
        return actor, {**buf, "opt": st_r}, jax.lax.pmean(jnp.mean(ls), "d")

    def actor_step(actor, st_a, critic, cnorm, state, pool, ptr, probs, key, tables, buf):
        k1, k2 = jax.random.split(key)
        k3, k4 = jax.random.split(jax.random.fold_in(key, 1))
        reset_state = jax.lax.stop_gradient(sample_resets(k1, probs, tables))
        if use_replay:  # ptr = [write position, number of filled entries]
            pick = jax.random.randint(k3, (n,), 0, ptr[1])
            start = jax.vmap(env.bind(tables).restore)(jax.tree_util.tree_map(lambda x: x[pick], pool))
            # fresh keys, so a state drawn twice does not replay the same sensor noise
            start = start._replace(key=jax.random.split(jax.random.fold_in(k4, 1), n))
        else:  # continue the previous windows
            start = state
        if use_at:  # gradient with respect to the actions only
            (loss, (state, outs)), (gd, g_probe) = jax.value_and_grad(rollout, argnums=(8, 7), has_aux=True)(
                actor, critic, cnorm, start, reset_state, k2, tables, jnp.zeros(2),
                jnp.zeros((args.horizon, n, env.nu)))
            clip_stats = jax.lax.pmean(g_probe / args.horizon, "d")  # in-graph clipping: [rows clipped, norm removed]
        elif args.act_cot_clip is not None:  # the actor's window gradient with the in-graph clipping (no action targets)
            (loss, (state, outs)), (g, g_probe) = jax.value_and_grad(rollout, argnums=(0, 7), has_aux=True)(
                actor, critic, cnorm, start, reset_state, k2, tables, jnp.zeros(2))
            clip_stats = jax.lax.pmean(g_probe / args.horizon, "d")  # [rows clipped, norm removed]
        else:
            (loss, (state, outs)), g = jax.value_and_grad(rollout, has_aux=True)(
                actor, critic, cnorm, start, reset_state, k2, tables)
            clip_stats = jnp.zeros(2)
        if use_replay:
            pool = jnp.zeros(())  # the replay is read-only here and stays with the caller
        # RMS change of the joint setpoint [rad] an update causes at the batch's current states (logged)
        hist, goal = vmapped(env, tables)["policy_obs"](state, jax.random.split(k1, state.frame.shape[0]))
        a_old = actor_apply(actor, hist, goal)

        def da_of(params):
            return (actor_apply(params, hist, goal) - a_old) * env.action_scale

        def dq_of(params):
            return jnp.sqrt(jax.lax.pmean(jnp.mean(da_of(params) ** 2), "d"))

        def dq_env(params):  # per-env RMS over joints
            return jax.lax.all_gather(jnp.sqrt(jnp.mean(da_of(params) ** 2, -1)), "d").reshape(-1)

        if use_at:  # gnorm reports the mean regression loss (rad^2)
            actor, buf, gnorm = action_target_update(actor, buf, gd, outs, jax.random.fold_in(key, 2))
        else:
            g = jax.lax.pmean(g, "d")
            gnorm = optax.global_norm(g)
            finite = jnp.isfinite(gnorm) & (gnorm <= SHAC_GUARD) if args.shac_strict else jnp.isfinite(gnorm)
            g = jax.tree_util.tree_map(lambda x: jnp.where(finite, x, 0.0), g)
            upd, st_a_new = opt_a.update(g, st_a, actor)
            actor_new = optax.apply_updates(actor, upd)
            actor = jax.tree_util.tree_map(lambda a, b: jnp.where(finite, a, b), actor_new, actor)
            st_a = jax.tree_util.tree_map(lambda a, b: jnp.where(finite, a, b), st_a_new, st_a)
        outs["dq_rms"] = dq_of(actor)
        outs["dq_env_q"] = jnp.quantile(dq_env(actor), jnp.array([0.5, 0.95, 1.0]))
        outs["clip_stats"] = jnp.concatenate([clip_stats, outs.pop("at_clip")]) if use_at else clip_stats
        if "obs_norm" in actor:  # EMA of the actor-input statistics at the batch's current states
            x = jnp.concatenate([hist, goal], -1)
            mean, sq = jax.lax.pmean(x.mean(0), "d"), jax.lax.pmean((x ** 2).mean(0), "d")
            m0, v0 = actor["obs_norm"]
            actor = {**actor, "obs_norm": (0.99 * m0 + 0.01 * mean, 0.99 * v0 + 0.01 * jnp.maximum(sq - mean ** 2, 1e-6))}
        return actor, st_a, state, pool, ptr, outs, jax.lax.pmean(loss, "d"), gnorm, buf

    def critic_step(critic, st_c, cnorm, outs, key):
        tgt = nets.td_lambda(outs["r"], outs["term"], outs["done"], outs["v2"], args.gamma, args.lam)
        x = normalize(cnorm, outs["cobs_pre" if args.critic_pre_state else "cobs"].reshape(-1, env.critic_dim))
        y = tgt.reshape(-1)
        wt = jnp.ones_like(y)  # unit sample weights (the loss keeps its weighted-mean form)
        m = x.shape[0]
        mb = m // args.critic_minibatches

        def fit(c):  # MSE and explained variance of the window's targets
            e = critic_apply(c, x)[:, 0] - y
            mse, ym = jax.lax.pmean(jnp.mean(e ** 2), "d"), jax.lax.pmean(jnp.mean(y), "d")
            vy = jax.lax.pmean(jnp.mean((y - ym) ** 2), "d")
            return jnp.stack([mse, 1.0 - mse / (vy + 1e-12)])

        fit_pre = fit(critic)

        def loss_fn(c, xb, yb, wb):
            return jnp.sum(wb * (critic_apply(c, xb)[:, 0] - yb) ** 2) / jnp.maximum(jnp.sum(wb), 1.0)

        def epoch(carry, k):
            c, s = carry
            perm = jax.random.permutation(k, m)

            def mini(carry, i):
                c, s = carry
                idx = jax.lax.dynamic_slice(perm, (i * mb,), (mb,))
                lb, gc = jax.value_and_grad(loss_fn)(c, x[idx], y[idx], wt[idx])
                gc = jax.lax.pmean(gc, "d")
                u, s = opt_c.update(gc, s, c)
                return (optax.apply_updates(c, u), s), lb

            (c, s), ls = jax.lax.scan(mini, (c, s), jnp.arange(args.critic_minibatches))
            return (c, s), ls.mean()

        (critic, st_c), ls = jax.lax.scan(epoch, (critic, st_c), jax.random.split(key, args.critic_epochs))
        # [MSE before, after, explained variance before, after]: "before" is the fit to fresh data
        cfit = jnp.concatenate([fit_pre, fit(critic)])[jnp.array([0, 2, 1, 3])]
        return critic, st_c, jax.lax.pmean(ls.mean(), "d"), tgt, cfit

    return {'sample_resets': sample_resets, 'collect_step': collect_step, 'action_target_update': action_target_update, 'actor_step': actor_step, 'critic_step': critic_step}
