"""验证会改变实验结论的边界：失败尾帧、成功指标、断点原子一致性与配置拒绝。"""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import NamedTuple
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from hand_training.evaluation import evaluate_policy
from hand_training.checkpoint import validate_run_directory, save_latest, load_latest, run_signature
from hand_training.nets import td_lambda


class PhysicsData(NamedTuple):
    qpos: object
    qvel: object


class State(NamedTuple):
    dx: PhysicsData
    frame: object
    ep: object


class FixtureEnv:
    def reset(self, key, frame):
        q = jnp.zeros(23).at[19].set(1.0)
        return State(PhysicsData(q, jnp.zeros(22)), frame, jnp.int32(0))

    def terminated(self, state):
        return state.dx.qpos[16] > 0.15

    def clip_ended(self, state):
        return state.frame >= 299


class TrainingContract(unittest.TestCase):
    def test_failed_environment_tail_excluded(self):
        env = FixtureEnv()

        def step(params, state):
            # 第一环境第一步失败，第二环境三步都存活；失败后的夸大误差不能污染均值。
            q = state.dx.qpos.at[:, 16].add(jnp.array([0.2, 0.01]))
            return (
                State(PhysicsData(q, state.dx.qvel), state.frame + 1, state.ep + 1),
                jnp.ones(2),
                jnp.zeros((2, 16)),
            )

        ref = np.zeros((600, 23))
        ref[:, 19] = 1
        with TemporaryDirectory() as temp:
            result = evaluate_policy(
                None,
                [0, 0],
                3,
                "eval.npz",
                env=env,
                eval_step=step,
                reference_qpos=ref,
                model=SimpleNamespace(opt=SimpleNamespace(gravity=np.zeros(3))),
                output=Path(temp),
                num_envs=2,
            )
            self.assertEqual(result["executed_control_steps_per_env"], [1, 3])
            self.assertEqual(result["first_failure_control_step"], [1, -1])
            self.assertAlmostEqual(result["position_error_mean_m"], (0.2 + 0.01 + 0.02 + 0.03) / 4)
            self.assertEqual(result["return_mean"], 2.0)
            data = np.load(Path(temp) / "eval.npz")
            np.testing.assert_array_equal(
                data["valid_step"], [[True, True], [False, True], [False, True]]
            )
            np.testing.assert_array_equal(data["qpos"][:, 0, 16], [0.2, 0.2, 0.2])

    def test_endpoint_not_sustained_success(self):
        env = FixtureEnv()

        def step(params, state):
            x = jnp.where(state.ep == 0, 0.03, 0.01)
            q = state.dx.qpos.at[:, 16].set(x)
            return (
                State(PhysicsData(q, state.dx.qvel), state.frame + 1, state.ep + 1),
                jnp.ones(2),
                jnp.zeros((2, 16)),
            )

        ref = np.zeros((600, 23))
        ref[:, 19] = 1
        with TemporaryDirectory() as temp:
            result = evaluate_policy(
                None,
                [0, 0],
                2,
                "eval.npz",
                env=env,
                eval_step=step,
                reference_qpos=ref,
                model=SimpleNamespace(opt=SimpleNamespace(gravity=np.zeros(3))),
                output=Path(temp),
                num_envs=2,
            )
            self.assertEqual(result["success_fraction"], 1.0)
            self.assertEqual(result["sustained_success_fraction"], 0.0)

    def test_signature_rejects_mismatch_overwrite_and_missing(self):
        with TemporaryDirectory() as temp:
            output = Path(temp)
            sig = {"pods": True, "horizon": 8}
            validate_run_directory(output, sig, resume=False)
            with self.assertRaises(ValueError):
                validate_run_directory(output, sig, resume=False)
            with self.assertRaises(ValueError):
                validate_run_directory(output, sig, resume=True)
            save_latest(output / "checkpoint_latest.pkl", (np.array([1]),), [{"round": 1}])
            validate_run_directory(output, sig, resume=True)
            with self.assertRaises(ValueError):
                validate_run_directory(output, {"pods": False, "horizon": 8}, resume=True)
            state, history = load_latest(output / "checkpoint_latest.pkl")
            np.testing.assert_array_equal(state[0], [1])
            self.assertEqual(history, [{"round": 1}])
            self.assertFalse((output / "checkpoint_latest.tmp").exists())

    def test_reference_layout_limits_and_boundaries(self):
        from hand_training.data import load_experiment_model
        from nablaengine.hand_engine.reference_tracking import HandReferenceTracking
        from nablaengine.hand_engine.mimic_interface import HandActionMap

        with TemporaryDirectory() as temp:
            model, layout, q, v = load_experiment_model(temp)
        self.assertEqual(q.shape, (600, 23))
        self.assertEqual(v.shape, (600, 22))
        self.assertTrue(np.isfinite(q).all() and np.isfinite(v).all())
        np.testing.assert_allclose(np.linalg.norm(q[:, 19:23], axis=1), 1, atol=1e-12)
        self.assertLessEqual(float(np.max(model.jnt_range[:16, 0] - q[:, :16])), 1e-10)
        self.assertLessEqual(float(np.max(q[:, :16] - model.jnt_range[:16, 1])), 1e-10)
        np.testing.assert_array_equal(model.opt.gravity, [-9.81, 0, 0])
        physics = SimpleNamespace(
            layout=layout,
            info={"control_dt": 0.0166},
            mode="position_pd",
            kp=np.full(16, 20.0),
            kv=np.ones(16),
            nworld=2,
        )
        action_map = HandActionMap(
            q[0, :16], np.full(16, 0.1), model.jnt_range[:16, 0], model.jnt_range[:16, 1]
        )
        task = HandReferenceTracking(
            physics,
            action_map,
            reference_qpos=q,
            reference_qvel=v,
            reference_last=np.r_[np.full(300, 299), np.full(300, 599)],
            control_dt=0.0166,
        )
        for frame, last in [(298, 299), (299, 299), (598, 599), (599, 599)]:
            state = task.reset(jax.random.PRNGKey(1), frame)
            h, g = task.policy_obs(state, None)
            self.assertEqual(h.shape, (task.hist_dim,))
            self.assertEqual(g.shape, (task.goal_dim,))
            expected = np.clip(q[last, :16], model.jnt_range[:16, 0], model.jnt_range[:16, 1])
            np.testing.assert_allclose(task.ctrl_of(jnp.zeros(16), jnp.int32(frame)), expected)
            finished = task.finish_physics(
                state, jnp.zeros(16), state.dx.qpos, state.dx.qvel, state.dx.qacc_warmstart
            )
            self.assertEqual(int(finished.frame), last)
            self.assertTrue(bool(task.clip_ended(finished)))

    def test_bootstrap_termination_vs_timeout(self):
        # 真失败不加末端价值；时间到期但未失败，应保留bootstrap。
        reward = jnp.array([[1.0, 1.0]])
        values = td_lambda(
            reward,
            jnp.array([[True, False]]),
            jnp.array([[True, True]]),
            jnp.array([[10.0, 10.0]]),
            0.9,
            0.95,
        )
        np.testing.assert_allclose(values, [[1.0, 10.0]])

    def test_rollout_failure_mask_and_bootstrap(self):
        from unittest.mock import patch
        from hand_training import updates

        class RollState(NamedTuple):
            frame: object
            ep: object
            prev_a: object

        state = RollState(jnp.zeros(2, jnp.int32), jnp.zeros(2, jnp.int32), jnp.zeros((2, 1)))

        def step(s, a):
            return RollState(s.frame + 1, s.ep + 1, a)

        functions = dict(
            policy_obs=lambda s, k: (jnp.zeros((2, 1)), jnp.zeros((2, 1))),
            critic_obs=lambda s: jnp.zeros((2, 1)),
            step=step,
            reward=lambda s, a, previous_action, **kw: (jnp.ones(2), {}),
            terminated=lambda s: (s.ep >= 1) & jnp.array([True, False]),
            clip_ended=lambda s: jnp.zeros(2, bool),
        )
        with patch.object(updates, "vmapped", return_value=functions), patch.object(
            updates, "actor_apply", return_value=jnp.zeros((2, 1))
        ), patch.object(updates, "critic_apply", return_value=jnp.full((2, 1), 2.0)), patch.object(
            updates, "normalize", side_effect=lambda n, x: x
        ):
            rollout = updates.make_rollout(
                SimpleNamespace(), 3, 0.9, 16, False, mask_after_done=True
            )
            loss, _ = rollout(None, None, None, state, state, jax.random.PRNGKey(1), None)
            expected = (1 + (1 + 0.9 + 0.9**2 + 0.9**3 * 2)) / 2
            self.assertAlmostEqual(float(loss), -expected)
            # 两个环境都仅时间到期时，第一步奖励后仍应补估价值。
            functions["terminated"] = lambda s: jnp.zeros(2, bool)
            rollout = updates.make_rollout(
                SimpleNamespace(), 3, 0.9, 1, False, mask_after_done=True
            )
            loss, _ = rollout(None, None, None, state, state, jax.random.PRNGKey(1), None)
            self.assertAlmostEqual(float(loss), -(1 + 0.9 * 2))

    def test_signature_covers_mode_data_and_source(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src").mkdir()
            (root / "datasets/dextrack").mkdir(parents=True)
            (root / "assets").mkdir()
            (root / "assets/allegro_plastic_cube.xml").write_text("model")
            code = root / "src/train.py"
            code.write_text("before")
            args = SimpleNamespace(
                pods=True,
                horizon=8,
                feasible_only=False,
                velocity_feedforward=False,
                feasibility_report=None,
            )
            first = run_signature(args, root)
            code.write_text("after")
            self.assertNotEqual(first, run_signature(args, root))
            args.pods = False
            self.assertNotEqual(first, run_signature(args, root))

    def test_resume_rejects_changed_training_batch(self):
        # 环境数改变后，状态池和PODS缓存的数组形状不同，禁止混用旧断点。
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src").mkdir()
            (root / "datasets/dextrack").mkdir(parents=True)
            (root / "assets").mkdir()
            (root / "assets/allegro_plastic_cube.xml").write_text("model")
            args = SimpleNamespace(
                pods=True,
                horizon=8,
                feasible_only=False,
                velocity_feedforward=False,
                feasibility_report=None,
                num_envs=2,
            )
            signature = run_signature(args, root)
            output = root / "run"
            validate_run_directory(output, signature, resume=False)
            save_latest(output / "checkpoint_latest.pkl", (np.array([1]),), [{"round": 1}])
            args.num_envs = 4
            with self.assertRaises(ValueError):
                validate_run_directory(output, run_signature(args, root), resume=True)


if __name__ == "__main__":
    unittest.main()
