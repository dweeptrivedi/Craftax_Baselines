"""Unit tests for ``RewardCompositionWrapper`` and the term functions.

The wrapper computes ``reward = env_coef * env_reward + accept_coef *
accept_term``. Tests cover:

- Each term function in :mod:`automata_rl.reward_shaping` (continuous,
  continuous_relu, sparse_accept, dense_accept_prob, none).
- The linear-combination contract.
- Info-key population (``task/env_reward``, ``task/accept_term``,
  ``task/reward``).
- ``prev_accept`` reset on ``reset``.
- ``vmap`` / ``jit`` composition.
- The post-fix invariant that ``info["embedding/accept"]`` carries the
  live pre-reset accept on ``done=True`` (not ``init_accept``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Make the submodule importable.
_CB_PATH = Path(__file__).resolve().parents[1]
if str(_CB_PATH) not in sys.path:
    sys.path.insert(0, str(_CB_PATH))


# ---------------------------------------------------------------------------
# Term-function tests (pure, no wrapper / env)
# ---------------------------------------------------------------------------


def test_term_fn_continuous() -> None:
    """``continuous`` returns the raw curr_accept value."""
    from automata_rl.reward_shaping import continuous

    for x in (0.0, 0.0008, 0.5, 0.999, 1.0):
        out = float(continuous(jnp.bool_(False), jnp.float32(0.0), jnp.float32(x)))
        np.testing.assert_allclose(out, x, rtol=1e-6, atol=1e-7)


def test_term_fn_continuous_relu() -> None:
    """``continuous_relu`` zeroes below the 0.1 threshold; identity at/above."""
    from automata_rl.reward_shaping import continuous_relu

    cases = {0.0: 0.0, 0.05: 0.0, 0.0999: 0.0, 0.1: 0.1, 0.5: 0.5, 1.0: 1.0}
    for x, expected in cases.items():
        out = float(continuous_relu(
            jnp.bool_(False), jnp.float32(0.0), jnp.float32(x),
        ))
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)


def test_term_fn_sparse_accept_no_done_mask() -> None:
    """``sparse_accept`` fires on rising edge regardless of ``done``.

    Regression test for the removed ``& ~done`` mask. Under the old
    masked behavior, a successful done-on-rising-edge step (typical
    when ``--task_terminate_on_complete=True`` fires on the
    achievement-acquisition step) returned 0.0 instead of 1.0.
    """
    from automata_rl.reward_shaping import sparse_accept

    # Rising edge with done=True (success step).
    out = float(sparse_accept(
        jnp.bool_(True), jnp.float32(0.2), jnp.float32(0.9),
    ))
    np.testing.assert_allclose(out, 1.0, rtol=1e-6, atol=1e-7)

    # Rising edge with done=False.
    out = float(sparse_accept(
        jnp.bool_(False), jnp.float32(0.2), jnp.float32(0.9),
    ))
    np.testing.assert_allclose(out, 1.0, rtol=1e-6, atol=1e-7)

    # No rising edge (already accepting).
    out = float(sparse_accept(
        jnp.bool_(False), jnp.float32(0.7), jnp.float32(0.8),
    ))
    np.testing.assert_allclose(out, 0.0, rtol=1e-6, atol=1e-7)

    # No rising edge (still below threshold).
    out = float(sparse_accept(
        jnp.bool_(False), jnp.float32(0.1), jnp.float32(0.3),
    ))
    np.testing.assert_allclose(out, 0.0, rtol=1e-6, atol=1e-7)


def test_term_fn_dense_accept_prob_no_done_mask() -> None:
    """``dense_accept_prob`` returns ``curr - prev`` regardless of ``done``.

    Regression test for the removed ``where(done, 0, delta)`` mask.
    """
    from automata_rl.reward_shaping import dense_accept_prob

    # Positive delta with done=True (success step).
    out = float(dense_accept_prob(
        jnp.bool_(True), jnp.float32(0.0), jnp.float32(1.0),
    ))
    np.testing.assert_allclose(out, 1.0, rtol=1e-6, atol=1e-7)

    # Negative delta (model lost confidence).
    out = float(dense_accept_prob(
        jnp.bool_(False), jnp.float32(0.7), jnp.float32(0.3),
    ))
    np.testing.assert_allclose(out, -0.4, rtol=1e-6, atol=1e-7)

    # No change.
    out = float(dense_accept_prob(
        jnp.bool_(True), jnp.float32(0.5), jnp.float32(0.5),
    ))
    np.testing.assert_allclose(out, 0.0, rtol=1e-6, atol=1e-7)


def test_term_fn_none() -> None:
    """``none_term`` always returns 0."""
    from automata_rl.reward_shaping import none_term

    out = float(none_term(
        jnp.bool_(True), jnp.float32(0.7), jnp.float32(0.9),
    ))
    np.testing.assert_allclose(out, 0.0, rtol=1e-6, atol=1e-7)


def test_term_fns_dict_complete() -> None:
    """``TERM_FNS`` exposes exactly the five shipped kinds."""
    from automata_rl.reward_shaping import TERM_FNS

    assert set(TERM_FNS) == {
        "none", "sparse_accept", "dense_accept_prob",
        "continuous", "continuous_relu",
    }


# ---------------------------------------------------------------------------
# Wrapper-level tests with a stub inner env
# ---------------------------------------------------------------------------


from flax import struct  # noqa: E402


@struct.dataclass
class _StubState:
    """Tiny pytree state for the stub inner env."""

    step_count: jnp.ndarray  # int32 scalar


class _StubInnerEnv:
    """Minimal env that emits user-controlled ``env_reward``,
    ``info["embedding/accept"]``, and ``done``.

    Records the trajectory pattern at construction time as 1-D arrays of
    the desired per-step values; ``step`` indexes them by step count.
    """

    def __init__(
        self,
        env_rewards: jax.Array,
        accepts: jax.Array,
        dones: jax.Array,
    ) -> None:
        self._env_rewards = jnp.asarray(env_rewards, dtype=jnp.float32)
        self._accepts = jnp.asarray(accepts, dtype=jnp.float32)
        self._dones = jnp.asarray(dones, dtype=jnp.bool_)

    def reset(self, key, params=None):
        del key, params
        obs = jnp.zeros((1,), dtype=jnp.float32)
        return obs, _StubState(step_count=jnp.int32(0))

    def step(self, key, state, action, params=None):
        del key, action, params
        idx = state.step_count
        env_reward = self._env_rewards[idx]
        accept = self._accepts[idx]
        done = self._dones[idx]
        obs = jnp.zeros((1,), dtype=jnp.float32)
        info = {"embedding/accept": accept, "task/reward": env_reward}
        new_state = _StubState(step_count=idx + 1)
        return obs, new_state, env_reward, done, info


def _make_wrapper(stub: _StubInnerEnv, *, kind: str, accept_coef: float, env_coef: float):
    from automata_rl.wrappers import RewardCompositionWrapper

    return RewardCompositionWrapper(
        stub, kind=kind, accept_coef=accept_coef, env_coef=env_coef,
    )


@pytest.mark.parametrize(
    "kind,env_coef,accept_coef,env_reward,prev_accept_in,curr_accept,expected",
    [
        # env-only triple
        ("none", 1.0, 0.0, 0.5, 0.0, 0.4, 0.5),
        # pure-accept continuous
        ("continuous", 0.0, 1.0, 0.5, 0.0, 0.4, 0.4),
        # pure-accept continuous_relu (below threshold)
        ("continuous_relu", 0.0, 1.0, 0.5, 0.0, 0.05, 0.0),
        # pure-accept continuous_relu (at/above threshold)
        ("continuous_relu", 0.0, 1.0, 0.5, 0.0, 0.6, 0.6),
        # mixture: env + sparse_accept (NB: forbidden in production by
        # make_train asserts, but the wrapper itself must remain general).
        ("sparse_accept", 1.0, 1.0, 0.5, 0.0, 0.9, 1.5),
        # mixture: env + dense_accept_prob with non-zero prev_accept
        ("dense_accept_prob", 1.0, 1.0, 0.5, 0.2, 0.7, 1.0),
    ],
)
def test_wrapper_emits_linear_combination(
    kind: str,
    env_coef: float,
    accept_coef: float,
    env_reward: float,
    prev_accept_in: float,
    curr_accept: float,
    expected: float,
) -> None:
    """``reward == env_coef * env_reward + accept_coef * accept_term`` for several triples."""
    stub = _StubInnerEnv(
        env_rewards=jnp.array([env_reward]),
        accepts=jnp.array([curr_accept]),
        dones=jnp.array([False]),
    )
    env = _make_wrapper(stub, kind=kind, accept_coef=accept_coef, env_coef=env_coef)

    # Manually seed prev_accept to exercise dense_accept_prob.
    obs, state = env.reset(jax.random.key(0), None)
    state = state.replace(prev_accept=jnp.float32(prev_accept_in))

    obs2, state2, reward, done, info = env.step(
        jax.random.key(0), state, jnp.int32(0), None,
    )
    np.testing.assert_allclose(float(reward), expected, rtol=1e-6, atol=1e-6)


def test_both_info_keys_always_present() -> None:
    """``task/env_reward`` and ``task/accept_term`` populated regardless of coefs."""
    triples = [
        ("none", 1.0, 0.0),
        ("continuous", 0.0, 1.0),
        ("sparse_accept", 1.0, 1.0),
    ]
    for kind, ec, ac in triples:
        stub = _StubInnerEnv(
            env_rewards=jnp.array([0.7]),
            accepts=jnp.array([0.3]),
            dones=jnp.array([False]),
        )
        env = _make_wrapper(stub, kind=kind, accept_coef=ac, env_coef=ec)
        obs, state = env.reset(jax.random.key(0), None)
        obs2, state2, reward, done, info = env.step(
            jax.random.key(0), state, jnp.int32(0), None,
        )
        assert "task/env_reward" in info, f"missing for kind={kind}, coefs=({ec},{ac})"
        assert "task/accept_term" in info, f"missing for kind={kind}, coefs=({ec},{ac})"
        assert "task/reward" in info, f"missing for kind={kind}, coefs=({ec},{ac})"


def test_pre_reset_accept_used_on_done() -> None:
    """On a ``done=True`` step the wrapper must use the live pre-reset accept.

    Regression for the dependency on the ``pre_reset_accept`` fix in
    ``AutomatonAugmentedEnvWrapper``: if that fix were reverted,
    ``info["embedding/accept"]`` would be ``init_accept`` on done and
    the wrapper would emit a near-zero reward on success steps.
    """
    stub = _StubInnerEnv(
        env_rewards=jnp.array([0.0]),
        accepts=jnp.array([1.0]),  # success: accept just rose to 1
        dones=jnp.array([True]),    # success step terminates
    )
    env = _make_wrapper(stub, kind="continuous", accept_coef=1.0, env_coef=0.0)
    obs, state = env.reset(jax.random.key(0), None)
    obs2, state2, reward, done, info = env.step(
        jax.random.key(0), state, jnp.int32(0), None,
    )
    np.testing.assert_allclose(float(reward), 1.0, rtol=1e-6, atol=1e-6)


def test_prev_accept_resets_to_zero() -> None:
    """``reset`` initializes ``prev_accept`` to 0.0; step updates it."""
    stub = _StubInnerEnv(
        env_rewards=jnp.array([0.0, 0.0]),
        accepts=jnp.array([0.4, 0.7]),
        dones=jnp.array([False, False]),
    )
    env = _make_wrapper(stub, kind="continuous", accept_coef=1.0, env_coef=0.0)

    obs, state = env.reset(jax.random.key(0), None)
    np.testing.assert_allclose(float(state.prev_accept), 0.0, rtol=1e-6, atol=1e-7)

    obs, state, _, _, _ = env.step(jax.random.key(0), state, jnp.int32(0), None)
    np.testing.assert_allclose(float(state.prev_accept), 0.4, rtol=1e-6, atol=1e-6)

    obs, state, _, _, _ = env.step(jax.random.key(0), state, jnp.int32(0), None)
    np.testing.assert_allclose(float(state.prev_accept), 0.7, rtol=1e-6, atol=1e-6)


def test_vmap_and_jit() -> None:
    """Wrapper composes through ``vmap`` and ``jit`` over a batch of envs."""
    stub = _StubInnerEnv(
        env_rewards=jnp.array([0.5]),
        accepts=jnp.array([0.7]),
        dones=jnp.array([False]),
    )
    env = _make_wrapper(stub, kind="continuous", accept_coef=1.0, env_coef=0.0)

    keys = jax.random.split(jax.random.key(0), 4)

    def reset_one(k):
        return env.reset(k, None)

    def step_one(k, s):
        return env.step(k, s, jnp.int32(0), None)

    obs_batch, state_batch = jax.jit(jax.vmap(reset_one))(keys)
    out_obs, out_state, reward, done, info = jax.jit(jax.vmap(step_one))(keys, state_batch)

    assert reward.shape == (4,)
    assert jnp.all(jnp.isfinite(reward))
    np.testing.assert_allclose(np.asarray(reward), np.full((4,), 0.7), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_unknown_kind_raises() -> None:
    """Constructor rejects unknown ``kind`` values."""
    from automata_rl.wrappers import RewardCompositionWrapper

    stub = _StubInnerEnv(
        env_rewards=jnp.array([0.0]),
        accepts=jnp.array([0.0]),
        dones=jnp.array([False]),
    )
    with pytest.raises(ValueError, match="Unknown accept_reward_kind"):
        RewardCompositionWrapper(
            stub, kind="not_a_real_kind", accept_coef=1.0, env_coef=0.0,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
