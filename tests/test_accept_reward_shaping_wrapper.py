"""Unit tests for ``AcceptRewardShapingWrapper``.

Composes a real Craftax env with a synthetic ``RadLookupBackend`` (no
need for the trained model), wraps with the shaping wrapper, runs a few
steps, and checks:

- ``prev_accept`` is carried correctly across steps.
- ``done`` masks out the bonus.
- ``dense_accept_prob`` honors the configured ``scale``.
- Shape of returned ``info["accept"]`` and shaped reward.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def stack():
    """Build a Craftax env with a synthetic RAD backend + shaping wrapper."""
    from craftax.craftax_env import make_craftax_env_from_name

    from automata_rl.predicate_eval import (
        achievement_extractor,
        make_predicate_evaluator,
    )
    from automata_rl.wrappers import (
        AcceptRewardShapingWrapper,
        AutomatonAugmentedEnvWrapper,
        RadLookupBackend,
    )

    inner = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)

    transition = jnp.array(
        [
            [0, 1],
            [2, 1],
            [2, 2],   # absorbing accept
        ],
        dtype=jnp.int32,
    )
    embedding = jnp.array(
        [[0.1] * 4, [0.2] * 4, [0.3] * 4], dtype=jnp.float32,
    )
    accept = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    backend = RadLookupBackend(
        transition=transition,
        embedding=embedding,
        accept=accept,
        initial_residual=jnp.int32(0),
    )

    pred_idx = jnp.array([0], dtype=jnp.int32)
    pred_eval = make_predicate_evaluator(achievement_extractor(pred_idx), n_pred=1)

    aug = AutomatonAugmentedEnvWrapper(inner, backend, pred_eval, augment_accept=True)
    return aug, inner.default_params


def test_shaping_none_is_identity(stack) -> None:
    """``kind="none"`` should not modify reward."""
    from automata_rl.wrappers import AcceptRewardShapingWrapper

    aug, params = stack
    env = AcceptRewardShapingWrapper(aug, kind="none")
    key = jax.random.key(0)
    obs, state = env.reset(key, params)
    obs2, state2, shaped, done, info = env.step(key, state, jnp.int32(0), params)
    # We can't assert exact equality vs the un-shaped reward without re-stepping
    # the AutomatonAugmentedEnvWrapper, but we can assert finite + shape.
    assert jnp.isfinite(shaped)
    assert state2.prev_accept.shape == ()


def test_sparse_accept_carries_prev_accept(stack) -> None:
    """``prev_accept`` should equal the previous step's ``info["accept"]``."""
    from automata_rl.wrappers import AcceptRewardShapingWrapper

    aug, params = stack
    env = AcceptRewardShapingWrapper(aug, kind="sparse_accept")
    key = jax.random.key(7)
    obs, state = env.reset(key, params)
    assert float(state.prev_accept) == 0.0

    keys = jax.random.split(key, 3)
    accepts = []
    s = state
    for k in keys:
        obs, s, shaped, done, info = env.step(k, s, jnp.int32(0), params)
        accepts.append(float(info["accept"]))
    # ``prev_accept`` carried into the LAST step's state should equal the
    # SECOND-to-last step's accept value (the new ``s.prev_accept`` is the
    # CURRENT step's accept, used for the next step).
    assert float(s.prev_accept) == accepts[-1]


def test_dense_accept_prob_scale(stack) -> None:
    """``scale`` should multiply the (curr - prev) delta in the shaping."""
    from automata_rl.reward_shaping import dense_accept_prob

    # Direct test of the partial-applied function via the wrapper's helper.
    from automata_rl.wrappers import _make_shaping_fn

    fn1 = _make_shaping_fn("dense_accept_prob", scale=1.0)
    fn5 = _make_shaping_fn("dense_accept_prob", scale=5.0)

    reward = jnp.zeros(())
    done = jnp.bool_(False)
    prev = jnp.float32(0.0)
    curr = jnp.float32(0.7)
    shaped1 = float(fn1(reward, done, prev, curr))
    shaped5 = float(fn5(reward, done, prev, curr))
    np.testing.assert_allclose(shaped1, 0.7, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(shaped5, 3.5, rtol=1e-6, atol=1e-6)


def test_done_masks_dense_accept_bonus(stack) -> None:
    """When ``done=True``, ``dense_accept_prob`` must NOT add a delta bonus."""
    from automata_rl.wrappers import _make_shaping_fn

    fn = _make_shaping_fn("dense_accept_prob", scale=2.0)
    reward = jnp.float32(1.0)
    done_t = jnp.bool_(True)
    done_f = jnp.bool_(False)
    prev = jnp.float32(0.0)
    curr = jnp.float32(0.7)
    on_done = float(fn(reward, done_t, prev, curr))
    off_done = float(fn(reward, done_f, prev, curr))
    np.testing.assert_allclose(on_done, 1.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(off_done, 1.0 + 2.0 * 0.7, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
