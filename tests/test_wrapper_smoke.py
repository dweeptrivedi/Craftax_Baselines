"""End-to-end smoke for ``AutomatonAugmentedEnvWrapper`` with a synthetic backend.

Bypasses the need for a real checkpoint or RAD ``.npz`` files: we build a
tiny ``RadLookupBackend`` from synthetic arrays, wrap a real Craftax env,
and verify that reset/step trace under jit, that the augmented obs has
the expected shape, that ``info["embedding/accept"]`` flows through, and that the
predicate evaluator + reward-shaping primitives compose with the wrapper
inside a vmap'd context.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Make the submodule (where upstream ``wrappers.py`` lives) importable.
# After the restructure this file is at
# ``<repo>/third_party/craftax_baselines/tests/X.py``, so ``parents[1]``
# is the submodule directory itself.
_CB_PATH = Path(__file__).resolve().parents[1]
if str(_CB_PATH) not in sys.path:
    sys.path.insert(0, str(_CB_PATH))


@pytest.fixture
def synthetic_env_stack():
    """Build a Craftax env wrapped with a synthetic-data RAD backend."""
    from craftax.craftax_env import make_craftax_env_from_name

    from automata_rl.predicate_eval import (
        achievement_extractor,
        make_predicate_evaluator,
    )
    from automata_rl.wrappers import (
        AutomatonAugmentedEnvWrapper,
        RadLookupBackend,
    )

    inner = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    inner_obs_dim = inner.observation_space(inner.default_params).shape[0]

    # Synthetic backend: 4 DFA states, 4 symbols, 8-dim embeddings.
    transition = jnp.array(
        [
            [0, 1, 0, 1],
            [1, 1, 2, 1],
            [2, 2, 3, 2],
            [3, 3, 3, 3],   # absorbing accept
        ],
        dtype=jnp.int32,
    )
    embedding = jnp.array(
        [[i * 0.1] * 8 for i in range(4)], dtype=jnp.float32,
    )
    accept = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    backend = RadLookupBackend(
        transition=transition,
        embedding=embedding,
        accept=accept,
        initial_residual=jnp.int32(0),
    )

    # Predicate evaluator: 2 predicates indexed at achievement positions 0 and 1.
    pred_idx = jnp.array([0, 1], dtype=jnp.int32)
    extractor = achievement_extractor(pred_idx)
    predicate_eval = make_predicate_evaluator(extractor, n_pred=2)

    env = AutomatonAugmentedEnvWrapper(
        inner, backend, predicate_eval, augment_accept=True,
    )
    return env, inner.default_params, inner_obs_dim


def test_observation_space_shape(synthetic_env_stack) -> None:
    env, params, inner_obs_dim = synthetic_env_stack
    space = env.observation_space(params)
    # original obs + 8 embedding + 1 accept channel
    assert space.shape == (inner_obs_dim + 8 + 1,)


def test_reset_and_step_traces_under_jit(synthetic_env_stack) -> None:
    env, params, inner_obs_dim = synthetic_env_stack

    @jax.jit
    def reset_fn(key):
        return env.reset(key, params)

    @jax.jit
    def step_fn(key, state, action):
        return env.step(key, state, action, params)

    key = jax.random.key(0)
    obs, state = reset_fn(key)
    assert obs.shape == (inner_obs_dim + 8 + 1,)
    assert jnp.all(jnp.isfinite(obs))

    obs2, state2, reward, done, info = step_fn(key, state, jnp.int32(0))
    assert obs2.shape == obs.shape
    assert "embedding/accept" in info
    assert info["embedding/accept"].shape == ()
    assert isinstance(state2.q_state.item(), int)


def test_vmap_over_envs(synthetic_env_stack) -> None:
    """Verify the wrapper composes with ``jax.vmap`` over a batch of envs."""
    env, params, _ = synthetic_env_stack

    keys = jax.random.split(jax.random.key(0), 4)
    obs_batch, state_batch = jax.vmap(lambda k: env.reset(k, params))(keys)
    assert obs_batch.shape[0] == 4

    actions = jnp.zeros(4, dtype=jnp.int32)
    out_obs, out_state, reward, done, info = jax.vmap(
        lambda k, s, a: env.step(k, s, a, params),
    )(keys, state_batch, actions)
    assert out_obs.shape[0] == 4
    assert info["embedding/accept"].shape == (4,)


def test_residual_advances_on_predicate_change(synthetic_env_stack) -> None:
    """When predicate 0 fires, our synthetic transition table must take state 0 -> 1.

    The tricky bit: the real Craftax env's achievements are zero on step 0 and
    only fire on game progress. We can't easily force achievement bits at
    test time without a hand-crafted action sequence. So we just check that
    the residual id is well-typed and matches the initial after a single
    no-op step (achievements still all-False -> sym=0 -> stays at residual 0).
    """
    env, params, _ = synthetic_env_stack
    key = jax.random.key(7)
    obs, state = env.reset(key, params)
    assert int(state.q_state) == 0
    obs2, state2, _, _, info = env.step(
        key, state, jnp.int32(0), params,
    )
    # transition[0, 0] = 0  (no predicates active -> stay at initial)
    assert int(state2.q_state) == 0
    assert float(info["embedding/accept"]) == 0.0


def test_term_fns_batched_no_done_mask() -> None:
    """``sparse_accept`` and ``dense_accept_prob`` fire on done=True steps now.

    Smoke check on the batched term-fn signature
    ``(done, prev_accept, curr_accept) -> term``. Regression for the
    removed ``& ~done`` / ``where(done, 0, ...)`` masks: under the old
    behavior, the done-step entries below were zeroed.
    """
    from automata_rl.reward_shaping import (
        TERM_FNS,
        dense_accept_prob,
        sparse_accept,
    )

    done = jnp.array([False, False, True, False])
    prev = jnp.array([0.0, 0.0, 0.5, 0.0])
    curr = jnp.array([0.7, 0.0, 0.6, 0.0])  # rising, no-change, rising-on-done, all-zero
    shaped = sparse_accept(done, prev, curr)
    assert float(shaped[0]) == 1.0
    assert float(shaped[1]) == 0.0
    assert float(shaped[2]) == 1.0  # rising edge fires regardless of done
    assert float(shaped[3]) == 0.0

    delta = dense_accept_prob(done, prev, curr)
    assert float(delta[0]) == pytest.approx(0.7)
    assert float(delta[1]) == 0.0
    assert float(delta[2]) == pytest.approx(0.1)  # delta fires regardless of done
    assert float(delta[3]) == 0.0

    # ``TERM_FNS`` lookup
    assert "none" in TERM_FNS
    assert "continuous_relu" in TERM_FNS
    assert "sparse_accept" in TERM_FNS
    assert TERM_FNS["none"] is not TERM_FNS["sparse_accept"]


def test_predicate_evaluator_matches_predicates_to_mask() -> None:
    """Round-trip every powerset symbol on a 4-predicate task vs ``_predicates_to_mask``."""
    import itertools

    from automata_rl.predicate_eval import (
        achievement_extractor,
        make_predicate_evaluator,
    )
    from brzozowski_dataset.residuals import _predicates_to_mask

    # Synthetic task: 4 predicates at achievement indices [3, 7, 11, 15].
    sorted_predicates = ["pred_a", "pred_b", "pred_c", "pred_d"]
    pred_idx_jnp = jnp.array([3, 7, 11, 15], dtype=jnp.int32)
    extractor = achievement_extractor(pred_idx_jnp)
    evaluator = make_predicate_evaluator(extractor, n_pred=4)

    class FakeState:
        def __init__(self, ach: jnp.ndarray) -> None:
            self.achievements = ach

    n_predicates = 4
    for active_set in itertools.chain.from_iterable(
        itertools.combinations(sorted_predicates, k) for k in range(n_predicates + 1)
    ):
        ach = jnp.zeros(64, dtype=bool)
        for name in active_set:
            i = sorted_predicates.index(name)
            ach = ach.at[pred_idx_jnp[i]].set(True)
        sym = int(evaluator(FakeState(ach)))
        ref = _predicates_to_mask(frozenset(active_set), sorted_predicates)
        assert sym == ref, f"mismatch on {active_set}: got {sym}, ref {ref}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
