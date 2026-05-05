"""Integration tests for ``automata_rl.embedding_setup.build_embedding_stack``.

Tests are guarded with ``pytest.mark.skipif`` against the on-disk artefact
files; they pass automatically (skipped) until you've run
``scripts/convert_brzozowski_to_jax.py``. RAD-lookup tests skip until
``outputs/lookup_rad.npz`` exists (requires MONA system-wide for the
build step).
"""

from __future__ import annotations

from pathlib import Path

import pytest


# This file is at ``<repo>/third_party/craftax_baselines/tests/X.py`` after
# the restructure. ``parents[3]`` is the repo root.
_REPO = Path(__file__).resolve().parents[3]
_BRZ_PARAMS = _REPO / "outputs" / "brzozowski_jax_params.msgpack"
_BRZ_CONFIG = _REPO / "outputs" / "brzozowski_jax_config.yaml"
_BRZ_EVAL_POINTS = _REPO / "outputs" / "brzozowski_jax_eval_points.npy"
_TASK_PREDICATES = _REPO / "outputs" / "task_predicates.json"
_RAD_LOOKUP = _REPO / "outputs" / "lookup_rad.npz"


def _brz_config(target: str = "collect_wood") -> dict:
    return {
        "TARGET_ACHIEVEMENT": target,
        "TASK_PREDICATES_PATH": str(_TASK_PREDICATES),
        "EMBEDDING_KIND": "brzozowski_jax",
        "BRZOZOWSKI_PARAMS_PATH": str(_BRZ_PARAMS),
        "BRZOZOWSKI_CONFIG_PATH": str(_BRZ_CONFIG),
        "BRZOZOWSKI_EVAL_POINTS_PATH": str(_BRZ_EVAL_POINTS),
        "REWARD_SHAPING": "none",
    }


@pytest.fixture
def craftax_inner():
    from craftax.craftax_env import make_craftax_env_from_name

    return make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)


@pytest.mark.skipif(
    not (_BRZ_PARAMS.exists() and _BRZ_CONFIG.exists() and _BRZ_EVAL_POINTS.exists()),
    reason="Brzozowski JAX artefacts not built (run scripts/convert_brzozowski_to_jax.py).",
)
def test_brzozowski_jax_stack_observation_shape(craftax_inner) -> None:
    """``build_embedding_stack`` returns env whose obs shape is inner + embed_dim + 1."""
    from automata_rl.embedding_setup import build_embedding_stack

    inner_obs = craftax_inner.observation_space(craftax_inner.default_params).shape[0]
    env, split = build_embedding_stack(craftax_inner, _brz_config())

    aug_shape = env.observation_space(env.default_params).shape
    # eval_points has shape (M, num_vars), so embed_dim = M = 512 for our model.
    assert aug_shape[0] == inner_obs + 512 + 1
    assert split == inner_obs


@pytest.mark.skipif(
    not (_BRZ_PARAMS.exists() and _TASK_PREDICATES.exists()),
    reason="Brzozowski artefacts or task_predicates not built.",
)
def test_unknown_task_raises(craftax_inner) -> None:
    """An unknown task name should raise ``ValueError`` listing available tasks."""
    from automata_rl.embedding_setup import build_embedding_stack

    cfg = _brz_config(target="not_a_real_task")
    with pytest.raises(ValueError, match="not in"):
        build_embedding_stack(craftax_inner, cfg)


@pytest.mark.skipif(
    not (_BRZ_PARAMS.exists() and _BRZ_CONFIG.exists() and _BRZ_EVAL_POINTS.exists()),
    reason="Brzozowski JAX artefacts not built.",
)
def test_unknown_embedding_kind_raises(craftax_inner) -> None:
    from automata_rl.embedding_setup import build_embedding_stack

    cfg = _brz_config()
    cfg["EMBEDDING_KIND"] = "made_up_backend"
    with pytest.raises(ValueError, match="Unknown EMBEDDING_KIND"):
        build_embedding_stack(craftax_inner, cfg)


@pytest.mark.skipif(
    not (_BRZ_PARAMS.exists() and _BRZ_CONFIG.exists()),
    reason="Brzozowski JAX artefacts not built.",
)
def test_dense_shaping_with_rad_lookup_rejected(craftax_inner) -> None:
    """``REWARD_SHAPING=dense_accept_prob`` is incompatible with ``rad_lookup``."""
    from automata_rl.embedding_setup import build_embedding_stack

    cfg = _brz_config()
    cfg["EMBEDDING_KIND"] = "rad_lookup"
    cfg["RAD_LOOKUP_PATH"] = str(_RAD_LOOKUP)
    cfg["REWARD_SHAPING"] = "dense_accept_prob"
    cfg["REWARD_SHAPING_SCALE"] = 1.0
    # The RAD-lookup branch never runs for this assertion -- the
    # ``dense_accept_prob + rad_lookup`` cross-validation is at the END of
    # build_embedding_stack, so we'd need the lookup file to exist for the
    # call to reach that check. Skip if it doesn't.
    if not _RAD_LOOKUP.exists():
        pytest.skip("lookup_rad.npz not built (requires MONA).")
    with pytest.raises(ValueError, match="dense_accept_prob.*rad_lookup"):
        build_embedding_stack(craftax_inner, cfg)


@pytest.mark.skipif(
    not (_BRZ_PARAMS.exists() and _BRZ_CONFIG.exists() and _BRZ_EVAL_POINTS.exists()),
    reason="Brzozowski JAX artefacts not built.",
)
def test_shaping_wrapper_chained(craftax_inner) -> None:
    """When ``REWARD_SHAPING != none``, the returned env should advertise ``info["accept"]``."""
    import jax
    import jax.numpy as jnp

    from automata_rl.embedding_setup import build_embedding_stack

    cfg = _brz_config()
    cfg["REWARD_SHAPING"] = "sparse_accept"
    env, _ = build_embedding_stack(craftax_inner, cfg)

    key = jax.random.key(0)
    params = env.default_params
    obs, state = env.reset(key, params)
    obs2, state2, reward, done, info = env.step(key, state, jnp.int32(0), params)
    assert "accept" in info
    # Shaping wrapper carries prev_accept on its own state field.
    assert hasattr(state2, "prev_accept")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
