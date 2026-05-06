"""Tests for the Hydra config tree under ``conf/``.

These exercises composition only -- no full training run. They
validate group selection, the per-embedding default-reward override,
the ``experiment/*.yaml`` matrix, the ``train.py`` post-load
adjustments (uppercasing, total-timesteps cast, seed fill, wandb run
name, cross-cutting validation), and Hydra's strict-mode rejection
of unknown keys (which preserves today's argparse "unknown args"
behavior).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


_CB_PATH = Path(__file__).resolve().parents[1]
_CONF_DIR = _CB_PATH / "conf"
_EXPERIMENT_DIR = _CONF_DIR / "experiment"

if str(_CB_PATH) not in sys.path:
    sys.path.insert(0, str(_CB_PATH))


# Expected (embedding_kind, env_reward_coef, accept_reward_coef,
# accept_reward_kind) for each experiment.
_EXPECTED_EXPERIMENT_BUNDLE: dict[str, tuple[str, float, float, str]] = {
    "none_env_only": ("none", 1.0, 0.0, "none"),
    "brzozowski_continuous": ("brzozowski_jax", 0.0, 1.0, "continuous"),
    "brzozowski_continuous_relu": (
        "brzozowski_jax",
        0.0,
        1.0,
        "continuous_relu",
    ),
    "brzozowski_env_only": ("brzozowski_jax", 1.0, 0.0, "none"),
    "rad_continuous": ("rad_lookup", 0.0, 1.0, "continuous"),
    "rad_continuous_relu": ("rad_lookup", 0.0, 1.0, "continuous_relu"),
    "rad_env_only": ("rad_lookup", 1.0, 0.0, "none"),
}


def _compose(overrides: list[str] | None = None) -> dict[str, Any]:
    """Compose the root config with optional CLI-style overrides."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        cfg = compose(config_name="config", overrides=overrides or [])
    return OmegaConf.to_container(cfg, resolve=True)


# ---------------------------------------------------------------------------
# Default + group composition
# ---------------------------------------------------------------------------


def test_default_compose() -> None:
    """No overrides yields the env-only baseline (today's PPO defaults)."""
    cfg = _compose()
    assert cfg["embedding_kind"] == "none"
    assert cfg["env_reward_coef"] == 1.0
    assert cfg["accept_reward_coef"] == 0.0
    assert cfg["accept_reward_kind"] == "none"
    assert cfg["variant_name"] == "ppo"
    assert cfg["run_target"] == "ppo.run_ppo"


def test_embedding_brzozowski_flips_reward_default() -> None:
    cfg = _compose(["embedding=brzozowski"])
    assert cfg["embedding_kind"] == "brzozowski_jax"
    assert cfg["env_reward_coef"] == 0.0
    assert cfg["accept_reward_coef"] == 1.0
    assert cfg["accept_reward_kind"] == "continuous"


def test_embedding_rad_flips_reward_default() -> None:
    cfg = _compose(["embedding=rad"])
    assert cfg["embedding_kind"] == "rad_lookup"
    assert cfg["accept_reward_kind"] == "continuous"


def test_user_reward_override_wins() -> None:
    """CLI ``reward=...`` beats the embedding's default."""
    cfg = _compose(["embedding=brzozowski", "reward=accept_sparse"])
    assert cfg["embedding_kind"] == "brzozowski_jax"
    assert cfg["accept_reward_kind"] == "sparse_accept"


def test_mixture_composable() -> None:
    """Strict-pairing asserts are gone: mixtures compose without error."""
    cfg = _compose(["reward=accept_continuous", "env_reward_coef=1.0"])
    assert cfg["env_reward_coef"] == 1.0
    assert cfg["accept_reward_coef"] == 1.0
    assert cfg["accept_reward_kind"] == "continuous"


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------


def test_experiment_glob_count() -> None:
    """``conf/experiment/`` has exactly the seven YAMLs listed in the matrix."""
    yamls = sorted(p.stem for p in _EXPERIMENT_DIR.glob("*.yaml"))
    assert yamls == sorted(_EXPECTED_EXPERIMENT_BUNDLE.keys())


@pytest.mark.parametrize("name", sorted(_EXPECTED_EXPERIMENT_BUNDLE.keys()))
def test_experiment_compositions(name: str) -> None:
    """Each experiment YAML composes the documented bundle."""
    cfg = _compose([f"experiment={name}"])
    embedding, env_coef, acc_coef, kind = _EXPECTED_EXPERIMENT_BUNDLE[name]
    assert cfg["embedding_kind"] == embedding
    assert cfg["env_reward_coef"] == env_coef
    assert cfg["accept_reward_coef"] == acc_coef
    assert cfg["accept_reward_kind"] == kind
    assert cfg["experiment_name"] == name


def test_sweep_cardinality() -> None:
    """The 21-cell sweep yields 21 distinct configurations."""
    seen: set[tuple] = set()
    for variant in ("ppo", "ppo_rnn", "ppo_rnd"):
        for experiment in _EXPECTED_EXPERIMENT_BUNDLE:
            cfg = _compose([f"variant={variant}", f"experiment={experiment}"])
            cell = (
                cfg["variant_name"],
                cfg["embedding_kind"],
                cfg["env_reward_coef"],
                cfg["accept_reward_coef"],
                cfg["accept_reward_kind"],
            )
            seen.add(cell)
    assert len(seen) == 21


# ---------------------------------------------------------------------------
# Variant fields
# ---------------------------------------------------------------------------


def test_variant_ppo_carries_icm_fields() -> None:
    cfg = _compose(["variant=ppo"])
    assert "train_icm" in cfg
    assert "use_e3b" in cfg
    assert "icm_reward_coeff" in cfg
    # ppo_rnn variant lacks them entirely.
    cfg_rnn = _compose(["variant=ppo_rnn"])
    assert "train_icm" not in cfg_rnn
    assert "use_e3b" not in cfg_rnn


def test_variant_ppo_rnd_carries_rnd_fields() -> None:
    cfg = _compose(["variant=ppo_rnd"])
    assert cfg["use_rnd"] is True
    assert cfg["rnd_layer_size"] == 256
    assert cfg["rnd_output_size"] == 512
    assert "train_icm" not in cfg


def test_variant_ppo_rnd_overrides_exploration_update_epochs() -> None:
    cfg = _compose(["variant=ppo_rnd"])
    assert cfg["exploration_update_epochs"] == 1
    cfg_ppo = _compose(["variant=ppo"])
    assert cfg_ppo["exploration_update_epochs"] == 4


def test_exploration_update_epochs_isolated_to_ppo_and_ppo_rnd() -> None:
    """ppo_rnn deliberately omits ``exploration_update_epochs``."""
    cfg = _compose(["variant=ppo_rnn"])
    assert "exploration_update_epochs" not in cfg


def test_target_dispatch_strings() -> None:
    """``run_target`` is a plain dotted-path string per variant."""
    assert _compose(["variant=ppo"])["run_target"] == "ppo.run_ppo"
    assert _compose(["variant=ppo_rnn"])["run_target"] == "ppo_rnn.run_ppo"
    assert _compose(["variant=ppo_rnd"])["run_target"] == "ppo_rnd.run_ppo"


# ---------------------------------------------------------------------------
# train.py glue
# ---------------------------------------------------------------------------


def test_uppercasing_round_trip() -> None:
    from train import _to_uppercase_dict

    plain = {
        "env_name": "Craftax-Symbolic-v1",
        "lr": 2.0e-4,
        "embedding_kind": "none",
        "env_reward_coef": 1.0,
        "gamma": 0.99,
    }
    upper = _to_uppercase_dict(plain)
    assert upper["ENV_NAME"] == "Craftax-Symbolic-v1"
    assert upper["LR"] == 2.0e-4
    assert upper["EMBEDDING_KIND"] == "none"
    assert upper["ENV_REWARD_COEF"] == 1.0
    assert upper["GAMMA"] == 0.99


def test_wandb_run_name_includes_experiment() -> None:
    from train import _build_wandb_run_name

    base_config = {
        "ENV_NAME": "Craftax-Symbolic-v1",
        "VARIANT_NAME": "ppo",
        "TOTAL_TIMESTEPS": 1_000_000_000,
        "TARGET_ACHIEVEMENT": "place_table",
        "EMBEDDING_KIND": "brzozowski_jax",
        "EXPERIMENT_NAME": None,
    }
    # No experiment: byte-identical to today's argparse format.
    name_no_exp = _build_wandb_run_name(base_config)
    assert name_no_exp == (
        "Craftax-Symbolic-v1-PPO-1000M-place_table-brzozowski_jax"
    )

    # With experiment: suffix appended.
    cfg_with = {**base_config, "EXPERIMENT_NAME": "brzozowski_continuous"}
    name_with = _build_wandb_run_name(cfg_with)
    assert name_with == name_no_exp + "-brzozowski_continuous"

    # PPO_RNN / PPO_RND variants get the right uppercase prefix.
    cfg_rnn = {**base_config, "VARIANT_NAME": "ppo_rnn"}
    assert "-PPO_RNN-" in _build_wandb_run_name(cfg_rnn)
    cfg_rnd = {**base_config, "VARIANT_NAME": "ppo_rnd"}
    assert "-PPO_RND-" in _build_wandb_run_name(cfg_rnd)

    # Default target tag.
    cfg_no_target = {**base_config, "TARGET_ACHIEVEMENT": None}
    assert "-default-" in _build_wandb_run_name(cfg_no_target)


def test_total_timesteps_cast_to_int() -> None:
    """YAML scientific notation ``1.0e9`` must arrive as ``int`` for ``//`` math."""
    from train import _to_uppercase_dict

    cfg_raw = _compose([])  # has total_timesteps as float (1e9)
    cfg_upper = _to_uppercase_dict(cfg_raw)
    cfg_upper["TOTAL_TIMESTEPS"] = int(float(cfg_upper["TOTAL_TIMESTEPS"]))
    assert isinstance(cfg_upper["TOTAL_TIMESTEPS"], int)
    assert cfg_upper["TOTAL_TIMESTEPS"] == 1_000_000_000


def test_seed_null_randomized() -> None:
    """``seed=null`` becomes a fresh int via ``np.random.randint``."""
    import numpy as np
    from train import _to_uppercase_dict

    cfg_raw = _compose([])
    cfg = _to_uppercase_dict(cfg_raw)
    assert cfg.get("SEED") is None
    cfg["SEED"] = int(np.random.randint(2**31))
    assert isinstance(cfg["SEED"], int)
    assert 0 <= cfg["SEED"] < 2**31


def test_nonfinite_coef_rejected() -> None:
    from train import _validate_cross_cutting

    cfg_nan = {
        "ENV_REWARD_COEF": float("nan"),
        "ACCEPT_REWARD_COEF": 0.0,
    }
    with pytest.raises(AssertionError, match="finite"):
        _validate_cross_cutting(cfg_nan)

    cfg_inf = {
        "ENV_REWARD_COEF": 1.0,
        "ACCEPT_REWARD_COEF": float("inf"),
    }
    with pytest.raises(AssertionError, match="finite"):
        _validate_cross_cutting(cfg_inf)


def test_e3b_without_icm_rejected() -> None:
    from train import _validate_cross_cutting

    cfg = {
        "ENV_REWARD_COEF": 1.0,
        "ACCEPT_REWARD_COEF": 0.0,
        "USE_E3B": True,
        "TRAIN_ICM": False,
        "ICM_REWARD_COEFF": 0.0,
    }
    with pytest.raises(AssertionError, match="TRAIN_ICM"):
        _validate_cross_cutting(cfg)


def test_e3b_with_icm_but_nonzero_coeff_rejected() -> None:
    from train import _validate_cross_cutting

    cfg = {
        "ENV_REWARD_COEF": 1.0,
        "ACCEPT_REWARD_COEF": 0.0,
        "USE_E3B": True,
        "TRAIN_ICM": True,
        "ICM_REWARD_COEFF": 1.0,  # must be 0
    }
    with pytest.raises(AssertionError, match="ICM_REWARD_COEFF"):
        _validate_cross_cutting(cfg)


# ---------------------------------------------------------------------------
# Hydra strict-mode behaviors
# ---------------------------------------------------------------------------


def test_unknown_arg_rejected_by_hydra() -> None:
    """Hydra rejects unknown CLI keys (matches today's argparse strictness)."""
    from hydra.errors import ConfigCompositionException

    with pytest.raises((ConfigCompositionException, Exception)):
        _compose(["nonexistent_field=42"])


def test_variant_flag_isolation_train_icm() -> None:
    """``train_icm`` only available with ``variant=ppo``."""
    from hydra.errors import ConfigCompositionException

    with pytest.raises((ConfigCompositionException, Exception)):
        _compose(["variant=ppo_rnn", "train_icm=true"])


def test_variant_flag_isolation_use_rnd() -> None:
    """``use_rnd`` only available with ``variant=ppo_rnd``."""
    from hydra.errors import ConfigCompositionException

    with pytest.raises((ConfigCompositionException, Exception)):
        _compose(["variant=ppo", "use_rnd=true"])


def test_variant_flag_isolation_exploration_update_epochs_on_ppo_rnn() -> None:
    """``exploration_update_epochs`` not available under ``variant=ppo_rnn``."""
    from hydra.errors import ConfigCompositionException

    with pytest.raises((ConfigCompositionException, Exception)):
        _compose(["variant=ppo_rnn", "exploration_update_epochs=8"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
