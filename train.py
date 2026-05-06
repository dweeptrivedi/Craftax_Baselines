"""Hydra entry point for the three PPO variants.

Loads the Hydra config tree under ``conf/``, fills in cross-cutting
defaults (random seed, ``WANDB_RUN_NAME``), validates a small set of
constraints that span groups, and dispatches to the variant's
``run_ppo`` function. Trainer-internal logic remains in the variant
files (``ppo.py``, ``ppo_rnn.py``, ``ppo_rnd.py``).
"""
from __future__ import annotations

import importlib
import math
from typing import Any, Callable

import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf


def _resolve_run_target(target: str) -> Callable:
    """Import ``module.attr`` from a dotted-path string."""
    module_name, attr = target.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr)


def _to_uppercase_dict(plain: dict) -> dict:
    """Top-level lowercase keys to uppercase (matches ``make_train`` reads)."""
    return {key.upper(): value for key, value in plain.items()}


def _build_wandb_run_name(config: dict) -> str:
    """Compose the wandb run name, appending ``experiment_name`` when set.

    Format::

        {ENV_NAME}-{VARIANT_NAME_UPPER}-{TIMESTEPS_M}M-{TARGET_TAG}-{EMBEDDING_KIND}

    optionally suffixed with ``-{EXPERIMENT_NAME}`` when an
    ``experiment=`` is selected. With no experiment selected, the
    name is byte-identical to today's argparse-era format
    (``-PPO-`` / ``-PPO_RNN-`` / ``-PPO_RND-`` come from uppercasing
    ``variant_name``).
    """
    target_tag = config.get("TARGET_ACHIEVEMENT") or "default"
    timesteps_m = int(config["TOTAL_TIMESTEPS"] // 1e6)
    variant = str(config["VARIANT_NAME"]).upper()
    base = (
        f"{config['ENV_NAME']}-{variant}-{timesteps_m}M-"
        f"{target_tag}-{config['EMBEDDING_KIND']}"
    )
    experiment = config.get("EXPERIMENT_NAME")
    return f"{base}-{experiment}" if experiment else base


def _validate_cross_cutting(config: dict) -> None:
    """Cross-group constraints not expressible via group composition.

    Preserved verbatim from the original ``__main__`` blocks:

    - finite-coef check on ``ENV_REWARD_COEF`` / ``ACCEPT_REWARD_COEF``
      (was in ``apply_reward_config_defaults``);
    - E3B requires ICM and ``ICM_REWARD_COEFF == 0`` (was in
      ``ppo.py``'s ``__main__``).
    """
    for key in ("ENV_REWARD_COEF", "ACCEPT_REWARD_COEF"):
        value = float(config[key])
        assert math.isfinite(value), f"{key} must be finite, got {value}"
    if config.get("USE_E3B", False):
        assert config.get("TRAIN_ICM", False), "USE_E3B requires TRAIN_ICM"
        assert config.get("ICM_REWARD_COEFF", 0.0) == 0, (
            "USE_E3B requires ICM_REWARD_COEFF == 0"
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: compose, normalize, validate, dispatch."""
    plain: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)

    target = plain.pop("run_target")
    config = _to_uppercase_dict(plain)

    # argparse used ``type=lambda x: int(float(x))`` so e.g. ``1e9``
    # arrived in ``make_train`` as ``int(1000000000)``; preserve that
    # so ``NUM_UPDATES = TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENVS``
    # stays in integer arithmetic.
    config["TOTAL_TIMESTEPS"] = int(float(config["TOTAL_TIMESTEPS"]))

    if config.get("SEED") is None:
        config["SEED"] = int(np.random.randint(2**31))

    _validate_cross_cutting(config)

    config["WANDB_RUN_NAME"] = _build_wandb_run_name(config)

    run_fn = _resolve_run_target(target)

    if config.get("JIT", True):
        run_fn(config)
    else:
        with jax.disable_jit():
            run_fn(config)


if __name__ == "__main__":
    main()
