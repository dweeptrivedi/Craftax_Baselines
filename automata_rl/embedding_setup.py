r"""Factory for the embedding wrapper stack used by :file:`ppo.py`.

Loads on-disk artefacts (Brzozowski Flax params msgpack + config yaml +
eval_points npy, or RAD lookup npz), builds the corresponding backend,
constructs a :class:`automata_rl.wrappers.AutomatonAugmentedEnvWrapper`,
and chains on a :class:`automata_rl.wrappers.RewardCompositionWrapper`
configured by ``ACCEPT_REWARD_KIND``, ``ACCEPT_REWARD_COEF``, and
``ENV_REWARD_COEF`` from the ``ppo.py`` config.

Public entry point :func:`build_embedding_stack` returns
``(wrapped_env, emb_split_idx)`` where ``emb_split_idx`` is the
length of the inner Craftax obs vector before any embedding columns
are concatenated. ``ppo.py`` stores this in
``config["EMB_SPLIT_IDX"]`` and threads it into the FiLM / late-fusion
policy constructors so they can slice the augmented obs internally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import flax.serialization
import jax.numpy as jnp
import numpy as np
import yaml

from automata_jax.config import AFAEmbeddingConfig, EquivariantSetEncoderConfig
from automata_jax.model import AFAEmbedding
from automata_rl.predicate_eval import (
    achievement_extractor,
    make_predicate_evaluator,
)
from automata_rl.wrappers import (
    AutomatonAugmentedEnvWrapper,
    JaxBrzozowskiBackend,
    RadLookupBackend,
    RewardCompositionWrapper,
)


def _load_brzozowski_jax_backend(
    *,
    params_path: Path,
    config_path: Path,
    eval_points_path: Path,
    task_idx: int,
) -> JaxBrzozowskiBackend:
    """Reconstruct :class:`JaxBrzozowskiBackend` from the converter outputs.

    The converter (``scripts/convert_brzozowski_to_jax.py``) writes:

    - ``params_path``: msgpack of ``{"params": <Flax pytree>}``.
    - ``config_path``: flat YAML with ``set_encoder`` as a nested dict;
      every other field maps directly to an :class:`AFAEmbeddingConfig`
      attribute.
    - ``eval_points_path``: ``(M, num_vars)`` float32 npy.

    We DO NOT call ``model.init`` -- the msgpack already contains the full
    params pytree, and :class:`JaxBrzozowskiBackend` only ever calls
    ``model.apply(params, ...)`` against externally-provided params.
    """
    raw_cfg = yaml.safe_load(config_path.read_text())
    set_cfg = EquivariantSetEncoderConfig(**raw_cfg.pop("set_encoder"))
    cfg = AFAEmbeddingConfig(set_encoder=set_cfg, **raw_cfg)

    model = AFAEmbedding(cfg=cfg)
    params = flax.serialization.msgpack_restore(params_path.read_bytes())
    eval_points = jnp.asarray(np.load(eval_points_path))
    return JaxBrzozowskiBackend(model, params, task_idx, eval_points)


def _load_rad_lookup_backend(
    *,
    lookup_path: Path,
    task_name: str,
) -> RadLookupBackend:
    """Reconstruct :class:`RadLookupBackend` for ``task_name`` from ``lookup_rad.npz``.

    The lookup builder (``scripts/build_rad_lookup.py``) packs per-task
    arrays as ``transition_<name>``, ``embedding_<name>``, ``accept_<name>``,
    ``initial_residual_<name>``. If ``task_name`` is missing we raise with
    the list of compatible task names recorded in the npz.
    """
    raw = np.load(lookup_path, allow_pickle=False)
    available = list(map(str, raw["task_names"]))
    if task_name not in available:
        raise ValueError(
            f"RAD lookup at {lookup_path} does not contain task {task_name!r}. "
            f"Available: {available}"
        )
    return RadLookupBackend(
        transition=jnp.asarray(raw[f"transition_{task_name}"]),
        embedding=jnp.asarray(raw[f"embedding_{task_name}"]),
        accept=jnp.asarray(raw[f"accept_{task_name}"]),
        initial_residual=jnp.int32(int(raw[f"initial_residual_{task_name}"])),
    )


def build_embedding_stack(env: Any, config: dict) -> tuple[Any, int]:
    """Build the embedding-augmented env stack.

    Args:
        env: The inner Craftax env (or task-wrapped env) returned from
            ``make_craftax_env_from_name(...)`` /
            ``CraftaxSymbolicTaskEnv(...)`` in ``ppo.py``. Should not yet
            have ``LogWrapper`` or any vec wrapper applied.
        config: ppo.py's flat config dict. Reads ``EMBEDDING_KIND``,
            ``TARGET_ACHIEVEMENT``, ``BRZOZOWSKI_*_PATH`` /
            ``RAD_LOOKUP_PATH``, ``TASK_PREDICATES_PATH``,
            ``ACCEPT_REWARD_KIND``, ``ACCEPT_REWARD_COEF``,
            ``ENV_REWARD_COEF``.

    Returns:
        ``(wrapped_env, emb_split_idx)``. ``emb_split_idx`` is the inner
        Craftax obs dimension; the augmented obs has shape
        ``(emb_split_idx + backend.embed_dim + 1,)``.
    """
    task_name: str = config["TARGET_ACHIEVEMENT"]
    task_predicates_path = Path(config["TASK_PREDICATES_PATH"])
    db = json.loads(task_predicates_path.read_text())
    if task_name not in db:
        raise ValueError(
            f"Task {task_name!r} not in {task_predicates_path}. "
            f"Available: {sorted(db.keys())}"
        )
    meta = db[task_name]
    pred_idx = jnp.asarray(meta["predicate_indices"], dtype=jnp.int32)
    n_pred = int(meta["n_predicates"])
    task_idx = int(meta["task_idx"])

    extractor = achievement_extractor(pred_idx)
    pred_eval = make_predicate_evaluator(extractor, n_pred=n_pred)

    kind: str = config["EMBEDDING_KIND"]
    if kind == "brzozowski_jax":
        backend = _load_brzozowski_jax_backend(
            params_path=Path(config["BRZOZOWSKI_PARAMS_PATH"]),
            config_path=Path(config["BRZOZOWSKI_CONFIG_PATH"]),
            eval_points_path=Path(config["BRZOZOWSKI_EVAL_POINTS_PATH"]),
            task_idx=task_idx,
        )
    elif kind == "rad_lookup":
        backend = _load_rad_lookup_backend(
            lookup_path=Path(config["RAD_LOOKUP_PATH"]),
            task_name=task_name,
        )
    else:
        raise ValueError(
            f"Unknown EMBEDDING_KIND: {kind!r}; expected 'brzozowski_jax' or 'rad_lookup'."
        )

    inner_obs_dim = int(env.observation_space(env.default_params).shape[0])
    env = AutomatonAugmentedEnvWrapper(
        env, backend, pred_eval, augment_accept=True,
    )

    accept_reward_kind = config.get("ACCEPT_REWARD_KIND", "continuous")
    if accept_reward_kind == "dense_accept_prob" and kind == "rad_lookup":
        # RAD's accept is binary; dense_accept_prob's delta would flicker.
        raise ValueError(
            "ACCEPT_REWARD_KIND='dense_accept_prob' is incompatible with "
            "EMBEDDING_KIND='rad_lookup' (RAD accept is binary 0/1)."
        )
    env = RewardCompositionWrapper(
        env,
        kind=accept_reward_kind,
        accept_coef=float(config.get("ACCEPT_REWARD_COEF", 1.0)),
        env_coef=float(config.get("ENV_REWARD_COEF", 0.0)),
    )

    return env, inner_obs_dim


__all__ = ("build_embedding_stack",)
