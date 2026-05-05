#!/usr/bin/env python3
r"""Run upstream Craftax_Baselines PPO for exactly one update step.

Wires up our :class:`AutomatonAugmentedEnvWrapper` to the upstream env
construction by monkey-patching ``make_craftax_env_from_name`` inside
the vendored ``ppo`` module. This leaves upstream code untouched while
proving that:

- The wrapped env composes with upstream's wrapper stack
  (``LogWrapper`` -> ``OptimisticResetVecEnvWrapper``).
- Upstream's ``make_train`` builds the network on the wrapped obs shape.
- Exactly one PPO update step runs end-to-end on GPU.
- All trained params end up on the configured GPU.

Wandb is disabled. Synthetic backends (random init for Brzozowski,
random arrays for RAD) since no torch checkpoint or RAD ``.npz`` files
exist locally.

Usage::

    .venv/bin/python scripts/run_one_ppo_update.py
    .venv/bin/python scripts/run_one_ppo_update.py --backend rad_lookup
    .venv/bin/python scripts/run_one_ppo_update.py --backend none
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# XLA env vars BEFORE jax imports.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

# This file is at ``<repo>/third_party/craftax_baselines/smokes/X.py`` after
# the restructure. ``parents[3]`` is the repo root; ``parents[1]`` is the
# submodule (where upstream ``ppo`` and ``wrappers`` live).
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _build_synthetic_rad_backend():
    """Synthetic RadLookupBackend: 4 states, 4 symbols, 8-dim embeddings."""
    from automata_rl.wrappers import RadLookupBackend
    rng = np.random.RandomState(0)
    return RadLookupBackend(
        transition=jnp.asarray(rng.randint(0, 4, size=(4, 4)).astype(np.int32)),
        embedding=jnp.asarray(rng.rand(4, 8).astype(np.float32)),
        accept=jnp.asarray(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)),
        initial_residual=jnp.int32(0),
    )


def _build_synthetic_brzozowski_backend():
    """Synthetic JaxBrzozowskiBackend with random-init Flax params."""
    from automata_jax.config import (
        AFAEmbeddingConfig,
        EquivariantSetEncoderConfig,
    )
    from automata_jax.model import AFAEmbedding
    from automata_jax.utils import default_evaluation_points_jax
    from automata_rl.wrappers import JaxBrzozowskiBackend

    cfg = AFAEmbeddingConfig(
        num_vars=10, cp_rank=4, max_degree=4,
        embed_dim=64, task_dim=32, symbol_dim=32,
        num_tasks=67, num_predicates=2,
        generator_hidden_dim=128,
        set_encoder=EquivariantSetEncoderConfig(
            kind="set_transformer",
            embed_dim=64, hidden_dim=128, num_heads=4, num_layers=2,
        ),
    )
    model = AFAEmbedding(cfg=cfg)
    params = model.init(
        jax.random.key(123),
        jnp.array([0], dtype=jnp.int32),
        jnp.zeros((1, 1), dtype=jnp.int32),
    )
    eval_points = default_evaluation_points_jax(num_vars=cfg.num_vars)
    return JaxBrzozowskiBackend(model, params, task_idx=0, eval_points=eval_points)


def _make_predicate_eval(n_pred: int = 2):
    from automata_rl.predicate_eval import (
        achievement_extractor,
        make_predicate_evaluator,
    )
    pred_idx = jnp.array([0, 1], dtype=jnp.int32)
    return make_predicate_evaluator(achievement_extractor(pred_idx), n_pred=n_pred)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--backend",
        choices=["none", "rad_lookup", "brzozowski_jax"],
        default="brzozowski_jax",
    )
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"\n{'=' * 70}\nBackend: {args.backend}, num_envs={args.num_envs}, num_steps={args.num_steps}")
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    gpu = next((d for d in devices if d.platform == "gpu"), None)
    if gpu is None:
        print("No GPU available; aborting.")
        return 1
    jax.config.update("jax_default_device", gpu)

    # Monkey-patch ``make_craftax_env_from_name`` inside the upstream ppo
    # module so its env construction picks up our AutomatonAugmentedEnvWrapper.
    from craftax.craftax_env import make_craftax_env_from_name as _orig_make_env

    if args.backend == "none":
        patched_make_env = _orig_make_env
    else:
        from automata_rl.wrappers import AutomatonAugmentedEnvWrapper

        if args.backend == "rad_lookup":
            backend = _build_synthetic_rad_backend()
        else:
            backend = _build_synthetic_brzozowski_backend()
        predicate_eval = _make_predicate_eval(n_pred=2)

        def patched_make_env(env_name: str, auto_reset: bool):
            inner = _orig_make_env(env_name, auto_reset)
            return AutomatonAugmentedEnvWrapper(
                inner, backend, predicate_eval, augment_accept=True,
            )

    import ppo as cb_ppo  # vendored
    cb_ppo.make_craftax_env_from_name = patched_make_env  # noqa: SLF001

    # Build minimal config: total_timesteps = num_envs * num_steps -> NUM_UPDATES=1.
    config: dict = {
        "ENV_NAME": "Craftax-Symbolic-v1",
        "USE_OPTIMISTIC_RESETS": True,
        "OPTIMISTIC_RESET_RATIO": 16,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.num_steps,
        "TOTAL_TIMESTEPS": args.num_envs * args.num_steps,
        "LR": 2e-4,
        "ANNEAL_LR": True,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.8,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 1.0,
        "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES": 4,
        "LAYER_SIZE": 128,
        "ACTIVATION": "tanh",
        "TRAIN_ICM": False,
        "USE_E3B": False,
        "DEBUG": False,
        "JIT": True,
        "USE_WANDB": False,
        "SAVE_POLICY": False,
        "SEED": args.seed,
        "NUM_REPEATS": 1,
    }

    print("Calling upstream make_train ...")
    t0 = time.time()
    train_fn = cb_ppo.make_train(config)
    print(f"  NUM_UPDATES (computed): {config['NUM_UPDATES']}")
    print(f"  MINIBATCH_SIZE: {config['MINIBATCH_SIZE']}")
    print(f"  Build time: {time.time() - t0:.2f}s")

    print("JIT-compiling and running 1 update step ...")
    t0 = time.time()
    train_jit = jax.jit(train_fn)
    rng = jax.random.PRNGKey(args.seed)
    out = train_jit(rng)
    # Block until the train fn finishes.
    jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
    print(f"  Total wall time (compile + 1 update): {time.time() - t0:.2f}s")

    # Inspect: where are the resulting params?
    runner_state = out["runner_state"]
    train_state = runner_state[0]
    leaves = jax.tree.leaves(train_state.params)
    devices_set = {l.device for l in leaves}
    on_gpu = all(d.platform == "gpu" for d in devices_set)
    n_params = sum(int(np.prod(l.shape)) for l in leaves)
    sizes = sum(int(np.prod(l.shape)) * l.dtype.itemsize for l in leaves) / 1e6
    print(f"\nFinal train_state.params:")
    print(f"  count: {n_params:,}")
    print(f"  size: {sizes:.2f} MB")
    print(f"  devices: {devices_set}")
    print(f"  on_gpu: {on_gpu}")

    # Sample a few values to confirm they're not NaN
    sample = jax.tree.leaves(train_state.params)[0]
    print(f"  sample leaf shape: {sample.shape}, dtype: {sample.dtype}, "
          f"finite: {bool(jnp.all(jnp.isfinite(sample)))}")
    print(f"  step count after 1 update: {int(train_state.step)}")

    print(f"\n[OK] Backend={args.backend!r}: 1 PPO update via upstream make_train completed on GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
