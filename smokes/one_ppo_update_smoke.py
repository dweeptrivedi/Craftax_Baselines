#!/usr/bin/env python3
r"""End-to-end smoke: one PPO update with synthetic embeddings on GPU.

Bypasses the need for a real torch checkpoint or RAD ``.npz`` files by
constructing both backend variants from synthetic data:

- ``rad_lookup``: a ``RadLookupBackend`` with random transition / embedding
  / accept arrays.
- ``brzozowski_jax``: a ``JaxBrzozowskiBackend`` using the Flax
  ``AFAEmbedding`` with freshly random-initialised params (NOT loaded
  from a torch checkpoint; the converter script handles that).

For each backend, runs:

1. Env stack construction (raw Craftax -> wrapper -> LogWrapper ->
   OptimisticResetVecEnvWrapper).
2. 32 rollout steps with random actions.
3. One PPO-style policy loss + optax update.
4. Asserts every leaf of the resulting params lives on the configured
   GPU and is finite.

Prints per-stage timings + device summaries so it's easy to spot
unexpected host fallbacks.

Usage::

    XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/one_ppo_update_smoke.py
    XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/one_ppo_update_smoke.py --backend rad_lookup
    XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/one_ppo_update_smoke.py --backend brzozowski_jax
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure XLA env vars are picked up before any jax import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import distrax  # noqa: E402  — only to assert availability before ppo import
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

# This file is at ``<repo>/third_party/craftax_baselines/smokes/X.py`` after
# the restructure. ``parents[3]`` is the repo root; ``parents[1]`` is the
# submodule (where upstream ``ppo`` and ``wrappers`` live).
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from craftax.craftax_env import make_craftax_env_from_name  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402
from models.actor_critic import ActorCritic  # noqa: E402  (vendored)
from wrappers import LogWrapper, OptimisticResetVecEnvWrapper  # noqa: E402

from automata_jax.config import AFAEmbeddingConfig, EquivariantSetEncoderConfig  # noqa: E402
from automata_jax.model import AFAEmbedding  # noqa: E402
from automata_jax.utils import default_evaluation_points_jax  # noqa: E402
from automata_rl.predicate_eval import (  # noqa: E402
    achievement_extractor,
    make_predicate_evaluator,
)
from automata_rl.wrappers import (  # noqa: E402
    AutomatonAugmentedEnvWrapper,
    JaxBrzozowskiBackend,
    RadLookupBackend,
)


def _device_summary(name: str, leaves) -> None:
    """Print which device each leaf array lives on."""
    devices = {l.device for l in leaves if hasattr(l, "device")}
    on_gpu = all(d.platform == "gpu" for d in devices)
    sizes = sum(int(np.prod(l.shape)) * l.dtype.itemsize for l in leaves) / 1e6
    print(
        f"  {name:25s}  devices={devices}  size={sizes:.2f} MB  "
        f"on_gpu={'YES' if on_gpu else 'NO'}",
    )


def _build_synthetic_rad_backend() -> RadLookupBackend:
    """Synthetic RAD backend: small DFA, 8-dim embeddings."""
    rng = np.random.RandomState(0)
    transition = rng.randint(0, 4, size=(4, 4)).astype(np.int32)
    embedding = rng.rand(4, 8).astype(np.float32)
    accept = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return RadLookupBackend(
        transition=jnp.asarray(transition),
        embedding=jnp.asarray(embedding),
        accept=jnp.asarray(accept),
        initial_residual=jnp.int32(0),
    )


def _build_synthetic_brzozowski_backend() -> JaxBrzozowskiBackend:
    """Synthetic JAX-Brzozowski backend: random-init Flax params."""
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
    key = jax.random.key(123)
    sample_task = jnp.array([0], dtype=jnp.int32)
    sample_sym = jnp.zeros((1, 1), dtype=jnp.int32)
    params = model.init(key, sample_task, sample_sym)
    eval_points = default_evaluation_points_jax(
        num_vars=cfg.num_vars, max_exact_vars=10,
    )
    return JaxBrzozowskiBackend(model, params, task_idx=0, eval_points=eval_points)


def build_env_stack(backend_kind: str, num_envs: int):
    """Wrap a real Craftax-Symbolic env with our augmented wrapper + upstream wrappers."""
    inner = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)

    if backend_kind == "rad_lookup":
        backend = _build_synthetic_rad_backend()
        # Need 2 predicates for the synthetic 4-symbol DFA.
        pred_idx = jnp.array([0, 1], dtype=jnp.int32)
        n_pred = 2
    elif backend_kind == "brzozowski_jax":
        backend = _build_synthetic_brzozowski_backend()
        pred_idx = jnp.array([0, 1], dtype=jnp.int32)
        n_pred = 2
    elif backend_kind == "none":
        env_params = inner.default_params
        env = LogWrapper(inner)
        env = OptimisticResetVecEnvWrapper(
            env, num_envs=num_envs, reset_ratio=min(16, num_envs),
        )
        return env, env_params, None

    extractor = achievement_extractor(pred_idx)
    predicate_eval = make_predicate_evaluator(extractor, n_pred=n_pred)
    env = AutomatonAugmentedEnvWrapper(inner, backend, predicate_eval)
    env = LogWrapper(env)
    env = OptimisticResetVecEnvWrapper(
        env, num_envs=num_envs, reset_ratio=min(16, num_envs),
    )
    return env, inner.default_params, backend


def run(backend_kind: str, num_envs: int, num_steps: int) -> None:
    print(f"\n{'='*70}\nBackend: {backend_kind}, num_envs={num_envs}, num_steps={num_steps}")
    print(f"JAX devices: {jax.devices()}")
    gpu = jax.devices("cuda")[0] if any(d.platform == "gpu" for d in jax.devices()) else None
    if gpu is None:
        print("No GPU available; aborting smoke.")
        return

    jax.config.update("jax_default_device", gpu)

    t0 = time.time()
    env, env_params, backend = build_env_stack(backend_kind, num_envs)
    print(f"  Env stack built in {time.time()-t0:.2f}s")

    obs_shape = env.observation_space(env_params).shape
    print(f"  Obs space shape: {obs_shape}")

    # Build network
    network = ActorCritic(
        action_dim=env.action_space(env_params).n,
        layer_width=128,
        activation="tanh",
    )
    rng = jax.random.key(0)
    rng, key_init = jax.random.split(rng)
    dummy_obs = jnp.zeros((num_envs,) + obs_shape, dtype=jnp.float32)
    params = network.init(key_init, dummy_obs)
    n_params = sum(int(np.prod(l.shape)) for l in jax.tree.leaves(params))
    print(f"  Network params: {n_params:,}")
    _device_summary("Network params", jax.tree.leaves(params))

    # Initial reset
    rng, key_reset = jax.random.split(rng)
    obs, env_state = env.reset(key_reset, env_params)
    print(f"  After reset: obs shape={obs.shape}, on device={obs.device}")
    _device_summary("Wrapped env state", jax.tree.leaves(env_state))

    # Optimizer + train state
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=2e-4),
    )
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    # ---- collect a tiny rollout ----
    def env_step_jit(rng_in, obs_in, env_state_in):
        @jax.jit
        def _step(rng_in, obs_in, env_state_in):
            pi, value = network.apply(train_state.params, obs_in)
            rng_in, key_sample = jax.random.split(rng_in)
            action = pi.sample(seed=key_sample)
            log_prob = pi.log_prob(action)
            rng_in, key_step = jax.random.split(rng_in)
            obs_next, env_state_next, reward, done, info = env.step(
                key_step, env_state_in, action, env_params,
            )
            return obs_next, env_state_next, action, value, log_prob, reward, done, info, rng_in
        return _step(rng_in, obs_in, env_state_in)

    print("  Collecting rollout (first step compiles the graph) ...")
    obs_buf, action_buf, value_buf, lp_buf, reward_buf, done_buf = [], [], [], [], [], []
    next_obs_buf = []
    t_compile = time.time()
    for t in range(num_steps):
        obs_next, env_state, action, value, log_prob, reward, done, info, rng = env_step_jit(rng, obs, env_state)
        obs_buf.append(obs)
        action_buf.append(action)
        value_buf.append(value)
        lp_buf.append(log_prob)
        reward_buf.append(reward)
        done_buf.append(done)
        next_obs_buf.append(obs_next)
        obs = obs_next
        if t == 0:
            print(f"    First step (with compile): {time.time()-t_compile:.2f}s")
            t_compile = time.time()
    print(f"    Remaining {num_steps-1} steps: {time.time()-t_compile:.2f}s")

    obs_t = jnp.stack(obs_buf, axis=0)         # (T, B, obs_dim)
    actions_t = jnp.stack(action_buf, axis=0)
    values_t = jnp.stack(value_buf, axis=0)
    log_probs_t = jnp.stack(lp_buf, axis=0)
    rewards_t = jnp.stack(reward_buf, axis=0)
    dones_t = jnp.stack(done_buf, axis=0).astype(jnp.float32)
    print(f"  Rollout collected: obs={obs_t.shape}, rewards on device={rewards_t.device}")
    _device_summary("Rollout obs",  [obs_t])

    # ---- one PPO update ----
    # Compute discounted returns (no GAE for the smoke; just simple returns)
    gamma = 0.99
    def _returns(rewards, dones, last_value, gamma):
        T = rewards.shape[0]
        ret = last_value
        out = jnp.zeros_like(rewards)
        for t in range(T - 1, -1, -1):
            ret = rewards[t] + gamma * (1.0 - dones[t]) * ret
            out = out.at[t].set(ret)
        return out

    last_pi, last_value = network.apply(train_state.params, obs)
    returns = _returns(rewards_t, dones_t, last_value, gamma)
    advantages = returns - values_t
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    flat_obs = obs_t.reshape(-1, obs_t.shape[-1])
    flat_actions = actions_t.reshape(-1)
    flat_lp = log_probs_t.reshape(-1)
    flat_adv = advantages.reshape(-1)
    flat_ret = returns.reshape(-1)
    flat_val = values_t.reshape(-1)

    @jax.jit
    def update_step(state, obs, actions, old_lp, advantages, returns, old_values):
        def loss_fn(params):
            pi, value = state.apply_fn(params, obs)
            log_prob = pi.log_prob(actions)
            ratio = jnp.exp(log_prob - old_lp)
            clipped = jnp.clip(ratio, 1 - 0.2, 1 + 0.2)
            policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped * advantages))
            value_loss = 0.5 * jnp.mean((value - returns) ** 2)
            ent = jnp.mean(pi.entropy())
            return policy_loss + 0.5 * value_loss - 0.01 * ent, (policy_loss, value_loss, ent)
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), loss, aux

    print("  Running one PPO update ...")
    t_update = time.time()
    train_state, loss, (pl, vl, ent) = update_step(
        train_state, flat_obs, flat_actions, flat_lp,
        flat_adv, flat_ret, flat_val,
    )
    loss.block_until_ready()
    print(f"    Update step (with compile): {time.time()-t_update:.2f}s")
    print(f"    loss={float(loss):.4f}  policy={float(pl):.4f}  value={float(vl):.4f}  entropy={float(ent):.4f}")

    _device_summary("Updated params", jax.tree.leaves(train_state.params))

    # Final sanity: run one more env step with updated params (no compile)
    t_post = time.time()
    obs2, *_ = env_step_jit(rng, obs, env_state)
    obs2.block_until_ready()
    print(f"    One env step with updated params: {time.time()-t_post:.3f}s")
    print(f"  ✓ End-to-end PPO update completed on GPU for backend={backend_kind!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["none", "rad_lookup", "brzozowski_jax", "all"], default="all")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--num-steps", type=int, default=16)
    args = ap.parse_args()

    targets = ["none", "rad_lookup", "brzozowski_jax"] if args.backend == "all" else [args.backend]
    for kind in targets:
        run(kind, args.num_envs, args.num_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
