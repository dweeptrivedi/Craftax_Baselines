r"""Env wrapper that augments the observation with automaton fingerprint + accept.

Per the plan (:file:`AFA_INFER_CRAFTAX_BASELINE.md`, "Backend abstraction"
section), the wrapper is polymorphic over two backends:

- :class:`JaxBrzozowskiBackend` — live JAX inference of the ported
  ``AFAEmbedding`` model on every env step. Handles all 67 standard
  Craftax tasks plus custom LTLf.
- :class:`RadLookupBackend` — gather from precomputed ``(num_states,
  embed_dim)``/``(num_states, num_symbols)`` arrays. Limited to the RAD-
  compatible subset of tasks.

Both backends expose the same interface:

- ``initial_q()`` -> initial automaton state (a polynomial pytree for
  Brzozowski, an int32 residual id for RAD).
- ``step_automaton(q, sym)`` -> ``(next_q, fingerprint, accept)``.
- ``current_features(q)`` -> ``(fingerprint, accept)`` without advancing.
- ``embed_dim`` (property) for ``observation_space`` shape inference.

The wrapper subclasses Craftax_Baselines's ``GymnaxWrapper`` and is
inserted between ``make_craftax_env_from_name(...)`` and ``LogWrapper``.
``stop_gradient`` on the embedding outputs is unconditional per the
no-joint-training scope.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Protocol

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float, Int

# Add the submodule root to sys.path so ``from wrappers import GymnaxWrapper``
# resolves to ``craftax_baselines/wrappers.py`` (the upstream wrapper file).
# After the restructure this file lives at
# ``<repo>/third_party/craftax_baselines/automata_rl/wrappers.py``, so
# ``parents[1]`` is the submodule directory itself.
_CB_PATH = Path(__file__).resolve().parents[1]
if str(_CB_PATH) not in sys.path:
    sys.path.insert(0, str(_CB_PATH))

from wrappers import GymnaxWrapper  # noqa: E402

from automata_jax.utils import evaluate_many_lowrank_jax  # noqa: E402


# ---------------------------------------------------------------------------
# Augmented env-state pytrees
# ---------------------------------------------------------------------------


@struct.dataclass
class BrzozowskiAugState:
    """Carries craftax env_state + the automaton's polynomial state."""

    env_state: Any
    q_state: Float[Array, "R D N1"]  # merged factors


@struct.dataclass
class RadAugState:
    """Carries craftax env_state + a discrete RAD DFA state id."""

    env_state: Any
    q_state: Int[Array, ""]  # int32 residual id


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class Backend(Protocol):
    """Common interface for Brzozowski-JAX and RAD-lookup backends."""

    embed_dim: int
    initial_state_factory: type

    def initial_q(self) -> Any: ...
    def step_automaton(self, q_state: Any, sym: Int[Array, ""]) -> tuple[Any, Float[Array, "M"], Float[Array, ""]]: ...
    def current_features(self, q_state: Any) -> tuple[Float[Array, "M"], Float[Array, ""]]: ...


# ---------------------------------------------------------------------------
# JAX-native Brzozowski backend
# ---------------------------------------------------------------------------


class JaxBrzozowskiBackend:
    """Live JAX inference of the Flax ``AFAEmbedding`` model.

    Holds the model module + its trained params, plus the per-task
    automaton (precomputed once at construction time). Every call to
    ``step_automaton`` runs a forward pass through the embedding model
    and applies ``stop_gradient`` to all outputs (per the no-joint-
    training scope decision).

    Args:
        model: A Flax ``AFAEmbedding`` instance.
        params: Trained Flax params (loaded via the converter).
        task_idx: Index of the task this backend handles (single-task MVP).
        eval_points: Fingerprint evaluation points, ``(M, num_vars)``.
            Should be loaded from the converter's saved
            ``brzozowski_jax_eval_points.npy`` for reproducibility with
            the torch reference.
    """

    def __init__(
        self,
        model: Any,  # AFAEmbedding (avoiding circular import)
        params: Any,
        task_idx: int,
        eval_points: Float[Array, "M N"],
    ) -> None:
        from automata_jax.model import AFAEmbedding  # local to avoid cycle

        self._model: AFAEmbedding = model
        self._params = params
        self._eval_points = eval_points
        # Precompute automaton for the configured task (constant for single-task).
        autom_b = model.apply(
            params, jnp.array([task_idx], dtype=jnp.int32),
            method=AFAEmbedding.initialize_automaton,
        )
        # Squeeze batch dim so per-env state is unbatched.
        self._automaton = jax.tree.map(lambda x: x[0], autom_b)
        self._initial_q = self._automaton.q_hat_merged
        self.embed_dim = int(eval_points.shape[0])
        self.initial_state_factory = BrzozowskiAugState

    def initial_q(self) -> Float[Array, "R D N1"]:
        return self._initial_q

    def _wrap(self, q_state: Float[Array, "R D N1"]) -> Float[Array, "1 R D N1"]:
        """Add a leading singleton batch axis (model methods expect batched)."""
        return q_state[None]

    def _unwrap(self, batched: Float[Array, "1 ..."]) -> Float[Array, "..."]:
        return batched[0]

    def step_automaton(
        self,
        q_state: Float[Array, "R D N1"],
        sym: Int[Array, ""],
    ) -> tuple[Float[Array, "R D N1"], Float[Array, "M"], Float[Array, ""]]:
        from automata_jax.model import AFAEmbedding

        q_b = self._wrap(q_state)
        sym_b = sym[None]
        # Re-batch the (single) automaton for the model methods.
        autom_b = jax.tree.map(lambda x: x[None], self._automaton)

        next_q_b = self._model.apply(
            self._params, autom_b, q_b, sym_b,
            method=AFAEmbedding.transition,
        )
        accept_b = self._model.apply(
            self._params, next_q_b, autom_b,
            method=AFAEmbedding.is_accepting,
        )
        next_q = self._unwrap(next_q_b)
        accept = self._unwrap(accept_b)
        # evaluate_many_lowrank_jax: (1, R, D, N+1) + (M, N) -> (1, M); take [0].
        fp = evaluate_many_lowrank_jax(next_q[None], self._eval_points)[0]
        # Stop-gradient on all outputs (frozen embedding, no joint training).
        next_q = jax.lax.stop_gradient(next_q)
        fp, accept = jax.lax.stop_gradient((fp, accept))
        return next_q, fp, accept

    def current_features(
        self,
        q_state: Float[Array, "R D N1"],
    ) -> tuple[Float[Array, "M"], Float[Array, ""]]:
        from automata_jax.model import AFAEmbedding

        q_b = self._wrap(q_state)
        autom_b = jax.tree.map(lambda x: x[None], self._automaton)
        accept_b = self._model.apply(
            self._params, q_b, autom_b,
            method=AFAEmbedding.is_accepting,
        )
        accept = self._unwrap(accept_b)
        fp = evaluate_many_lowrank_jax(q_state[None], self._eval_points)[0]
        return jax.lax.stop_gradient(fp), jax.lax.stop_gradient(accept)


# ---------------------------------------------------------------------------
# RAD lookup backend
# ---------------------------------------------------------------------------


class RadLookupBackend:
    """Discrete-DFA lookup backend for RAD-compatible tasks.

    Args:
        transition: ``(num_states, num_symbols)`` int32.
        embedding: ``(num_states, embed_dim)`` float32.
        accept: ``(num_states,)`` float32 (cast from bool).
        initial_residual: ``()`` int32.
    """

    def __init__(
        self,
        transition: Int[Array, "S A"],
        embedding: Float[Array, "S D"],
        accept: Float[Array, "S"],
        initial_residual: Int[Array, ""],
    ) -> None:
        self._transition = transition
        self._embedding = embedding
        self._accept = accept
        self._initial = initial_residual
        self.embed_dim = int(embedding.shape[-1])
        self.initial_state_factory = RadAugState

    def initial_q(self) -> Int[Array, ""]:
        return self._initial

    def step_automaton(
        self,
        q_state: Int[Array, ""],
        sym: Int[Array, ""],
    ) -> tuple[Int[Array, ""], Float[Array, "D"], Float[Array, ""]]:
        sym = jnp.clip(sym, 0, self._transition.shape[1] - 1)
        next_rid = self._transition[q_state, sym]
        return next_rid, self._embedding[next_rid], self._accept[next_rid]

    def current_features(
        self,
        q_state: Int[Array, ""],
    ) -> tuple[Float[Array, "D"], Float[Array, ""]]:
        return self._embedding[q_state], self._accept[q_state]


# ---------------------------------------------------------------------------
# Env wrapper
# ---------------------------------------------------------------------------


class AutomatonAugmentedEnvWrapper(GymnaxWrapper):
    """Augment the observation with automaton fingerprint + accept.

    Insertion point in the upstream ``make_train``:

    .. code-block:: python

        env = make_craftax_env_from_name(env_name, auto_reset=...)
        env = AutomatonAugmentedEnvWrapper(env, backend, predicate_eval)  # NEW
        env = LogWrapper(env)
        if optimistic_reset:
            env = OptimisticResetVecEnvWrapper(env, ...)
        else:
            env = AutoResetEnvWrapper(env)
            env = BatchEnvWrapper(env, num_envs)

    Args:
        env: Inner Craftax env (or any gymnax-style env).
        backend: One of :class:`JaxBrzozowskiBackend` or
            :class:`RadLookupBackend`.
        predicate_eval: Callable from
            :func:`automata_rl.predicate_eval.make_predicate_evaluator`.
            Takes the FULL env_state (not just achievements).
        augment_accept: If True, append the scalar accept value to the
            observation. Default True.
    """

    def __init__(
        self,
        env: Any,
        backend: Backend,
        predicate_eval: Callable[[Any], Int[Array, ""]],
        augment_accept: bool = True,
    ) -> None:
        super().__init__(env)
        self._backend = backend
        self._predicate_eval = predicate_eval
        self._augment_accept = augment_accept

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: jax.Array, params: Any = None):
        obs, env_state = self._env.reset(key, params)
        q0 = self._backend.initial_q()
        fp, accept = self._backend.current_features(q0)
        aug_obs = self._concat(obs, fp, accept)
        state = self._backend.initial_state_factory(env_state=env_state, q_state=q0)
        return aug_obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key: jax.Array, state: Any, action: Any, params: Any = None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params,
        )
        sym = self._predicate_eval(env_state)
        next_q, fp, accept = self._backend.step_automaton(state.q_state, sym)

        # Snapshot the model's accept of the trajectory's actual final
        # residual BEFORE the done-reset clobbers it. Without this, on
        # ``done=True`` the post-reset overwrite below would replace it
        # with the constant ``init_accept`` and the W&B-side mean (taken
        # only over done rows) would collapse to that constant.
        pre_reset_accept = accept

        # Reset to backend's initial on done. Use jax.tree.map so it
        # works for both pytree (Brzozowski) and scalar (RAD) q_state.
        init_q = self._backend.initial_q()
        next_q = jax.tree.map(
            lambda new, init: jnp.where(done, init, new), next_q, init_q,
        )
        # Match (fp, accept) to the post-reset state on done.
        init_fp, init_accept = self._backend.current_features(init_q)
        fp = jnp.where(done, init_fp, fp)
        accept = jnp.where(done, init_accept, accept)

        # Emit under the ``embedding/`` namespace so the upstream
        # ``logz.batch_logging.create_log_dict`` forwards it to W&B
        # (mirrors the existing ``task/*`` namespace pattern).
        info = {**info, "embedding/accept": pre_reset_accept}
        new_state = state.replace(env_state=env_state, q_state=next_q)
        aug_obs = self._concat(obs, fp, accept)
        return aug_obs, new_state, reward, done, info

    def _concat(
        self,
        obs: Float[Array, "..."],
        fp: Float[Array, "M"],
        accept: Float[Array, ""],
    ) -> Float[Array, "..."]:
        parts = [obs, fp.astype(obs.dtype)]
        if self._augment_accept:
            parts.append(accept[None].astype(obs.dtype))
        return jnp.concatenate(parts, axis=-1)

    def observation_space(self, params: Any = None):
        """Override required: the inner env's space has the original (smaller) shape."""
        from gymnax.environments import spaces

        inner = self._env.observation_space(params)
        extra = self._backend.embed_dim + (1 if self._augment_accept else 0)
        new_shape = (inner.shape[0] + extra,)
        return spaces.Box(0.0, 1.0, new_shape, dtype=inner.dtype)


# ---------------------------------------------------------------------------
# Reward composition wrapper
# ---------------------------------------------------------------------------


@struct.dataclass
class _RewardCompositionState:
    """State carry for :class:`RewardCompositionWrapper`.

    Wraps an inner env's state and tracks the previous step's accept value
    so the configured term function can detect rising edges or compute
    deltas.
    """

    env_state: Any
    prev_accept: Float[Array, ""]


class RewardCompositionWrapper(GymnaxWrapper):
    r"""Compute reward as a linear combination of env reward and accept term.

    The trainer reward emitted on every step is

    .. math::

        \text{reward} = \text{env\_coef} \cdot \text{env\_reward}
                        + \text{accept\_coef} \cdot \text{accept\_term},

    where ``accept_term`` is one of the functions in
    :mod:`automata_rl.reward_shaping` (selected by ``kind``).

    Inserted strictly downstream of
    :class:`AutomatonAugmentedEnvWrapper` so
    ``info["embedding/accept"]`` is populated, and strictly upstream of
    :class:`wrappers.LogWrapper` so the combined reward is what
    ``LogWrapper`` records as the episode return.

    Always emits the raw env reward and raw accept term to ``info``
    under the keys ``"task/env_reward"`` and ``"task/accept_term"``,
    and overwrites ``info["task/reward"]`` (originally set by
    :class:`task_env.CraftaxSymbolicTaskEnv` to the env hit signal)
    with the combined trainer reward so downstream W&B logging shows
    what PPO actually optimizes.

    Args:
        env: Inner env that emits ``info["embedding/accept"]`` (typically
            an :class:`AutomatonAugmentedEnvWrapper`).
        kind: Key into :data:`automata_rl.reward_shaping.TERM_FNS`.
            One of ``"none"``, ``"sparse_accept"``, ``"dense_accept_prob"``,
            ``"continuous"``, ``"continuous_relu"``.
        accept_coef: Coefficient on the accept term.
        env_coef: Coefficient on the inner env reward.
    """

    def __init__(
        self,
        env: Any,
        *,
        kind: str,
        accept_coef: float,
        env_coef: float,
    ) -> None:
        from automata_rl.reward_shaping import TERM_FNS

        super().__init__(env)
        if kind not in TERM_FNS:
            raise ValueError(
                f"Unknown accept_reward_kind: {kind!r}; expected one of {list(TERM_FNS)}",
            )
        self._term_fn = TERM_FNS[kind]
        self._kind = kind
        self._accept_coef = float(accept_coef)
        self._env_coef = float(env_coef)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: jax.Array, params: Any = None):
        obs, inner_state = self._env.reset(key, params)
        return obs, _RewardCompositionState(
            env_state=inner_state, prev_accept=jnp.float32(0.0),
        )

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: jax.Array,
        state: _RewardCompositionState,
        action: Any,
        params: Any = None,
    ):
        obs, inner_state, env_reward, done, info = self._env.step(
            key, state.env_state, action, params,
        )
        # ``info["embedding/accept"]`` is the live pre-reset accept on
        # every step (post the AutomatonAugmentedEnvWrapper fix). The
        # outer auto-reset wrapper replaces this whole state on
        # ``done=True``, so storing ``prev_accept = curr_accept``
        # unconditionally is fine — the carry is a don't-care across
        # episode boundaries.
        curr_accept = info["embedding/accept"]
        accept_term = self._term_fn(done, state.prev_accept, curr_accept)
        combined = self._env_coef * env_reward + self._accept_coef * accept_term
        info = {
            **info,
            "task/env_reward": env_reward,
            "task/accept_term": accept_term,
            "task/reward": combined,
        }
        new_state = _RewardCompositionState(
            env_state=inner_state, prev_accept=curr_accept,
        )
        return obs, new_state, combined, done, info
