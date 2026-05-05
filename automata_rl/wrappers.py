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

import functools
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

        info = {**info, "accept": accept}
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
# Reward shaping wrapper
# ---------------------------------------------------------------------------


@struct.dataclass
class _ShapingState:
    """State carry for :class:`AcceptRewardShapingWrapper`.

    Wraps an inner env's state and tracks the previous step's accept value
    so the shaping function can detect rising edges or compute deltas.
    """

    env_state: Any
    prev_accept: Float[Array, ""]


def _make_shaping_fn(kind: str, scale: float) -> Callable[..., Float[Array, "..."]]:
    """Return a 4-arg ``(reward, done, prev_accept, curr_accept) -> shaped_reward`` callable.

    ``dense_accept_prob`` has a 5th ``scale`` argument; we partial-apply it
    to keep all variants 4-arg from the wrapper's perspective. ``none``
    and ``sparse_accept`` are already 4-arg with no scale.
    """
    from automata_rl.reward_shaping import dense_accept_prob, get_shaping_fn

    if kind == "dense_accept_prob":
        return functools.partial(dense_accept_prob, scale=float(scale))
    return get_shaping_fn(kind)


class AcceptRewardShapingWrapper(GymnaxWrapper):
    r"""Apply accept-based reward shaping on top of an :class:`AutomatonAugmentedEnvWrapper`.

    Reads ``info["accept"]`` produced by the inner wrapper, carries
    ``prev_accept`` per env in its own state pytree, and modifies the
    returned reward via the configured shaping function.

    Insert between :class:`AutomatonAugmentedEnvWrapper` and
    upstream's ``LogWrapper`` so the shaped reward is what
    :class:`LogWrapper` records as the episode return.

    Args:
        env: Inner env that emits ``info["accept"]`` (typically an
            :class:`AutomatonAugmentedEnvWrapper`).
        kind: One of ``"sparse_accept"``, ``"dense_accept_prob"``,
            or ``"none"`` (identity).
        scale: Multiplier for ``dense_accept_prob`` only; ignored for
            ``sparse_accept`` / ``none``.
    """

    def __init__(self, env: Any, kind: str, scale: float = 1.0) -> None:
        super().__init__(env)
        self._shaping_fn = _make_shaping_fn(kind, scale)
        self._kind = kind
        self._scale = float(scale)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: jax.Array, params: Any = None):
        obs, inner_state = self._env.reset(key, params)
        return obs, _ShapingState(
            env_state=inner_state, prev_accept=jnp.float32(0.0),
        )

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key: jax.Array, state: _ShapingState, action: Any, params: Any = None):
        obs, inner_state, reward, done, info = self._env.step(
            key, state.env_state, action, params,
        )
        accept = info["accept"]
        shaped = self._shaping_fn(reward, done, state.prev_accept, accept)
        # ``AutomatonAugmentedEnvWrapper`` already overwrites ``info["accept"]``
        # with the initial-state accept on ``done=True``, so a single
        # ``next_prev = accept`` is correct for both branches.
        new_state = _ShapingState(env_state=inner_state, prev_accept=accept)
        return obs, new_state, shaped, done, info
