r"""Map Craftax env_state to a symbol bitmask.

Per the plan (:file:`AFA_INFER_CRAFTAX_BASELINE.md`, "Symbol source"
section), the symbol comes from the FULL ``env_state`` — not from the
flat ``obs`` vector and not just from ``env_state.achievements``. The
default ``achievement_extractor`` covers all currently-defined Craftax
tech-tree tasks; custom Phase-2 tasks supply their own extractor.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import jax.numpy as jnp
from jaxtyping import Array, Int


class PredicateExtractor(Protocol):
    """Maps a Craftax ``env_state`` to a ``(n_pred,) bool`` array.

    The order is the per-task ``sorted_predicates`` ordering (alphabetical
    by predicate name) — the same ordering that was used to encode the
    bz-2.0 dataset's symbol bitmasks.
    """

    def __call__(self, env_state: Any) -> jnp.ndarray: ...


def achievement_extractor(predicate_indices: Int[Array, "n_pred"]) -> PredicateExtractor:
    """Default extractor: index into ``env_state.achievements``.

    Covers all standard Craftax tech-tree tasks (every ``Leaf(event)`` in
    ``tech_tree.py`` becomes ``Eventually(Variable(event))`` with ``event``
    an Achievement enum name).
    """
    def extract(env_state: Any) -> jnp.ndarray:
        # env_state.achievements: (len(Achievement),) bool, indexed by Achievement.value
        return env_state.achievements[predicate_indices]
    return extract


def make_predicate_evaluator(
    extractor: PredicateExtractor,
    n_pred: int,
) -> Callable[[Any], Int[Array, ""]]:
    """Build env_state -> int32 symbol bitmask.

    Encoding matches :func:`brzozowski_dataset.residuals._predicates_to_mask`:
    bit ``i`` of the returned mask is the truth value of
    ``sorted_predicates[i]``.

    Args:
        extractor: Maps env_state to ``(n_pred,) bool``.
        n_pred: Number of predicates for this task.

    Returns:
        Callable that consumes one env_state and returns a scalar int32.
    """
    bit_weights = (1 << jnp.arange(n_pred, dtype=jnp.int32))

    def evaluate(env_state: Any) -> jnp.ndarray:
        active = extractor(env_state).astype(jnp.int32)
        return jnp.sum(active * bit_weights).astype(jnp.int32)

    return evaluate
