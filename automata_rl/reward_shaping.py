r"""Reward shaping based on automaton acceptance.

Per the plan (:file:`AFA_INFER_CRAFTAX_BASELINE.md`, "Reward shaping"
section): both shaping functions mask out ``done=True`` to prevent the
phantom ``accept[initial] - accept[prev]`` artifact when the env
auto-resets.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


RewardShapingFn = Callable[
    [Float[Array, "B"], Bool[Array, "B"], Float[Array, "B"], Float[Array, "B"]],
    Float[Array, "B"],
]


def none_shaping(
    reward: Float[Array, "B"],
    done: Bool[Array, "B"],
    prev_accept: Float[Array, "B"],
    curr_accept: Float[Array, "B"],
) -> Float[Array, "B"]:
    """Identity — no reward shaping. Default."""
    return reward


def sparse_accept(
    reward: Float[Array, "B"],
    done: Bool[Array, "B"],
    prev_accept: Float[Array, "B"],
    curr_accept: Float[Array, "B"],
) -> Float[Array, "B"]:
    """+1 bonus on the FIRST step that arrives in an accepting residual.

    Both backends populate ``accept`` in ``[0, 1]``; we threshold at 0.5.
    """
    rising_edge = (curr_accept > 0.5) & (prev_accept <= 0.5)
    bonus = (rising_edge & ~done).astype(reward.dtype)
    return reward + bonus


def dense_accept_prob(
    reward: Float[Array, "B"],
    done: Bool[Array, "B"],
    prev_accept: Float[Array, "B"],
    curr_accept: Float[Array, "B"],
    scale: float = 1.0,
) -> Float[Array, "B"]:
    """Differential of accept-probability. Brzozowski-only (continuous accept).

    For RAD the accept channel is binary 0/1; using this fn there reduces
    to a flickering :math:`\\pm 1` shaping which is poorly conditioned.
    The runner errors when ``embedding=rad_lookup`` is paired with
    ``reward_shaping=dense_accept_prob``.
    """
    delta = curr_accept - prev_accept
    return reward + scale * jnp.where(done, 0.0, delta)


SHAPING_FNS: dict[str, RewardShapingFn] = {
    "none": none_shaping,
    "sparse_accept": sparse_accept,
    "dense_accept_prob": dense_accept_prob,
}


def get_shaping_fn(name: str) -> RewardShapingFn:
    """Look up a reward-shaping function by name (Hydra config-friendly)."""
    if name not in SHAPING_FNS:
        raise ValueError(
            f"Unknown reward_shaping {name!r}; expected one of {list(SHAPING_FNS)}",
        )
    return SHAPING_FNS[name]
