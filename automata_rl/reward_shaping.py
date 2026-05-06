r"""Term functions for the unified reward composition.

Each function returns the *raw accept-derived term* (not a shaped
reward) given ``(done, prev_accept, curr_accept)``. The
:class:`automata_rl.wrappers.RewardCompositionWrapper` linearly
combines this term with the inner env reward using user-supplied
coefficients:

.. math::

    \text{reward} = \text{env\_coef} \cdot \text{env\_reward}
                    + \text{accept\_coef} \cdot \text{term}(\text{done},
                                                            \text{prev},
                                                            \text{curr}).

History note: the previous ``& ~done`` / ``where(done, 0, ...)`` masks
on ``sparse_accept`` and ``dense_accept_prob`` were introduced to work
around a since-fixed accept-reset bug in
:class:`automata_rl.wrappers.AutomatonAugmentedEnvWrapper` (which used
to overwrite the live accept with the initial-state accept on
``done=True``). With ``info["embedding/accept"]`` now carrying the
live ``pre_reset_accept`` on every step, the masks are removed: a
successful ``done`` step *should* fire the rising-edge / delta reward.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


TermFn = Callable[
    [Bool[Array, "B"], Float[Array, "B"], Float[Array, "B"]],
    Float[Array, "B"],
]


def none_term(
    done: Bool[Array, "B"],
    prev_accept: Float[Array, "B"],
    curr_accept: Float[Array, "B"],
) -> Float[Array, "B"]:
    """Disable the accept term."""
    del done, prev_accept, curr_accept
    return jnp.float32(0.0)


def sparse_accept(
    done: Bool[Array, "B"],
    prev_accept: Float[Array, "B"],
    curr_accept: Float[Array, "B"],
) -> Float[Array, "B"]:
    """``1.0`` on the first step that arrives in an accepting residual.

    Threshold at ``0.5`` (both backends populate ``accept`` in
    ``[0, 1]``). Fires on the rising edge regardless of ``done``.
    """
    del done
    rising_edge = (curr_accept > 0.5) & (prev_accept <= 0.5)
    return rising_edge.astype(jnp.float32)


def dense_accept_prob(
    done: Bool[Array, "B"],
    prev_accept: Float[Array, "B"],
    curr_accept: Float[Array, "B"],
) -> Float[Array, "B"]:
    """Telescoping delta of accept probability.

    Cumulative reward over an episode equals
    ``curr_accept_T - curr_accept_0 = curr_accept_T``, so the floor
    cancels out by construction. Brzozowski-only — for RAD's binary
    accept the delta would flicker, so the runner errors when
    ``embedding=rad_lookup`` is paired with
    ``accept_reward_kind=dense_accept_prob``.
    """
    del done
    return curr_accept - prev_accept


def continuous(
    done: Bool[Array, "B"],
    prev_accept: Float[Array, "B"],
    curr_accept: Float[Array, "B"],
) -> Float[Array, "B"]:
    """Raw model belief: term equals the current accept score."""
    del done, prev_accept
    return curr_accept


def continuous_relu(
    done: Bool[Array, "B"],
    prev_accept: Float[Array, "B"],
    curr_accept: Float[Array, "B"],
) -> Float[Array, "B"]:
    """``curr_accept`` thresholded at ``0.1``; below threshold returns 0.

    The hard threshold zeroes the per-step floor reward that an
    imperfectly-calibrated model would otherwise emit in non-accepting
    states.
    """
    del done, prev_accept
    return jnp.where(curr_accept >= 0.1, curr_accept, jnp.float32(0.0))


TERM_FNS: dict[str, TermFn] = {
    "none": none_term,
    "sparse_accept": sparse_accept,
    "dense_accept_prob": dense_accept_prob,
    "continuous": continuous,
    "continuous_relu": continuous_relu,
}
