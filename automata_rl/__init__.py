r"""RL integration glue between Craftax_Baselines and the AFA embedding model.

Lives inside the ``craftax_baselines`` submodule so the RL code is
co-located with the PPO trainers it plugs into. The JAX port of the
Brzozowski encoder (:mod:`automata_jax`) stays in the parent repo's
``src/``; this package adds it to ``sys.path`` at import time so the
embedding-setup module can ``from automata_jax import ...`` regardless
of where the script that imports us was launched from.

Modules:

- :mod:`.predicate_eval` -- env_state -> symbol bitmask
- :mod:`.wrappers` -- ``AutomatonAugmentedEnvWrapper`` + backend abstractions
  + ``AcceptRewardShapingWrapper``
- :mod:`.reward_shaping` -- sparse/dense accept-based reward bonuses
- :mod:`.embedding_setup` -- ``build_embedding_stack`` factory used by
  ``ppo.py`` / ``ppo_rnn.py`` / ``ppo_rnd.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``automata_jax`` (in the parent repo's ``src/``) importable. ``__file__``
# is ``<repo>/third_party/craftax_baselines/automata_rl/__init__.py``, so
# ``parents[3]`` is the repo root and ``+ "src"`` is where ``automata_jax``
# lives.
_PARENT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(_PARENT_SRC))
