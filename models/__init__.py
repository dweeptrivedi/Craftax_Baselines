"""Models package init.

Installs the parent repo's ``rad_comparison._distrax_shim`` before any
submodule-level ``import distrax`` runs. ``distrax`` 0.1.5 (currently
locked) imports ``tensorflow_probability``, which uses removed JAX
internals (``jax.interpreters.xla.pytype_aval_mappings``) on JAX
:math:`\\geq 0.5`. The shim provides the only ``distrax`` symbol the
trainers use (``Categorical``) and bypasses the broken TFP path.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PARENT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(_PARENT_SRC))

try:
    from rad_comparison._distrax_shim import install_shim

    install_shim()
except ImportError:
    pass
