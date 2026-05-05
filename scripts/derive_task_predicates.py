#!/usr/bin/env python3
"""Derive per-task predicate orderings for the standard Craftax tech tree.

For each task in :func:`brzozowski_dataset.envs.craftax.tech_tree.craftax_task_specs`,
extracts the sorted list of free variables from the LTLf formula via
``formula.atomic_predicates()``. For the standard Craftax tech tree this
matches what :func:`brzozowski_dataset.residuals._derive_predicates` would
return after AFA enumeration -- every variable in an ``Eventually(...)``
clause appears in some reachable transition. We use the formula-direct path
because the AFA-enumeration path is orders of magnitude slower (45s+ for a
10-predicate task; would take hours for the full 67-task set), and gives
the same answer for every task in the standard tech tree.

Each predicate is then mapped to its ``Achievement.value`` so the JAX-side
``predicate_evaluator`` can index directly into ``env_state.achievements``.

Output schema (``outputs/task_predicates.json``)::

    {
      "<task_name>": {
        "sorted_predicates": ["collect_wood", ...],
        "predicate_indices": [3, 7, ...],
        "n_predicates": 5,
        "task_idx": 0
      },
      ...
    }

The ``task_idx`` field is the position of the spec in
``craftax_task_specs()``, which is what the trained ``AFAEmbedding``'s
``task_encoder`` uses as its embedding index. It is required by
:mod:`automata_rl.embedding_setup` to instantiate
:class:`automata_rl.wrappers.JaxBrzozowskiBackend` for the chosen task.

Note: ``n_states`` (AFA state count) is NOT emitted here because computing it
requires the slow AFA-enumeration path. The trained model carries its own
``num_vars`` budget; downstream code should use that for the wrapper's
``q_state`` shape rather than relying on per-task state counts.

Usage::

    python scripts/derive_task_predicates.py
    python scripts/derive_task_predicates.py --output custom/path.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This file lives at ``<repo>/third_party/craftax_baselines/scripts/X.py`` after
# the restructure. ``parents[3]`` is the repo root; add its ``src/`` so
# ``brzozowski_dataset`` is importable when this script is run from anywhere.
_PARENT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(_PARENT_SRC))

from craftax.craftax.constants import Achievement

from brzozowski_dataset.envs.craftax.tech_tree import craftax_task_specs


_DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "outputs" / "task_predicates.json"


def derive_for_spec(spec: object) -> dict[str, object] | None:
    """Extract sorted predicate names + Achievement-value indices.

    Args:
        spec: A ``brzozowski_dataset.types.TaskSpec`` instance.

    Returns:
        Dict with ``sorted_predicates``, ``predicate_indices``, ``n_predicates``.
        Returns ``None`` if any predicate name is not in the ``Achievement``
        enum (caller logs and skips).
    """
    sorted_predicates = sorted({v.name for v in spec.formula.atomic_predicates()})

    predicate_indices: list[int] = []
    for name in sorted_predicates:
        try:
            predicate_indices.append(Achievement[name.upper()].value)
        except KeyError:
            return None

    return {
        "sorted_predicates": sorted_predicates,
        "predicate_indices": predicate_indices,
        "n_predicates": len(sorted_predicates),
    }


def main() -> int:
    """Derive ``task_predicates.json`` for every Craftax task spec."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output path. Default resolves to {_DEFAULT_OUTPUT} (parent repo's outputs/).",
    )
    args = parser.parse_args()

    specs = craftax_task_specs()
    print(f"Processing {len(specs)} Craftax task specs...")

    results: dict[str, dict[str, object]] = {}
    skipped: list[str] = []
    for task_idx, spec in enumerate(specs):
        info = derive_for_spec(spec)
        if info is None:
            skipped.append(spec.name)
            print(f"  [skip] {spec.name}: predicate name not in Achievement enum")
            continue
        info["task_idx"] = task_idx
        results[spec.name] = info
        print(f"  [ok]   {spec.name}: task_idx={task_idx} n_pred={info['n_predicates']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} task entries to {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} tasks: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
