#!/usr/bin/env python3
r"""Build the RAD lookup table from precomputed per-task ``.npz`` files.

Reads each ``outputs/rad_embeddings/<task>.npz`` produced by
:file:`scripts/rad_embed_all.py` and assembles a single ``.npz`` containing
per-task ``(transition, embedding, accept, initial_residual,
predicate_indices)`` arrays keyed by task name.

Output schema (per task ``n``):

- ``transition_<n>: (num_states, num_symbols) int32``
- ``embedding_<n>: (num_states, embed_dim) float32``
- ``accept_<n>: (num_states,) float32`` (cast from bool)
- ``initial_residual_<n>: () int32`` (index into the remapped state space)
- ``predicate_indices_<n>: (n_predicates,) int32``  (into ``Achievement.value``)

Plus top-level metadata: ``task_names: array[str]``, ``embed_dim: int32``.

Sidecar ``outputs/lookup_rad_manifest.json`` lists compatible task names
for the runner's task-validation step.

**Critical detail:** RAD's ``state_ids`` may not be contiguous (raw MONA
ids). We build a ``state_id_to_idx`` remap so the transition table
indexes ``[0, num_states)`` directly. Embeddings are already row-aligned
with ``state_ids`` so no remap is needed for them.

Usage::

    # Prerequisite: scripts/rad_embed_all.py has been run.
    python scripts/build_rad_lookup.py
    python scripts/build_rad_lookup.py --output outputs/lookup_rad.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# This file lives at ``<repo>/third_party/craftax_baselines/scripts/X.py`` after
# the restructure. ``parents[3]`` is the repo root; add its ``src/`` so
# ``brzozowski_dataset`` and ``rad_comparison`` are importable.
_PARENT_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(_PARENT_SRC))

from craftax.craftax.constants import Achievement

from brzozowski_dataset.envs.craftax.tech_tree import craftax_task_specs
from rad_comparison.convert import survey_rad_compatibility
from rad_comparison.embed import minimal_dfa_to_dfa


_PARENT_OUTPUTS = Path(__file__).resolve().parents[3] / "outputs"


def _build_for_task(npz_path: Path, dfa, n_tokens: int) -> dict[str, np.ndarray] | None:
    """Build per-task lookup arrays.

    Args:
        npz_path: Path to ``outputs/rad_embeddings/<task>.npz``.
        dfa: ``dfa.DFA`` reconstructed via ``minimal_dfa_to_dfa``.
        n_tokens: Alphabet token count (already padded to RAD's encoder).

    Returns:
        Dict with the per-task arrays. Returns ``None`` if any predicate
        name fails to resolve into the ``Achievement`` enum.
    """
    data = np.load(npz_path)
    raw_embeddings = data["embeddings"]  # (num_states, embed_dim)
    raw_accept = data["accepting_mask"]  # (num_states,) bool
    raw_state_ids = list(map(int, data["state_ids"]))  # (num_states,) int

    # Remap raw state ids to contiguous 0..N-1.
    state_id_to_idx = {sid: i for i, sid in enumerate(raw_state_ids)}
    num_states = len(raw_state_ids)

    # Reconstruct dense transition table by querying the DFA's transition
    # function at every (state, symbol) pair.
    transition = np.empty((num_states, n_tokens), dtype=np.int32)
    for s_idx, sid in enumerate(raw_state_ids):
        for sigma in range(n_tokens):
            next_sid = dfa._transition(sid, sigma)
            transition[s_idx, sigma] = state_id_to_idx[next_sid]

    return {
        "transition": transition,
        "embedding": raw_embeddings.astype(np.float32),
        "accept": raw_accept.astype(np.float32),
        "initial_residual": np.int32(state_id_to_idx[dfa.start]),
    }


def main() -> int:
    """Assemble the RAD lookup table for all compatible Craftax tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rad-dir",
        type=Path,
        default=_PARENT_OUTPUTS / "rad_embeddings",
        help="Directory of per-task .npz files from rad_embed_all.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PARENT_OUTPUTS / "lookup_rad.npz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_PARENT_OUTPUTS / "lookup_rad_manifest.json",
    )
    parser.add_argument(
        "--predicate-indices",
        type=Path,
        default=_PARENT_OUTPUTS / "task_predicates.json",
        help="task_predicates.json from derive_task_predicates.py (for predicate_indices).",
    )
    args = parser.parse_args()

    if not args.rad_dir.exists():
        print(
            f"Error: {args.rad_dir} not found. Run scripts/rad_embed_all.py first.",
            file=sys.stderr,
        )
        return 1
    if not args.predicate_indices.exists():
        print(
            f"Error: {args.predicate_indices} not found. "
            "Run scripts/derive_task_predicates.py first.",
            file=sys.stderr,
        )
        return 1

    task_predicates = json.loads(args.predicate_indices.read_text())

    specs = craftax_task_specs()
    survey = survey_rad_compatibility(specs)
    compat_results = survey.compatible
    print(f"Survey: {len(compat_results)}/{len(specs)} tasks RAD-compatible.")

    bundle: dict[str, np.ndarray] = {}
    task_names: list[str] = []
    embed_dim: int | None = None
    skipped: list[str] = []

    for result in compat_results:
        npz_path = args.rad_dir / f"{result.task_name}.npz"
        if not npz_path.exists():
            print(f"  [skip] {result.task_name}: {npz_path} missing")
            skipped.append(result.task_name)
            continue

        n_tokens = 2 ** len(result.dfa.predicates)
        # rad_embed_all.py used pad_tokens_to=10 by default; the dfa has
        # already been padded. Re-create at the same shape.
        dfa = minimal_dfa_to_dfa(result.dfa, pad_tokens_to=max(n_tokens, 10))

        per_task = _build_for_task(npz_path, dfa, n_tokens=max(n_tokens, 10))
        if per_task is None:
            skipped.append(result.task_name)
            continue

        if result.task_name not in task_predicates:
            print(
                f"  [skip] {result.task_name}: not in task_predicates.json",
            )
            skipped.append(result.task_name)
            continue
        per_task["predicate_indices"] = np.array(
            task_predicates[result.task_name]["predicate_indices"], dtype=np.int32,
        )

        if embed_dim is None:
            embed_dim = int(per_task["embedding"].shape[-1])

        for k, v in per_task.items():
            bundle[f"{k}_{result.task_name}"] = v
        task_names.append(result.task_name)
        print(
            f"  [ok]   {result.task_name}: states={per_task['transition'].shape[0]}, "
            f"symbols={per_task['transition'].shape[1]}, embed_dim={embed_dim}",
        )

    if not task_names:
        print("Error: no tasks could be built into the lookup.", file=sys.stderr)
        return 1

    bundle["task_names"] = np.array(task_names)
    bundle["embed_dim"] = np.int32(embed_dim or 0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **bundle)
    print(f"\nWrote lookup to {args.output}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "task_names": task_names,
                "embed_dim": embed_dim,
                "skipped": skipped,
                "num_compatible": len(compat_results),
                "num_total": len(specs),
            },
            indent=2,
        ),
    )
    print(f"Wrote manifest to {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
