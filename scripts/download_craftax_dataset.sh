#!/usr/bin/env bash
# Download the Craftax bz-2.0 dataset from Hugging Face into the
# craftax_baselines submodule's ``data/craftax/`` directory.
#
# Files fetched (all under https://huggingface.co/datasets/anand-bala/automata-embeddings):
#   craftax/dataset_v2.h5    -- main HDF5 corpus
#   craftax/task_splits.json -- train/val/test split indices
#   craftax/README.md        -- dataset description
#
# Idempotent: re-running skips already-downloaded files. Existing local
# files are not overwritten unless you pass ``--force``.
#
# Usage:
#   third_party/craftax_baselines/scripts/download_craftax_dataset.sh
#   third_party/craftax_baselines/scripts/download_craftax_dataset.sh --force
#   DATA_DIR=/scratch/foo third_party/craftax_baselines/scripts/download_craftax_dataset.sh
set -euo pipefail

# ``CB_ROOT`` = craftax_baselines submodule (one level up from scripts/).
# ``PARENT_REPO`` = parent automata-embeddings repo (two levels up from CB_ROOT).
CB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_REPO="$(cd "$CB_ROOT/../.." && pwd)"

DATA_DIR="${DATA_DIR:-$CB_ROOT/data/craftax}"
HF_REPO="anand-bala/automata-embeddings"
FILES=(craftax/dataset_v2.h5 craftax/task_splits.json craftax/README.md)
FORCE=0

for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help)
      sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$DATA_DIR"

# Pick a downloader. The submodule has no venv of its own, so we resolve ``uv``
# against the parent repo's venv (which is the active project venv for all
# craftax_baselines work). ``huggingface_hub`` >=1.0 ships the CLI as ``hf``
# (the legacy ``huggingface-cli`` was removed); fall back to either name on
# the system PATH for older installs.
if command -v uv >/dev/null 2>&1 && [ -d "$PARENT_REPO/.venv" ]; then
  HF_CMD=(uv run --project "$PARENT_REPO" --with huggingface-hub hf)
elif command -v hf >/dev/null 2>&1; then
  HF_CMD=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CMD=(huggingface-cli)
else
  echo "Error: neither 'uv' (with $PARENT_REPO/.venv) nor 'hf'/'huggingface-cli' is available." >&2
  echo "Install one of:" >&2
  echo "  - uv (https://docs.astral.sh/uv/) and run 'uv sync' in $PARENT_REPO, OR" >&2
  echo "  - pipx install huggingface-hub" >&2
  exit 1
fi

echo "Downloading craftax dataset from $HF_REPO -> $DATA_DIR"

for remote in "${FILES[@]}"; do
  basename="${remote#craftax/}"
  local_path="$DATA_DIR/$basename"
  if [ -f "$local_path" ] && [ "$FORCE" -eq 0 ]; then
    echo "  skip $basename (already present; pass --force to redownload)"
    continue
  fi
  echo "  fetching $remote"
  "${HF_CMD[@]}" download "$HF_REPO" "$remote" \
    --repo-type dataset \
    --local-dir "$DATA_DIR/.hf-tmp" \
    >/dev/null
  mv -f "$DATA_DIR/.hf-tmp/$remote" "$local_path"
done

rm -rf "$DATA_DIR/.hf-tmp"

echo
echo "Done. Files now in $DATA_DIR:"
ls -lh "$DATA_DIR" | sed 's/^/  /'
echo
echo "Reference these paths from Hydra overrides, e.g.:"
echo "  python train.py target_achievement=collect_wood task_predicates_path=$DATA_DIR/task_splits.json"
