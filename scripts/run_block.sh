#!/usr/bin/env bash
# run_block.sh -- worker pool over a single (experiments x variant x tasks x seeds) block.
#
# Usage:
#   ./run_block.sh EXPERIMENTS VARIANT TASKS SEEDS
#
#   EXPERIMENTS  comma-separated experiment names (matches conf/experiment/*.yaml)
#   VARIANT      one of {ppo, ppo_rnn, ppo_rnd}
#   TASKS        comma-separated target_achievement values
#   SEEDS        comma-separated seed integers
#
# Required env vars:
#   NUM_GPUS        e.g. 8
#   JOBS_PER_GPU    e.g. 2
#   MEM_FRAC        e.g. 0.45  (XLA_PYTHON_CLIENT_MEM_FRACTION per worker)
#
# Optional env vars:
#   TOTAL_TIMESTEPS   default 2e8
#   NUM_ENVS          default 1024
#   WANDB_PROJECT     default craftx-baselines
#   USE_WANDB         default 1   (set 0 to disable wandb)
#   GPU_OFFSET        default 0   (skip the first N GPUs; useful when sharing a host)
#
# Example:
#   NUM_GPUS=8 JOBS_PER_GPU=2 MEM_FRAC=0.45 \
#     ./run_block.sh "brzozowski_continuous,brzozowski_env_only" ppo \
#                    "collect_wood,eat_cow" "0,1,2"

set -euo pipefail

if [ $# -lt 4 ]; then
    sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    exit 2
fi

EXPERIMENTS=$1
VARIANT=$2
TASKS=$3
SEEDS=$4

: "${NUM_GPUS:?NUM_GPUS is required}"
: "${JOBS_PER_GPU:?JOBS_PER_GPU is required}"
: "${MEM_FRAC:?MEM_FRAC is required}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2e8}"
NUM_ENVS="${NUM_ENVS:-1024}"
WANDB_PROJECT="${WANDB_PROJECT:-AFA-final}"
USE_WANDB="${USE_WANDB:-1}"
GPU_OFFSET="${GPU_OFFSET:-0}"

# Resolve key paths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_REPO="$(cd "$CB_ROOT/../.." && pwd)"
PYTHON="$PARENT_REPO/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "Error: $PYTHON not found or not executable" >&2
    exit 1
fi

# Block tag for log dir + queue file naming.
EXP_HASH=$(printf '%s' "$EXPERIMENTS" | md5sum | cut -c1-8)
BLOCK_TAG="${VARIANT}_${EXP_HASH}"
LOG_DIR="$SCRIPT_DIR/logs/$BLOCK_TAG"
mkdir -p "$LOG_DIR"

QUEUE=$(mktemp -p "$SCRIPT_DIR" "block_queue_${BLOCK_TAG}.XXXX")
LOCK="${QUEUE}.lock"
trap 'rm -f "$QUEUE" "$LOCK"' EXIT

# Build the cross-product queue. One job per row: "<exp> <task> <seed>".
IFS=',' read -ra EXPS_ARR <<< "$EXPERIMENTS"
IFS=',' read -ra TASKS_ARR <<< "$TASKS"
IFS=',' read -ra SEEDS_ARR <<< "$SEEDS"

for exp in "${EXPS_ARR[@]}"; do
    for task in "${TASKS_ARR[@]}"; do
        for seed in "${SEEDS_ARR[@]}"; do
            echo "$exp $task $seed" >> "$QUEUE"
        done
    done
done

TOTAL_JOBS=$(wc -l < "$QUEUE")
TOTAL_SLOTS=$((NUM_GPUS * JOBS_PER_GPU))

cat <<EOF
==> run_block.sh: $BLOCK_TAG
    experiments:   $EXPERIMENTS
    variant:       $VARIANT
    tasks:         $TASKS
    seeds:         $SEEDS
    total jobs:    $TOTAL_JOBS
    slots:         $TOTAL_SLOTS  ($NUM_GPUS GPUs x $JOBS_PER_GPU jobs/GPU, offset=$GPU_OFFSET)
    mem fraction:  $MEM_FRAC
    timesteps:     $TOTAL_TIMESTEPS
    num envs:      $NUM_ENVS
    logs:          $LOG_DIR/<task>_seed<N>.log
EOF

pop_job() {
    flock -x 9
    local line
    line=$(head -n 1 "$QUEUE")
    if [ -n "$line" ]; then
        sed -i '1d' "$QUEUE"
        printf '%s\n' "$line"
    fi
} 9>"$LOCK"

FAIL_FLAG=$(mktemp -p "$SCRIPT_DIR" "block_fail_${BLOCK_TAG}.XXXX")
trap 'rm -f "$QUEUE" "$LOCK" "$FAIL_FLAG" "$FAIL_FLAG.failed"' EXIT

worker() {
    local gpu=$1 slot=$2
    while true; do
        local job
        job=$(pop_job)
        [ -z "$job" ] && break
        # shellcheck disable=SC2086
        set -- $job
        local exp=$1 task=$2 seed=$3
        local log="$LOG_DIR/${task}_seed${seed}_${exp}.log"

        local started
        started=$(date +%s)
        echo "[gpu=$gpu slot=$slot] START $exp/$task/seed$seed @ $(date '+%H:%M:%S')"

        local cmd=(
            "$PYTHON" "$CB_ROOT/train.py"
            "experiment=$exp"
            "variant=$VARIANT"
            "target_achievement=$task"
            "seed=$seed"
            "task_terminate_on_complete=true"
            "total_timesteps=$TOTAL_TIMESTEPS"
            "num_envs=$NUM_ENVS"
        )
        if [ "$USE_WANDB" = "1" ]; then
            cmd+=("wandb_project=$WANDB_PROJECT")
        else
            cmd+=("use_wandb=false")
        fi

        local status
        if CUDA_VISIBLE_DEVICES=$gpu \
           XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRAC \
              "${cmd[@]}" >"$log" 2>&1; then
            status="OK"
        else
            status="FAIL($?)"
            : > "$FAIL_FLAG.failed"
        fi

        local elapsed=$(( $(date +%s) - started ))
        printf '[gpu=%d slot=%d] %s %s/%s/seed%s in %dh%02dm (log: %s)\n' \
            "$gpu" "$slot" "$status" "$exp" "$task" "$seed" \
            $((elapsed/3600)) $((elapsed%3600/60)) "$log"
    done
    echo "[gpu=$gpu slot=$slot] queue empty, exiting"
}

BLOCK_START=$(date +%s)
for ((g=0; g<NUM_GPUS; g++)); do
    gpu=$((g + GPU_OFFSET))
    for ((s=0; s<JOBS_PER_GPU; s++)); do
        worker "$gpu" "$s" &
    done
done
wait

ELAPSED=$(( $(date +%s) - BLOCK_START ))
echo
echo "==> block $BLOCK_TAG finished in $((ELAPSED/3600))h$((ELAPSED%3600/60))m"

if [ -f "$FAIL_FLAG.failed" ]; then
    echo "==> some jobs failed; see logs in $LOG_DIR"
    exit 1
fi
