#!/usr/bin/env bash
# Sweep launcher tuned for 2x RTX A6000 Ada (48 GB, ~91 TFLOPs FP32 each).
#
# Defaults:
#   - TOTAL_TIMESTEPS=2e8 per run
#   - 7 task achievements (defeat_gnome_archer, cast_fireball, make_diamond_pickaxe
#     dropped from the test set as too hard for the budget)
#   - 3 algorithms (ppo, ppo_rnd, ppo_rnn)
#   - 3 baselines (one per algo, no --target_achievement)
#   - 2 concurrent jobs per GPU = 4 parallel slots total
#   - 1024 envs across all algos
#   - Reserve 8 GB / card for other users (40 GB usable per card)
#
# Total: 24 runs (21 task + 3 baseline). Estimated wall-clock: ~11-13 h.
#
# Usage:
#   ./run_sweep.sh                                      # full default sweep
#   JOBS_PER_GPU=3 ./run_sweep.sh                       # more aggressive (~10 h, more contention)
#   JOBS_PER_GPU=1 ./run_sweep.sh                       # safe single-stream (~18 h)
#   RESERVED_GB=0 ./run_sweep.sh                        # use full 48 GB (no co-tenants)
#   RESERVED_GB=16 ./run_sweep.sh                       # reserve more for others
#   CARD_GB=24 ./run_sweep.sh                           # if running on 24 GB cards
#   INCLUDE_BASELINES=0 ./run_sweep.sh                  # tasks only (~10 h)
#   TOTAL_TIMESTEPS=1e8 ./run_sweep.sh                  # half budget (~6 h)

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2e8}"
NUM_GPUS="${NUM_GPUS:-2}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
NUM_ENVS="${NUM_ENVS:-1024}"
RNN_NUM_ENVS="${RNN_NUM_ENVS:-1024}"
INCLUDE_BASELINES="${INCLUDE_BASELINES:-1}"
CARD_GB="${CARD_GB:-48}"
RESERVED_GB="${RESERVED_GB:-8}"
SEED="${SEED:-0}"

# Per-job XLA mem fraction = (usable VRAM per card) / JOBS_PER_GPU.
# Usable VRAM = CARD_GB - RESERVED_GB. Total per card stays under (CARD_GB - RESERVED_GB)
# so co-tenants always have at least RESERVED_GB free.
MEM_FRAC=$(awk "BEGIN { printf \"%.2f\", ($CARD_GB - $RESERVED_GB) / $CARD_GB / $JOBS_PER_GPU }")
USABLE_GB=$(awk "BEGIN { printf \"%.1f\", $CARD_GB - $RESERVED_GB }")
PER_JOB_GB=$(awk "BEGIN { printf \"%.1f\", ($CARD_GB - $RESERVED_GB) / $JOBS_PER_GPU }")

# Sanity: leave a tiny floor so we never hand a job <2 GB
if awk "BEGIN { exit !($PER_JOB_GB < 2.0) }"; then
    echo "ERROR: per-job VRAM = ${PER_JOB_GB} GB is too small. Reduce JOBS_PER_GPU or RESERVED_GB." >&2
    exit 1
fi

# 7 test-set tasks (dropped: defeat_gnome_archer, cast_fireball, make_diamond_pickaxe)
TASKS=(
    collect_wood eat_cow place_table place_stone
    make_stone_sword defeat_kobold fire_bow
)

QUEUE=$(mktemp -p . sweep_queue.XXXX)
LOCK="${QUEUE}.lock"
trap 'rm -f "$QUEUE" "$LOCK"' EXIT

# Queue order: longest-running first (RNN), so they don't trail the sweep.
# Each line: "<script> <num_envs> <task-or-dash>"
for t in "${TASKS[@]}"; do
    echo "train_ppo_rnn.sh $RNN_NUM_ENVS $t" >> "$QUEUE"
done
for t in "${TASKS[@]}"; do
    echo "train_ppo_rnd.sh $NUM_ENVS $t" >> "$QUEUE"
done
for t in "${TASKS[@]}"; do
    echo "train_ppo.sh $NUM_ENVS $t" >> "$QUEUE"
done

if [[ "$INCLUDE_BASELINES" == "1" ]]; then
    echo "train_ppo_rnn.sh $RNN_NUM_ENVS -" >> "$QUEUE"
    echo "train_ppo_rnd.sh $NUM_ENVS -"     >> "$QUEUE"
    echo "train_ppo.sh     $NUM_ENVS -"     >> "$QUEUE"
fi

TOTAL_JOBS=$(wc -l < "$QUEUE")
TOTAL_SLOTS=$((NUM_GPUS * JOBS_PER_GPU))

cat <<EOF
==> Sweep configuration
    GPUs:           $NUM_GPUS  (jobs/GPU = $JOBS_PER_GPU, total slots = $TOTAL_SLOTS)
    Card VRAM:      ${CARD_GB} GB total / ${USABLE_GB} GB usable / ${RESERVED_GB} GB reserved for co-tenants
    Per-job VRAM:   ${MEM_FRAC} fraction (~${PER_JOB_GB} GB)
    Total runs:     $TOTAL_JOBS  ($(grep -v '\-$' "$QUEUE" | wc -l) task + $(grep '\-$' "$QUEUE" | wc -l) baseline)
    Steps/run:      $TOTAL_TIMESTEPS
    Tasks:          ${TASKS[*]}
    Logs:           logs/<algo>_<task>.log
==> Starting workers...
EOF

pop_job() {
    flock -x 9
    local line
    line=$(head -n 1 "$QUEUE")
    if [[ -n "$line" ]]; then
        sed -i '1d' "$QUEUE"
        printf '%s\n' "$line"
    fi
} 9>"$LOCK"

worker() {
    local gpu=$1 slot=$2
    while true; do
        local job
        job=$(pop_job)
        [[ -z "$job" ]] && break
        # shellcheck disable=SC2086
        set -- $job
        local script=$1 num_envs=$2 task=$3
        local algo=${script#train_}; algo=${algo%.sh}

        local cmd=("./$script")
        local label
        if [[ "$task" == "-" ]]; then
            label="${algo}_baseline"
        else
            label="${algo}_${task}"
            cmd+=("$task")
        fi
        local log="logs/${label}_seed${SEED}.log"

        local started=$(date +%s)
        echo "[gpu=$gpu slot=$slot] START $label @ $(date '+%H:%M:%S')"
        CUDA_VISIBLE_DEVICES=$gpu \
        TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS \
        NUM_ENVS=$num_envs \
        SEED=$SEED \
        XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRAC \
        WANDB_NAME="$label" \
            "${cmd[@]}" >"$log" 2>&1 \
            && status="OK" || status="FAIL($?)"
        local elapsed=$(( $(date +%s) - started ))
        printf '[gpu=%d slot=%d] %s %s in %dh%02dm (log: %s)\n' \
            "$gpu" "$slot" "$status" "$label" $((elapsed/3600)) $((elapsed%3600/60)) "$log"
    done
    echo "[gpu=$gpu slot=$slot] queue empty, exiting"
}

SWEEP_START=$(date +%s)
for ((g=0; g<NUM_GPUS; g++)); do
    for ((s=0; s<JOBS_PER_GPU; s++)); do
        worker "$g" "$s" &
    done
done
wait

TOTAL=$(( $(date +%s) - SWEEP_START ))
echo ""
echo "==> Sweep finished in $((TOTAL/3600))h$((TOTAL%3600/60))m"
echo "==> Failed runs (if any):"
grep -l "Error\|Traceback" logs/*.log 2>/dev/null || echo "    (none)"
