#!/usr/bin/env bash
# Train PPO + RND on Craftax-Symbolic-v1.
# Recommended for sparse-reward task runs (the task reward is 0/1).
#
# Usage:
#   ./train_ppo_rnd.sh                   # baseline (default Craftax reward + RND)
#   ./train_ppo_rnd.sh collect_wood      # task: reward fires only on `collect_wood`
#
# Overridable env vars:
#   TOTAL_TIMESTEPS  default 2e8
#   NUM_ENVS         default 1024
#   SEED             default 0
#   USE_WANDB        default 1   (set 0 to disable)
#   WANDB_PROJECT    default craftx-baselines
#   WANDB_ENTITY     default unset (uses wandb's default for your account)

set -euo pipefail

TARGET="${1:-}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2e8}"
NUM_ENVS="${NUM_ENVS:-1024}"
SEED="${SEED:-0}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-craftx-baselines}"

ARGS=(
    --env_name Craftax-Symbolic-v1
    --total_timesteps "$TOTAL_TIMESTEPS"
    --num_envs "$NUM_ENVS"
    --seed "$SEED"
)

if [[ -z "$TARGET" ]]; then
    RUN_NAME="ppo-rnd-baseline"
else
    ARGS+=(--target_achievement "$TARGET" --task_terminate_on_complete)
    RUN_NAME="ppo-rnd-task-${TARGET}"
fi

if [[ "$USE_WANDB" == "1" ]]; then
    ARGS+=(--wandb_project "$WANDB_PROJECT")
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        ARGS+=(--wandb_entity "$WANDB_ENTITY")
    fi
else
    ARGS+=(--no-use_wandb)
fi

cd "$(dirname "$0")"
echo "==> Running: python ppo_rnd.py ${ARGS[*]}"
exec python ppo_rnd.py "${ARGS[@]}"
