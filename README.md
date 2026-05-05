<p align="center">
 <img width="80%" src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax_Baselines/main/images/logo.png" />
</p>

# Craftax Baselines

This repository contains the code for running the baselines from the [Craftax paper](https://arxiv.org/abs/2402.16801).
For packaging reasons, this is separate to the [main repository](https://github.com/MichaelTMatthews/Craftax/).

# Installation
```commandline
git clone https://github.com/MichaelTMatthews/Craftax_Baselines.git
cd Craftax_Baselines
pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pre-commit install
```

# Run Experiments

### PPO
```commandline
python ppo.py
```

### PPO-RNN
```commandline
python ppo_rnn.py
```

### ICM
```commandline
python ppo.py --train_icm
```

### E3B
```commandline
python ppo.py --train_icm --use_e3b --icm_reward_coeff 0
```

### RND
```commandline
python ppo_rnd.py
```

# Visualisation
You can save trained policies with the `--save_policy` flag.  These can then be viewed with the `view_ppo_agent` script (pass in the path up to the `files` directory).

# Automaton-embedding-conditioned PPO (this fork)

This fork wires the per-task automaton embedding into `ppo.py`,
`ppo_rnn.py`, and `ppo_rnd.py`. When enabled, the env's flat observation
is concatenated with a per-task automaton fingerprint and an acceptance
scalar derived from the task's LTLf formula; the RL policy sees
`[craftax_obs | fingerprint | accept]` and learns to use the conditioning
signal as it sees fit (UVFA-style flat conditioning).

The RL-side glue lives in `automata_rl/` (a sibling subpackage of the
runners) and the JAX port of the encoder lives in the parent repo at
`<repo>/src/automata_jax/`. `automata_rl/__init__.py` adds that `src/`
directory to `sys.path` at import time, so the runners just do
`from automata_rl.embedding_setup import build_embedding_stack` and
everything resolves.

## Two backends

| `--embedding_kind` | What it does | Coverage |
|---|---|---|
| `none` (default) | No embedding; standard PPO. | n/a |
| `brzozowski_jax` | Live JAX inference of the trained `AFAEmbedding` model on every env step. Computes a `(M=512,)` fingerprint plus a scalar accept value. | All 67 standard Craftax tasks. |
| `rad_lookup` | Precomputed `(num_states, embed_dim)` lookup table indexed by a discrete DFA state. | Subset of tasks (the per-task `.npz` file must exist; lookup-table build needs the MONA binary). |

The conditioning architecture is always flat-concat: the augmented obs
flows through the upstream policy network unchanged (`ActorCritic` for
`ppo.py`, `ActorCriticRNN` for `ppo_rnn.py`, `ActorCriticRND` for
`ppo_rnd.py`).

## CLI args

```text
--embedding_kind {none, brzozowski_jax, rad_lookup}     default none
--brzozowski_params_path PATH                           default outputs/brzozowski_jax_params.msgpack
--brzozowski_config_path PATH                           default outputs/brzozowski_jax_config.yaml
--brzozowski_eval_points_path PATH                      default outputs/brzozowski_jax_eval_points.npy
--rad_lookup_path PATH                                  default outputs/lookup_rad.npz
--task_predicates_path PATH                             default outputs/task_predicates.json
--reward_shaping {none, sparse_accept, dense_accept_prob}   default none
--reward_shaping_scale FLOAT                            default 1.0 (only used by dense_accept_prob)
```

The default-path values are relative; when running from inside
`third_party/craftax_baselines/`, pass absolute paths into the parent
repo's `outputs/` directory.

## Hard constraints (errors at make_train time)

- **`--embedding_kind != none` requires `--target_achievement <name>`.**
  The default Craftax sum-of-achievements reward is not a single LTLf
  task, and feeding it through a per-task AFA fingerprint is semantically
  meaningless. `make_train` raises `ValueError` and aborts.
- **`--reward_shaping dense_accept_prob` is incompatible with
  `--embedding_kind rad_lookup`.** RAD's accept channel is binary 0/1,
  so the shaping bonus would flicker. Use `sparse_accept` or `none` with
  RAD.

## One-time artefact build

The artefact-builder scripts live in this submodule's `scripts/`. They
write into the parent repo's `outputs/` directory by default (resolved
relative to each script's own `__file__`), so you can run them from
anywhere.

```bash
# 1. Per-task predicate orderings + task_idx (committed, ~10 s)
python scripts/derive_task_predicates.py

# 2. Brzozowski-JAX backend (gitignored, ~30 s, requires the torch ckpt)
#    The converter stays in the PARENT repo because it's about converting
#    a PyTorch Lightning checkpoint, not RL.
.venv/bin/python ../../scripts/convert_brzozowski_to_jax.py \
    --checkpoint /path/to/last.ckpt \
    --hdf5 /path/to/dataset_v2.h5 \
    --params-output      ../../outputs/brzozowski_jax_params.msgpack \
    --eval-points-output ../../outputs/brzozowski_jax_eval_points.npy \
    --config-output      ../../outputs/brzozowski_jax_config.yaml \
    --device cpu

# 3. RAD lookup table (gitignored, requires the MONA binary system-wide)
uv sync --group rad
python scripts/build_rad_lookup.py --rad-dir /path/to/rad_per_task_npz_dir
```

If MONA is unavailable, skip step 3 and use `--embedding_kind brzozowski_jax`
only. The 67 standard Craftax tasks are all covered by the Brzozowski path.

## Example experiment commands

These run from `third_party/craftax_baselines/` (so `task_env.py`,
`wrappers.py`, `logz/`, `models/`, `automata_rl/` resolve as siblings)
and pass absolute paths into the parent repo's `outputs/`:

```bash
# Substitute REPO with your actual path:
REPO=/path/to/automata-embeddings
cd "$REPO/third_party/craftax_baselines"

# 1) Stock PPO baseline on a single-achievement task (no embedding).
python ppo.py \
    --env_name Craftax-Symbolic-v1 \
    --target_achievement collect_wood --task_terminate_on_complete \
    --embedding_kind none \
    --total_timesteps 200000 --num_envs 64 --no-use_wandb

# 2) Brzozowski-JAX-conditioned PPO.
python ppo.py \
    --env_name Craftax-Symbolic-v1 \
    --target_achievement collect_wood --task_terminate_on_complete \
    --embedding_kind brzozowski_jax \
    --total_timesteps 200000 --num_envs 64 --no-use_wandb \
    --task_predicates_path           "$REPO/outputs/task_predicates.json" \
    --brzozowski_params_path         "$REPO/outputs/brzozowski_jax_params.msgpack" \
    --brzozowski_config_path         "$REPO/outputs/brzozowski_jax_config.yaml" \
    --brzozowski_eval_points_path    "$REPO/outputs/brzozowski_jax_eval_points.npy"

# 3) Brzozowski-JAX + dense reward shaping.
python ppo.py \
    --env_name Craftax-Symbolic-v1 \
    --target_achievement collect_wood --task_terminate_on_complete \
    --embedding_kind brzozowski_jax \
    --reward_shaping dense_accept_prob --reward_shaping_scale 0.1 \
    --total_timesteps 200000 --num_envs 64 --no-use_wandb \
    --task_predicates_path           "$REPO/outputs/task_predicates.json" \
    --brzozowski_params_path         "$REPO/outputs/brzozowski_jax_params.msgpack" \
    --brzozowski_config_path         "$REPO/outputs/brzozowski_jax_config.yaml" \
    --brzozowski_eval_points_path    "$REPO/outputs/brzozowski_jax_eval_points.npy"

# 4) RAD lookup (only if you have built the lookup with MONA).
python ppo.py \
    --env_name Craftax-Symbolic-v1 \
    --target_achievement collect_wood --task_terminate_on_complete \
    --embedding_kind rad_lookup \
    --total_timesteps 200000 --num_envs 64 --no-use_wandb \
    --task_predicates_path "$REPO/outputs/task_predicates.json" \
    --rad_lookup_path      "$REPO/outputs/lookup_rad.npz"

# 5) ppo_rnn.py with embedding.
python ppo_rnn.py \
    --env_name Craftax-Symbolic-v1 \
    --target_achievement collect_wood --task_terminate_on_complete \
    --embedding_kind brzozowski_jax \
    --total_timesteps 200000 --num_envs 64 --no-use_wandb \
    --task_predicates_path           "$REPO/outputs/task_predicates.json" \
    --brzozowski_params_path         "$REPO/outputs/brzozowski_jax_params.msgpack" \
    --brzozowski_config_path         "$REPO/outputs/brzozowski_jax_config.yaml" \
    --brzozowski_eval_points_path    "$REPO/outputs/brzozowski_jax_eval_points.npy"

# 6) ppo_rnd.py with embedding.
python ppo_rnd.py \
    --env_name Craftax-Symbolic-v1 \
    --target_achievement collect_wood --task_terminate_on_complete \
    --embedding_kind brzozowski_jax \
    --total_timesteps 200000 --num_envs 64 --no-use_wandb \
    --task_predicates_path           "$REPO/outputs/task_predicates.json" \
    --brzozowski_params_path         "$REPO/outputs/brzozowski_jax_params.msgpack" \
    --brzozowski_config_path         "$REPO/outputs/brzozowski_jax_config.yaml" \
    --brzozowski_eval_points_path    "$REPO/outputs/brzozowski_jax_eval_points.npy"
```

The first JIT-compile (per Python process) takes ~60-90 seconds before
the first PPO update. Subsequent updates in the same process reuse the
in-memory JAX cache.

## Tests

Move-time pytest entry point:

```bash
cd "$REPO"
.venv/bin/python -m pytest third_party/craftax_baselines/tests/ -v
```

Three test files:
- `test_wrapper_smoke.py` -- `AutomatonAugmentedEnvWrapper` shape +
  predicate-encoding round-trip vs `_predicates_to_mask`.
- `test_accept_reward_shaping_wrapper.py` -- shaping fn dispatch,
  done-mask, scale plumbing.
- `test_embedding_setup.py` -- `build_embedding_stack` integration tests
  (skipped when artefacts aren't built).

## Where the wrappers go in the env stack

```
inner Craftax env  =  CraftaxSymbolicEnv  OR  CraftaxSymbolicTaskEnv (if --target_achievement)
   |
   v
AutomatonAugmentedEnvWrapper           # concat fingerprint(+accept) into obs;
   |                                   # emit info["accept"]
   v
[optional] AcceptRewardShapingWrapper  # carry prev_accept; modify reward
   |                                   # (sparse_accept | dense_accept_prob)
   v
LogWrapper                             # upstream (records episode return)
   |
   v
OptimisticResetVecEnvWrapper  OR  AutoResetEnvWrapper -> BatchEnvWrapper
   |
   v
ActorCritic / ActorCriticRNN / ActorCriticRND
```

The shaping wrapper sits BELOW `LogWrapper` so the recorded episode
returns reflect the shaped reward.

## Caveats

- **Auto-reset interaction.** With `--use_optimistic_resets` (default
  `True`), the inner env is the no-auto-reset variant and predicate
  firings on `done=True` steps are observed correctly by the AFA. With
  `--no-use_optimistic_resets` (plain `AutoReset + BatchEnv`), the inner
  env auto-resets at `step()` and clears achievements, so the AFA misses
  predicate firings that coincide with episode termination. The task
  wrapper's `info["task/reward"]` still fires on the rising edge in
  either case; only the AFA's accept transition is suppressed for
  terminal-step firings under auto-reset.
- **MONA dependency for RAD.** `scripts/build_rad_lookup.py` calls into
  `rad_comparison.embed.minimal_dfa_to_dfa` which shells out to the MONA
  binary. Install MONA system-wide (or skip RAD entirely; Brzozowski
  covers all 67 tasks).
- **Compile time.** Adding the embedding wrapper roughly halves
  throughput (the AFA forward pass runs once per env step). For the
  `set_transformer-d128` checkpoint with `NUM_ENVS=64`, expect
  ~50-100 SPS.
- **`info["accept"]` is not auto-forwarded to W&B.** Smokes use
  `--no-use_wandb`; if you want it logged for real runs, add a one-line
  forwarder in `logz/batch_logging.py` (mirror the `task/*` namespace
  pattern that's already there).