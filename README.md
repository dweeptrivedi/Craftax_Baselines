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

All three PPO variants (vanilla, RNN, RND) are launched through a single
Hydra entry point ``train.py`` with a ``variant=`` config group:

### PPO
```commandline
python train.py variant=ppo
```

### PPO-RNN
```commandline
python train.py variant=ppo_rnn
```

### ICM
```commandline
python train.py variant=ppo train_icm=true
```

### E3B
```commandline
python train.py variant=ppo train_icm=true use_e3b=true icm_reward_coeff=0
```

### RND
```commandline
python train.py variant=ppo_rnd
```

# Visualisation
You can save trained policies with ``save_policy=true`` (default ``true``). These can then be viewed with the `view_ppo_agent` script (pass in the path up to the `files` directory).

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

| ``embedding=`` group | What it does | Coverage |
|---|---|---|
| `none` (default) | No embedding; standard PPO. | n/a |
| `brzozowski` | Live JAX inference of the trained `AFAEmbedding` model on every env step. Computes a `(M=512,)` fingerprint plus a scalar accept value. (Internally sets ``embedding_kind=brzozowski_jax``.) | All 67 standard Craftax tasks. |
| `rad` | Precomputed `(num_states, embed_dim)` lookup table indexed by a discrete DFA state. (Internally sets ``embedding_kind=rad_lookup``.) | Subset of tasks (the per-task `.npz` file must exist; lookup-table build needs the MONA binary). |

The conditioning architecture is always flat-concat: the augmented obs
flows through the upstream policy network unchanged (`ActorCritic` for
`ppo.py`, `ActorCriticRNN` for `ppo_rnn.py`, `ActorCriticRND` for
`ppo_rnd.py`).

## CLI / config

Hydra config tree under ``conf/``:

| Group | Choices | Notes |
|---|---|---|
| ``variant`` | ``ppo``, ``ppo_rnn``, ``ppo_rnd`` | Variant-specific defaults (ICM/E3B for ``ppo``, RND for ``ppo_rnd``). Selects the ``run_target`` dispatched to. |
| ``embedding`` | ``none`` (default), ``brzozowski``, ``rad`` | Each option also flips the default ``reward`` group via ``defaults: - override /reward: ...``. |
| ``reward`` | ``env_only``, ``accept_continuous``, ``accept_continuous_relu``, ``accept_sparse``, ``accept_dense`` | Each YAML pins a triple ``(env_reward_coef, accept_reward_coef, accept_reward_kind)``. |
| ``experiment`` | ``none_env_only``, ``brzozowski_continuous``, ``brzozowski_continuous_relu``, ``brzozowski_env_only``, ``rad_continuous``, ``rad_continuous_relu``, ``rad_env_only`` | Composite preset that overrides both ``embedding`` and ``reward`` for sweeps. |
| ``env``, ``ppo_hyperparameters``, ``network``, ``logging`` | one option each | Stock Craftax-Symbolic settings, PPO scalars, network width/activation, wandb knobs. |

Top-level scalars (set on the root config or via the CLI):
``seed`` (null = randomized in ``train.py``), ``debug``, ``jit``,
``num_repeats``, ``target_achievement``, ``task_terminate_on_complete``,
``task_predicates_path`` (default ``outputs/task_predicates.json``).

Override any leaf via Hydra CLI:

```bash
python train.py num_envs=256 seed=0 target_achievement=place_table
python train.py embedding=brzozowski reward=accept_sparse
python train.py experiment=brzozowski_continuous variant=ppo_rnd
```

Sweep over the canonical 21-cell experiment matrix:

```bash
python train.py -m experiment=glob\(*\) variant=ppo,ppo_rnn,ppo_rnd
```

## Hard constraints (errors at make_train time)

- **``embedding != none`` requires ``target_achievement=<name>``.**
  The default Craftax sum-of-achievements reward is not a single LTLf
  task, and feeding it through a per-task AFA fingerprint is semantically
  meaningless. `make_train` raises ``ValueError`` and aborts.
- **``reward=accept_dense`` is incompatible with ``embedding=rad``.**
  RAD's accept channel is binary 0/1, so the delta would flicker. Use
  ``accept_sparse``, ``accept_continuous``, or ``accept_continuous_relu``
  with RAD.
- **``use_e3b=true`` requires ``train_icm=true`` and ``icm_reward_coeff=0``.**
  E3B replaces the ICM bonus, so the underlying ICM features must be
  trained but their bonus must be zeroed. Asserted in ``train.py``.
- **Variant-flag isolation** is enforced by Hydra's strict mode: ICM/E3B
  fields (``train_icm``, ``use_e3b``, etc.) are only declared by
  ``variant/ppo.yaml``; RND fields (``use_rnd``, ``rnd_*``) are only in
  ``variant/ppo_rnd.yaml``. Setting them under the wrong variant raises
  ``ConfigCompositionException``, matching today's argparse behavior of
  rejecting flags that the variant's argparse never registered.

## One-time artefact build

The artefact-builder scripts live in this submodule's `scripts/`. They
write into the parent repo's `outputs/` directory by default (resolved
relative to each script's own `__file__`), so you can run them from
anywhere.

```bash
# 1. Per-task predicate orderings + task_idx (committed, ~10 s)
python scripts/derive_task_predicates.py

# 2. Brzozowski-JAX backend (committed under ``weights/brzozowski/<variant>/``;
#    only re-run to refresh from a different checkpoint).
#    Variants: ``deepsets`` (default for ``embedding=brzozowski``) and
#    ``set_transformer`` (selected via ``embedding=brzozowski_st``).
#    The converter lives in the PARENT repo because it's about converting
#    a PyTorch Lightning checkpoint, not RL.
.venv/bin/python ../../scripts/convert_brzozowski_to_jax.py \
    --checkpoint /path/to/last.ckpt \
    --hdf5 /path/to/dataset_v2.h5 \
    --params-output      weights/brzozowski/deepsets/params.msgpack \
    --eval-points-output weights/brzozowski/deepsets/eval_points.npy \
    --config-output      weights/brzozowski/deepsets/config.yaml \
    --device cpu

# 3. RAD lookup table (gitignored, requires the MONA binary system-wide)
uv sync --group rad
python scripts/build_rad_lookup.py --rad-dir /path/to/rad_per_task_npz_dir
```

If MONA is unavailable, skip step 3 and use `--embedding_kind brzozowski_jax`
only. The 67 standard Craftax tasks are all covered by the Brzozowski path.

## Example experiment commands

Run from `third_party/craftax_baselines/` so `task_env.py`,
`wrappers.py`, `logz/`, `models/`, `automata_rl/` resolve as siblings.
Settings below match ``scripts/run_sweep.sh`` defaults (2e8 steps,
1024 envs, W&B on). Substitute ``<task>`` and ``<seed>`` for your
values; pick ``<task>`` from any entry in
``outputs/task_predicates.json``.

```bash
cd /path/to/automata-embeddings/third_party/craftax_baselines
```

Vanilla PPO with brzozowski embedding:

```bash
python train.py variant=ppo \
    embedding=brzozowski \
    target_achievement=<task> task_terminate_on_complete=true \
    total_timesteps=200000000 num_envs=1024 seed=<seed> \
    wandb_project=craftx-baselines
```

(``embedding=brzozowski`` automatically flips the reward bundle to
``accept_continuous`` via the embedding YAML's ``override /reward``.)

PPO + RND:

```bash
python train.py variant=ppo_rnd \
    embedding=brzozowski \
    target_achievement=<task> task_terminate_on_complete=true \
    total_timesteps=200000000 num_envs=1024 seed=<seed>
```

PPO + RNN:

```bash
python train.py variant=ppo_rnn \
    embedding=brzozowski \
    target_achievement=<task> task_terminate_on_complete=true \
    total_timesteps=200000000 num_envs=1024 seed=<seed>
```

**Variants:**

* **No-embedding baseline.** Drop ``embedding=brzozowski`` from any of
  the commands above (or set ``embedding=none`` explicitly).
* **RAD lookup** (only if ``outputs/lookup_rad.npz`` was built with
  MONA). Replace ``embedding=brzozowski`` with ``embedding=rad``.
* **Picking an accept-reward kind.** With the embedding active, the
  default kind is ``continuous``. Override with
  ``reward=accept_sparse``, ``reward=accept_continuous_relu``,
  or ``reward=accept_dense`` (Brzozowski-only). See
  ``docs/accept_reward_types.md``. Mixtures (e.g. ``env_reward_coef=1
  reward=accept_sparse``) are now reachable -- the prior strict-pairing
  asserts have been removed.
* **Pre-defined experiment.** Use ``experiment=...`` to compose an
  embedding+reward bundle in one flag, e.g.
  ``experiment=brzozowski_continuous_relu`` (auto-sets
  ``embedding=brzozowski reward=accept_continuous_relu``).

Canonical 21-cell sweep:

```bash
python train.py -m experiment=glob\(*\) variant=ppo,ppo_rnn,ppo_rnd \
    target_achievement=<task> task_terminate_on_complete=true \
    total_timesteps=200000000 num_envs=1024 seed=<seed>
```

For a quick local smoke (~95 s on a single A6000), swap
``total_timesteps=5000``, ``num_envs=16``, and add ``use_wandb=false``.
The first JIT-compile takes 60-90 s; subsequent updates in the same
process reuse the in-memory JAX cache. Expect ~50-100 SPS during real
training (the AFA forward pass is the bottleneck).

## Tests

Move-time pytest entry point:

```bash
cd "$REPO"
.venv/bin/python -m pytest third_party/craftax_baselines/tests/ -v
```

Four test files:
- ``test_wrapper_smoke.py`` -- `AutomatonAugmentedEnvWrapper` shape +
  predicate-encoding round-trip vs `_predicates_to_mask`, plus
  batched term-fn smoke checks.
- ``test_reward_composition_wrapper.py`` -- `RewardCompositionWrapper`
  unit tests: term functions, linear-combination contract, info-key
  population, `prev_accept` carry, vmap/jit composition.
- ``test_embedding_setup.py`` -- `build_embedding_stack` integration tests
  (artefact-dependent tests skipped when artefacts aren't built).
- ``test_hydra_config.py`` -- composition tests for the ``conf/`` tree:
  default + group composition, the seven-experiment matrix, sweep
  cardinality, train.py glue (uppercasing, total-timesteps cast, seed
  fill, wandb run name, cross-cutting validation), and Hydra
  strict-mode rejection of unknown keys / variant-flag isolation.

## Where the wrappers go in the env stack

```
inner Craftax env  =  CraftaxSymbolicEnv  OR  CraftaxSymbolicTaskEnv (if --target_achievement)
   |
   v
AutomatonAugmentedEnvWrapper           # concat fingerprint(+accept) into obs;
   |                                   # emit info["embedding/accept"]
   v
RewardCompositionWrapper               # carry prev_accept; emit
   |                                   # reward = env_coef*env + accept_coef*accept_term
   v                                   # (kind selects the term function)
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
- **W&B logging of the AFA acceptance signal.**
  `AutomatonAugmentedEnvWrapper` emits the per-step accept value as
  `info["embedding/accept"]`, and `logz/batch_logging.create_log_dict`
  forwards every key under the `embedding/` namespace to W&B (mirrors
  the existing `task/*` pattern). Charts appear as `embedding/accept`
  in the W&B UI when `--use_wandb` is on.