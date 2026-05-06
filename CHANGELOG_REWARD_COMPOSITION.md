# Reward composition refactor + `embedding/accept` logging fix

Session date: 2026-05-05 / 2026-05-06.

Two related changes shipped together: a logging bug fix in
`AutomatonAugmentedEnvWrapper`, and a unified reward composition
mechanism that replaces the prior `--reward_shaping` flag.

## 1. `pre_reset_accept` fix in `AutomatonAugmentedEnvWrapper`

### Problem

In W&B run `ktq4arxj` (and any embedding-conditioned run), the
`embedding/accept` curve was flat at `~0.0008426` and did not track
success rate.

### Root cause

`AutomatonAugmentedEnvWrapper.step` was overwriting the live `accept`
with the model's accept score for the *initial* automaton state on
`done=True`, then writing that overwritten value into
`info["embedding/accept"]`. Upstream's W&B aggregator averages info
rows weighted by `traj_batch.info["returned_episode"]`, which is
non-zero only on done steps, so every contributing value was the same
constant `init_accept` ~ 0.0008. Mean of a constant is the constant.

### Fix

In `automata_rl/wrappers.py:AutomatonAugmentedEnvWrapper.step`,
snapshot the live `accept` before the `jnp.where(done, init_accept,
accept)` overwrite, and use the snapshot for
`info["embedding/accept"]`. The post-overwrite `accept` continues to
feed the obs concat unchanged (the next obs needs the freshly-reset
automaton's accept value, not the dying trajectory's).

Also updated the comment in `AcceptRewardShapingWrapper.step` (later
renamed `RewardCompositionWrapper`) that justified its
`prev_accept` carry by referencing the now-fixed bug.

After this change, `embedding/accept` correctly reflects the model's
accept score for the trajectory just ended on each W&B point.

## 2. Unified reward composition

### Motivation

Previously the trainer's extrinsic reward came from two unrelated
mechanisms layered on each other: `task_env.py` overrode the env
reward to a sparse target-hit signal, and `AcceptRewardShapingWrapper`
optionally added an accept-derived bonus on top via
`--reward_shaping`. We also wanted a third mode: train PPO directly
from the embedding model's accept score.

Rather than adding a parallel `--reward_source` switch, the three are
unified into a single linear combination:

    reward_e = env_coef * env_reward + accept_coef * accept_term(done, prev_accept, curr_accept)

### CLI flags

Replaced the old pair (`--reward_shaping`, `--reward_shaping_scale`)
with three flags. Defaults adapt to `--embedding_kind` so users
typically only need to set one or two flags:

- `--accept_reward_kind` in `{none, sparse_accept,
  dense_accept_prob, continuous, continuous_relu}`. Default adapts:
  `none` when embedding is off, `continuous` otherwise.
- `--accept_reward_coef` (float). Default adapts: `0.0` when
  embedding is off, `1.0` otherwise.
- `--env_reward_coef` (float). Default adapts: `1.0` when embedding
  is off (today's behavior), `0.0` otherwise (pure accept-as-reward).

A new helper `apply_reward_config_defaults(config)` in `ppo.py` fills
in those adaptive defaults and asserts the strict pure-mode pairing:

- `--embedding_kind=none` requires `(env_coef=1.0, accept_coef=0.0,
  kind=none)`.
- `--embedding_kind != none` requires `(env_coef=0.0, accept_coef=1.0,
  kind != none)`.

Mixtures are intentionally unreachable. The asserts are temporary
scaffolding for the current experiments; the wrapper itself is fully
general (parameterized by both coefs and any kind), so removing the
asserts will re-enable mixtures with no other code change.

### Term-function refactor

`automata_rl/reward_shaping.py` was rewritten so each function
returns just the raw `accept_term` (no `+ reward`, no `& ~done`
mask). Existing two functions kept under their old names:

- `sparse_accept(done, prev_accept, curr_accept)` —
  `(curr > 0.5) & (prev <= 0.5)` as float32. Binary rising-edge
  signal.
- `dense_accept_prob(done, prev_accept, curr_accept)` —
  `curr_accept - prev_accept`. Telescoping delta.

Three new functions added:

- `continuous(...)` — `curr_accept`. Raw model belief.
- `continuous_relu(...)` — `curr_accept` if `>= 0.1` else `0`.
  Hard threshold to zero per-step floor reward.
- `none_term(...)` — always 0.

The previously-existing `& ~done` / `where(done, 0, ...)` masks were
introduced to compensate for the (now-fixed) accept-reset bug. With
the `pre_reset_accept` fix in place the masks were actively
incorrect — they were zeroing the very reward we wanted on
successful done steps. Removed.

`terminal_only` (`curr_accept if done else 0`) was considered and
dropped — for our typical setup (sticky achievement +
`--task_terminate_on_complete=True`) it's mathematically near-
equivalent to `continuous`. Documented in
`docs/accept_reward_types.md` for easy revival if a non-terminating-
episode workload surfaces.

### Wrapper rename

`AcceptRewardShapingWrapper` was replaced by
`RewardCompositionWrapper` (same `prev_accept` carry pattern, new
state class `_RewardCompositionState`). The wrapper:

- Reads `info["embedding/accept"]` (live pre-reset, post the
  `pre_reset_accept` fix).
- Computes `accept_term` via the configured term function.
- Emits `combined = env_coef * env_reward + accept_coef * accept_term`
  as the wrapper's reward.
- Always logs both raw components to info under
  `task/env_reward` and `task/accept_term`.
- Overwrites `info["task/reward"]` (originally set by `task_env` to
  the env hit signal) with the combined trainer reward, so W&B's
  `task/reward` series equals what PPO actually optimizes.

Inserted unconditionally inside `embedding_setup.build_embedding_stack`
(which itself only runs when `--embedding_kind != none`).

### W&B metrics under the new design

| Key | Meaning |
|---|---|
| `task/reward` | combined trainer reward (overwritten by `RewardCompositionWrapper`); equals `env_hit` when embedding off, equals `accept_term` when embedding on |
| `task/env_reward` | raw env reward, before `env_coef`. Only populated when wrapper is in chain |
| `task/accept_term` | raw `accept_term`, before `accept_coef`. Only populated when wrapper is in chain |
| `embedding/accept` | raw model accept score (post the `pre_reset_accept` fix) |
| `task/completed`, `Achievements/*`, `episode_length`, `episode_return` | unchanged |

## Files changed

- `automata_rl/reward_shaping.py` — full rewrite.
- `automata_rl/wrappers.py` — `pre_reset_accept` fix in
  `AutomatonAugmentedEnvWrapper.step`; replaced
  `AcceptRewardShapingWrapper` + `_ShapingState` +
  `_make_shaping_fn` with `RewardCompositionWrapper` +
  `_RewardCompositionState`; dropped now-unused `import functools`.
- `automata_rl/embedding_setup.py` — builds
  `RewardCompositionWrapper` unconditionally; preserved the
  `dense_accept_prob` + `rad_lookup` flicker error.
- `automata_rl/__init__.py` — module docstring updated.
- `ppo.py` — added `apply_reward_config_defaults(config)` helper;
  called from `make_train`; replaced
  `--reward_shaping` / `--reward_shaping_scale` with
  `--accept_reward_kind` / `--accept_reward_coef` /
  `--env_reward_coef` (all `default=None` sentinels with adaptive
  fill-in); added `import math`.
- `ppo_rnn.py`, `ppo_rnd.py` — same flag changes; both call
  `apply_reward_config_defaults`.
- `tests/test_reward_composition_wrapper.py` (new, replaces
  `test_accept_reward_shaping_wrapper.py`) — 17 unit tests covering
  term functions, linear-combination contract, info-key population,
  `pre_reset_accept` invariant, `prev_accept` carry, vmap/jit
  composition, constructor validation.
- `tests/test_wrapper_smoke.py` — replaced
  `test_reward_shaping_with_done_mask` with
  `test_term_fns_batched_no_done_mask` (asserts the unmasked
  behavior).
- `tests/test_embedding_setup.py` — added `sys.path` bootstrap;
  added 11 new tests for strict-pairing asserts on
  `apply_reward_config_defaults`; updated existing tests to new
  flag names; added wrapper-chain integration test.
- `docs/accept_reward_types.md` (new) — explains each shipped mode,
  calibration check via W&B, and why `terminal_only` was dropped.
- `third_party/craftax_baselines/README.md` — flag tables and prose
  updated.
- `TODO.md` — flag references updated.

## Verification

- 37 / 37 tests pass on a clean GPU (1 skipped on missing RAD
  artefact); the failure mode of GPU-OOM during contention with the
  user's running training was confirmed to be environmental, not a
  code regression.
- Standalone smoke test of `apply_reward_config_defaults` confirms
  adaptive defaults flip correctly per `--embedding_kind` and
  mixtures raise `AssertionError` as designed.
- `--help` on each of `ppo.py`, `ppo_rnn.py`, `ppo_rnd.py` shows the
  new flags; old flags are gone.
- `ruff check` introduces zero new errors over baseline. Delta of +4
  is `F821 Undefined name "B"` in jaxtyping shape strings of new
  term functions, matching the file's pre-existing convention.

## Caveats / next steps

- The strict pure-mode pairing asserts in
  `apply_reward_config_defaults` are temporary scaffolding; they are
  expected to be removed in the future to allow mixtures (e.g.
  `env_coef=1, accept_coef=1, kind=sparse_accept`). When that
  happens, tests 12-15 in
  `tests/test_embedding_setup.py` will need to be deleted or
  adapted.
- An end-to-end W&B smoke run with `--embedding_kind=brzozowski_jax`
  is still recommended to confirm the on-policy training-reward
  curves track expectations (`embedding/accept` rising toward 1,
  `task/accept_term` tracking `task/completed`). Not yet executed
  in this session.
- The sibling `Achievements/<name>` info-only zeroing bug in
  `task_env.py` (Craftax's `log_achievements_to_info` runs with the
  upstream-only `done`, before `task_env` ORs in
  `terminate_on_complete & now`) is documented but out of scope
  here.
- `ICM_REWARD_COEFF` calibration: with the default flag pairing
  (`env_coef=1, kind=none`) the reward magnitude is unchanged, so
  ICM stays calibrated. With `--embedding_kind=none` the wrapper
  isn't built at all. The caveat applies only to future mixture
  experiments where reward-magnitude shifts may require retuning.

## References

- Plan file: `/home/dweept/.claude/plans/pure-giggling-bentley.md`
  (full design rationale).
- Per-mode docs: `docs/accept_reward_types.md`.
- W&B run referenced in the bug report: `ktq4arxj`.

## Postscript (Hydra config refactor)

The strict-pairing asserts described above have since been removed as
part of the argparse-to-Hydra migration. ``apply_reward_config_defaults``
no longer exists; its job is now done by config-group composition under
``conf/``:

- Each ``reward/*.yaml`` is internally consistent (env-only or
  accept-only triple).
- Each embedding YAML sets the per-embedding default reward bundle via
  ``defaults: - override /reward: ...``, replacing the adaptive defaults
  that ``apply_reward_config_defaults`` used to fill in.
- Mixtures are now first-class — ``experiment=brzozowski_env_only``
  composes embedding=brzozowski with reward=env_only; users can also
  hand-mix via ``embedding=brzozowski reward=accept_sparse env_reward_coef=1.0``.
- Tests 12-15 (``test_strict_pairing_*``, ``test_strict_assert_fails_*``)
  in ``tests/test_embedding_setup.py`` were deleted; ``test_nan_coef_raises``
  / ``test_inf_coef_raises`` moved into
  ``tests/test_hydra_config.py::test_nonfinite_coef_rejected``.
- The finite-coef and E3B-needs-ICM checks moved verbatim into
  ``train.py:_validate_cross_cutting``.

See plan file for the full preservation matrix.
