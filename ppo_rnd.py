import os
import time

import jax
import jax.numpy as jnp
import optax
from craftax.craftax_env import make_craftax_env_from_name
from task_env import CraftaxSymbolicTaskEnv, CraftaxSymbolicTaskEnvNoAutoReset

import wandb
from typing import NamedTuple

from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from logz.batch_logging import batch_log, create_log_dict
from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    AutoResetEnvWrapper,
    BatchEnvWrapper,
)
from models.rnd import RNDNetwork, ActorCriticRND

# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value_e: jnp.ndarray
    value_i: jnp.ndarray
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    if config.get("TARGET_ACHIEVEMENT"):
        if "Symbolic" not in config["ENV_NAME"]:
            raise ValueError(
                "--target_achievement is only supported for Symbolic envs."
            )
        cls = (CraftaxSymbolicTaskEnv if not config["USE_OPTIMISTIC_RESETS"]
               else CraftaxSymbolicTaskEnvNoAutoReset)
        env = cls(
            target_achievement=config["TARGET_ACHIEVEMENT"],
            terminate_on_complete=config.get("TASK_TERMINATE_ON_COMPLETE", False),
        )
    else:
        env = make_craftax_env_from_name(
            config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
        )
    env_params = env.default_params

    # Optionally augment the env with an automaton fingerprint. The
    # augmented obs (UVFA-style flat concat) flows through ``ActorCriticRND``
    # unchanged. The RND target/predictor networks operate on the same
    # augmented obs through their own internal Dense stacks.
    if config.get("EMBEDDING_KIND", "none") != "none":
        if not config.get("TARGET_ACHIEVEMENT"):
            raise ValueError(
                "--embedding_kind requires --target_achievement; default Craftax "
                "sum-of-achievements reward is not a single LTLf task and is unsupported.",
            )
        from automata_rl.embedding_setup import build_embedding_stack

        env, _ = build_embedding_stack(env, config)

    env = LogWrapper(env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        if "Symbolic" in config["ENV_NAME"]:
            network = ActorCriticRND(
                env.action_space(env_params).n, config["LAYER_SIZE"]
            )
        else:
            raise ValueError
            # network = ActorCriticConv(
            #     env.action_space(env_params).n, config["LAYER_SIZE"]
            # )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # Exploration state
        ex_state = {
            "rnd_model": None,
        }

        if config["USE_RND"]:
            obs_shape = env.observation_space(env_params).shape
            assert len(obs_shape) == 1, "Only configured for 1D observations"
            obs_shape = obs_shape[0]

            # Random network
            rnd_random_network = RNDNetwork(
                num_layers=3,
                output_dim=config["RND_OUTPUT_SIZE"],
                layer_size=config["RND_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            rnd_random_network_params = rnd_random_network.init(
                _rng, jnp.zeros((1, obs_shape))
            )

            # Distillation Network
            rnd_distillation_network = RNDNetwork(
                num_layers=3,
                output_dim=config["RND_OUTPUT_SIZE"],
                layer_size=config["RND_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            rnd_distillation_network_params = rnd_distillation_network.init(
                _rng, jnp.zeros((1, obs_shape))
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["RND_LR"], eps=1e-5),
            )
            ex_state["rnd_distillation_network"] = TrainState.create(
                apply_fn=rnd_distillation_network.apply,
                params=rnd_distillation_network_params,
                tx=tx,
            )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    ex_state,
                    rng,
                    update_step,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value_e, value_i = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward_e, done, info = env.step(
                    _rng, env_state, action, env_params
                )

                reward_i = jnp.zeros(config["NUM_ENVS"])

                if config["USE_RND"]:
                    random_pred = rnd_random_network.apply(
                        rnd_random_network_params, obsv
                    )

                    distill_pred = ex_state["rnd_distillation_network"].apply_fn(
                        ex_state["rnd_distillation_network"].params, obsv
                    )
                    error = (random_pred - distill_pred) * (1 - done[:, None])
                    mse = jnp.square(error).mean(axis=-1)

                    reward_i = mse * config["RND_REWARD_COEFF"]

                reward = reward_e + reward_i

                transition = Transition(
                    done=done,
                    action=action,
                    value_e=value_e,
                    value_i=value_i,
                    reward=reward,
                    reward_i=reward_i,
                    reward_e=reward_e,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=obsv,
                    info=info,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    ex_state,
                    rng,
                    update_step,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step,
            ) = runner_state
            _, last_val_e, last_val_i = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val, is_extrinsic):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value, is_extrinsic = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        jax.lax.select(
                            is_extrinsic, transition.value_e, transition.value_i
                        ),
                        jax.lax.select(
                            is_extrinsic, transition.reward_e, transition.reward_i
                        ),
                    )
                    done = jnp.logical_and(
                        done, jnp.logical_or(config["RND_IS_EPISODIC"], is_extrinsic)
                    )

                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value, is_extrinsic), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, is_extrinsic),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + jax.lax.select(
                    is_extrinsic, traj_batch.value_e, traj_batch.value_i
                )

            advantages_e, targets_e = _calculate_gae(traj_batch, last_val_e, True)
            advantages_i, targets_i = _calculate_gae(traj_batch, last_val_i, False)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    (
                        traj_batch,
                        advantages_e,
                        targets_e,
                        advantages_i,
                        targets_i,
                    ) = batch_info

                    # Policy/value network
                    def _loss_fn(
                        params, traj_batch, gae_e, targets_e, gae_i, targets_i
                    ):
                        # RERUN NETWORK
                        pi, value_e, value_i = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE EXTRINSIC VALUE LOSS
                        value_pred_clipped_e = traj_batch.value_e + (
                            value_e - traj_batch.value_e
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_e = jnp.square(value_e - targets_e)
                        value_losses_clipped_e = jnp.square(
                            value_pred_clipped_e - targets_e
                        )
                        value_loss_e = (
                            0.5
                            * jnp.maximum(value_losses_e, value_losses_clipped_e).mean()
                        )

                        # CALCULATE INTRINSIC VALUE LOSS
                        value_pred_clipped_i = traj_batch.value_i + (
                            value_i - traj_batch.value_i
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses_i = jnp.square(value_i - targets_i)
                        value_losses_clipped_i = jnp.square(
                            value_pred_clipped_i - targets_i
                        )
                        value_loss_i = (
                            0.5
                            * jnp.maximum(value_losses_i, value_losses_clipped_i).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        gae = gae_e
                        if config["USE_RND"]:
                            gae += gae_i * config["RND_GAE_COEFF"]
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        value_loss = value_loss_e
                        if config["USE_RND"]:
                            value_loss += value_loss_i

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (
                            value_loss_e,
                            value_loss_i,
                            loss_actor,
                            entropy,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params,
                        traj_batch,
                        advantages_e,
                        targets_e,
                        advantages_i,
                        targets_i,
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    losses = (total_loss, 0)
                    return train_state, losses

                (
                    train_state,
                    traj_batch,
                    advantages_e,
                    targets_e,
                    advantages_i,
                    targets_i,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (
                    traj_batch,
                    advantages_e,
                    targets_e,
                    advantages_i,
                    targets_i,
                )
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, losses = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    traj_batch,
                    advantages_e,
                    targets_e,
                    advantages_i,
                    targets_i,
                    rng,
                )
                return update_state, losses

            update_state = (
                train_state,
                traj_batch,
                advantages_e,
                targets_e,
                advantages_i,
                targets_i,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )

            rng = update_state[-1]

            # UPDATE EXPLORATION STATE
            def _update_ex_epoch(update_state, unused):
                def _update_ex_minbatch(ex_state, traj_batch):
                    rnd_loss = 0

                    if config["USE_RND"]:

                        def _rnd_loss_fn(rnd_distillation_params, traj_batch):
                            random_network_out = rnd_random_network.apply(
                                rnd_random_network_params, traj_batch.next_obs
                            )

                            distillation_network_out = ex_state[
                                "rnd_distillation_network"
                            ].apply_fn(rnd_distillation_params, traj_batch.next_obs)

                            error = (random_network_out - distillation_network_out) * (
                                1 - traj_batch.done[:, None]
                            )
                            return jnp.square(error).mean() * config["RND_LOSS_COEFF"]

                        rnd_grad_fn = jax.value_and_grad(_rnd_loss_fn, has_aux=False)
                        rnd_loss, rnd_grad = rnd_grad_fn(
                            ex_state["rnd_distillation_network"].params, traj_batch
                        )
                        ex_state["rnd_distillation_network"] = ex_state[
                            "rnd_distillation_network"
                        ].apply_gradients(grads=rnd_grad)

                    losses = (rnd_loss,)
                    return ex_state, losses

                (ex_state, traj_batch, rng) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                ex_state, losses = jax.lax.scan(
                    _update_ex_minbatch, ex_state, minibatches
                )
                update_state = (ex_state, traj_batch, rng)
                return update_state, losses

            if config["USE_RND"]:
                ex_update_state = (ex_state, traj_batch, rng)
                ex_update_state, ex_loss = jax.lax.scan(
                    _update_ex_epoch,
                    ex_update_state,
                    None,
                    config["EXPLORATION_UPDATE_EPOCHS"],
                )
                metric["rnd_loss"] = ex_loss[0].mean()
                metric["reward_i"] = traj_batch.reward_i.mean()
                metric["reward_e"] = traj_batch.reward_e.mean()

                ex_state = ex_update_state[0]
                rng = ex_update_state[-1]

            # wandb logging
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(
                    metric, update_step
                ):  # , loss_info, traj_batch, ex_state, advantages_i, targets_i):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)

                jax.debug.callback(
                    callback,
                    metric,
                    update_step,
                    # loss_info, traj_batch, ex_state, advantages_i, targets_i
                )

            runner_state = (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            ex_state,
            _rng,
            0,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}  # , "info": metric}

    return train


def run_ppo(config):
    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["WANDB_RUN_NAME"],
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_jit = jax.jit(make_train(config))
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_vmap(rngs)
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))
    # t1 = time.time()
    # out = train_vmap(rngs)
    # t2 = time.time()
    # print("t2", t2 - t1)
    # print("SPS2: ", config["TOTAL_TIMESTEPS"] / (t2 - t1))

    if config["USE_WANDB"]:
        # if config["DEBUG"] == "end":
        #     info = out["info"]
        #     for update in range(info["timestep"].shape[1]):
        #         if update % 10 == 0:
        #             for repeat in range(info["timestep"].shape[0]):
        #                 update_info = jax.tree.map(lambda x: x[repeat, update], info)
        #                 to_log = create_log_dict(update_info)
        #                 batch_log(update, to_log, config)
        #
        #     t2 = time.time()
        #     print("Time to log to wandb", t2 - t1)

        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree.map(lambda x: x[0], train_states)
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            path = os.path.join(wandb.run.dir, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            print(f"saved runner state to {path}")
            save_args = orbax_utils.save_args_from_target(train_state)
            checkpoint_manager.save(
                config["TOTAL_TIMESTEPS"],
                train_state,
                save_kwargs={"save_args": save_args},
            )

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")

