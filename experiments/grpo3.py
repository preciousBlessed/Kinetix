import os
import time
from typing import Any, NamedTuple, Tuple

import hydra
import jax
import jax.numpy as jnp
from kinetix.environment import make_reset_fn_from_config
import numpy as np
import optax
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

import wandb
from kinetix.environment import LogWrapper, PixelObservations
from kinetix.environment.env import make_kinetix_env
from kinetix.models import ScannedRNN, make_network_from_config
from kinetix.render.renderer_pixels import make_render_pixels
from kinetix.util import (
    general_eval,
    generate_params_from_config,
    load_evaluation_levels,
    get_video_frequency,
    init_wandb,
    load_train_state_from_wandb_artifact_path,
    normalise_config,
    save_model,
)

os.environ["WANDB_DISABLE_SERVICE"] = "True"


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    ref_log_prob: jnp.ndarray  # For KL penalty
    obs: Any
    info: jnp.ndarray


def make_train(config, env_params, static_env_params):
    config["num_updates"] = config["total_timesteps"] // config["num_steps"] // config["num_train_envs"]
    config["minibatch_size"] = config["num_train_envs"] * config["num_steps"] // config["num_minibatches"]

    def make_env(reset_fn, static_env_params):
        return LogWrapper(
            make_kinetix_env(config["action_type"], config["observation_type"], reset_fn, env_params, static_env_params)
        )

    eval_levels, eval_static_env_params = load_evaluation_levels(config["eval_levels"])
    env = make_env(make_reset_fn_from_config(config, env_params, static_env_params), static_env_params)
    eval_env = make_env(None, eval_static_env_params)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["num_minibatches"] * config["update_epochs"])) / config["num_updates"]
        return config["lr"] * frac

    time_start = time.time()

    def train(rng):
        last_time = time.time()
        # INIT NETWORK (No value head)
        network = make_network_from_config(env, env_params, config, value_head=False)
        rng, _rng = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset, (0, None))(jax.random.split(_rng, config["num_train_envs"]), env_params)
        dones = jnp.zeros((config["num_train_envs"]), dtype=jnp.bool_)
        rng, _rng = jax.random.split(rng)
        hstate = ScannedRNN.initialize_carry(config["num_train_envs"])
        init_x = jax.tree.map(lambda x: x[None, ...], (obsv, dones))
        network_params = network.init(_rng, hstate, init_x)

        # REFERENCE NETWORK FOR KL PENALTY
        reference_network = make_network_from_config(env, env_params, config, value_head=False)
        reference_params = network_params  # Initialize same as current policy

        if config["anneal_lr"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(config["lr"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset, (0, None))(jax.random.split(_rng, config["num_train_envs"]), env_params)
        hstate = ScannedRNN.initialize_carry(config["num_train_envs"])
        render_static_env_params = eval_static_env_params.replace(downscale=4)
        pixel_renderer = jax.jit(make_render_pixels(env_params, render_static_env_params))
        pixel_render_fn = lambda x: pixel_renderer(x) / 255.0

        # EVAL UTILITIES
        def _vmapped_eval_step(runner_state, rng):
            def _single_eval_step(rng):
                return general_eval(
                    rng,
                    eval_env,
                    env_params,
                    runner_state[0],
                    eval_levels,
                    env_params.max_timesteps,
                    config["num_eval_levels"],
                    keep_states=True,
                    return_trajectories=True,
                )

            (states, returns, done_idxs, episode_lengths, eval_infos), (eval_dones, eval_rewards) = jax.vmap(
                _single_eval_step
            )(jax.random.split(rng, config["eval_num_attempts"]))
            mask = jnp.arange(env_params.max_timesteps)[None, ..., None] < episode_lengths[:, None, :]
            eval_solves = (eval_infos["returned_episode_solved"] * eval_dones * mask).sum(axis=1) / jnp.maximum(
                1, (eval_dones * mask).sum(axis=1)
            )
            states_to_plot = jax.tree.map(lambda x: x[0], states)

            return (
                states_to_plot,
                done_idxs[0],
                returns[0],
                returns.mean(axis=0),
                episode_lengths.mean(axis=0),
                eval_solves.mean(axis=0),
            )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng, update_step = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                ac_in = (jax.tree.map(lambda x: x[np.newaxis, :], last_obs), last_done[np.newaxis, :])
                hstate, pi, _ = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # REFERENCE POLICY LOG PROB
                _, ref_pi, _ = reference_network.apply(reference_params, hstate, ac_in)
                ref_log_prob = ref_pi.log_prob(action)

                action, log_prob, ref_log_prob = (
                    action.squeeze(0),
                    log_prob.squeeze(0),
                    ref_log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    jax.random.split(_rng, config["num_train_envs"]), env_state, action, env_params
                )
                transition = Transition(done, action, reward, log_prob, ref_log_prob, last_obs, info)
                runner_state = (train_state, env_state, obsv, done, hstate, rng, update_step)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["num_steps"])

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step,
            ) = runner_state

            # GRPO LOSS COMPUTATION
            def _update_minbatch(train_state, batch_info):
                init_hstate, traj_batch, rewards = batch_info

                def _loss_fn(params, init_hstate, traj_batch, rewards):
                    # RERUN NETWORK
                    _, pi, _ = network.apply(params, init_hstate[0], (traj_batch.obs, traj_batch.done))
                    log_prob = pi.log_prob(traj_batch.action)
                    entropy = pi.entropy().mean()

                    # GROUP-RELATIVE NORMALIZATION
                    group_rewards = rewards.reshape(-1, config["group_size"])
                    normalized_rewards = (group_rewards - group_rewards.mean(axis=1, keepdims=True)) / (
                        group_rewards.std(axis=1, keepdims=True) + 1e-8
                    )
                    advantages = normalized_rewards.reshape(-1)

                    # CLIPPED SURROGATE LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    loss_actor1 = ratio * advantages
                    loss_actor2 = jnp.clip(ratio, 1.0 - config["clip_eps"], 1.0 + config["clip_eps"]) * advantages
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                    # KL PENALTY
                    kl_penalty = (
                        jnp.exp(traj_batch.ref_log_prob - log_prob) - (traj_batch.ref_log_prob - log_prob) - 1
                    ).mean()

                    total_loss = loss_actor - config["ent_coef"] * entropy + config["group_beta"] * kl_penalty
                    return total_loss, (loss_actor, entropy, kl_penalty)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch, rewards)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            # GROUP REWARDS FOR GRPO
            rewards = traj_batch.reward
            init_hstate = runner_state[-3][None, :]  # Get hstate from runner_state
            update_state = (train_state, init_hstate, traj_batch, rewards)
            update_state, loss_info = jax.lax.scan(_update_minbatch, update_state, None, config["update_epochs"])
            train_state = update_state[0]

            # METRICS COMPUTATION
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum() / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )

            # COMPREHENSIVE MONITORING
            if config["use_wandb"]:

                def _fake_video():
                    return jnp.zeros(
                        (
                            env_params.max_timesteps,
                            config["num_eval_levels"],
                            *PixelObservations(env_params, render_static_env_params)
                            .observation_space(env_params)
                            .shape,
                        )
                    )

                def _real_eval(rng, update_step):
                    vid_frequency = get_video_frequency(config, update_step)
                    rng, _rng = jax.random.split(rng)
                    to_log_videos = _vmapped_eval_step(runner_state, _rng)
                    should_log_videos = update_step % vid_frequency == 0
                    first = jax.lax.cond(
                        should_log_videos,
                        lambda: jax.vmap(jax.vmap(pixel_render_fn))(to_log_videos[0].env_state),
                        lambda: _fake_video(),
                    )
                    return (first, should_log_videos, True, *to_log_videos[1:])

                def _fake_eval(rng, update_step):
                    return (
                        _fake_video(),
                        False,
                        False,
                        jnp.zeros((config["num_eval_levels"],), jnp.int32),
                        jnp.zeros((config["num_eval_levels"],), jnp.float32),
                        jnp.zeros((config["num_eval_levels"],), jnp.float32),
                        jnp.zeros((config["num_eval_levels"],), jnp.float32),
                        jnp.zeros((config["num_eval_levels"],), jnp.float32),
                    )

                def callback(metric, raw_info, loss_info, update_step, to_log_videos):
                    nonlocal last_time
                    time_now = time.time()
                    delta_time = time_now - last_time
                    last_time = time_now

                    dones = raw_info["returned_episode"]
                    to_log = {
                        # Episode stats
                        "episode/return": (raw_info["returned_episode_returns"] * dones).sum()
                        / jnp.maximum(1, dones.sum()),
                        "episode/solved": (raw_info["returned_episode_solved"] * dones).sum()
                        / jnp.maximum(1, dones.sum()),
                        "episode/length": (raw_info["returned_episode_lengths"] * dones).sum()
                        / jnp.maximum(1, dones.sum()),
                        "episode/completed": dones.sum(),
                        # Loss components
                        "loss/total": loss_info[0].mean(),
                        "loss/actor": loss_info[1][0].mean(),
                        "loss/entropy": loss_info[1][1].mean(),
                        "loss/kl": config["kl_coef"] * loss_info[1][2].mean(),
                        # Training dynamics
                        "policy/ratio": jnp.exp(metric["log_prob"] - metric["old_log_prob"]).mean(),
                        "policy/kl": (metric["log_prob"] - metric["old_log_prob"]).mean(),
                        # Timing
                        "timing/updates": update_step,
                        "timing/env_steps": update_step * config["num_steps"] * config["num_train_envs"],
                        "timing/sps": (config["num_steps"] * config["num_train_envs"]) / delta_time,
                        "timing/sps_agg": (update_step * config["num_steps"] * config["num_train_envs"])
                        / (time_now - time_start),
                    }

                    # Evaluation metrics
                    (
                        obs_vid,
                        should_log_videos,
                        should_log_eval,
                        idx_vid,
                        eval_return_vid,
                        eval_return_mean,
                        eval_eplen_mean,
                        eval_solverate_mean,
                    ) = to_log_videos

                    if should_log_eval:
                        to_log.update(
                            {
                                "eval/mean_return": eval_return_mean.mean(),
                                "eval/mean_length": eval_eplen_mean.mean(),
                                "eval/mean_solve": eval_solverate_mean.mean(),
                            }
                        )
                        for i, name in enumerate(config["eval_levels"]):
                            to_log.update(
                                {
                                    f"eval/{name}_return": eval_return_mean[i],
                                    f"eval/{name}_solve": eval_solverate_mean[i],
                                }
                            )

                    if should_log_videos:
                        for i, name in enumerate(config["eval_levels"]):
                            frames = obs_vid[: idx_vid[i], i]
                            frames = np.asarray(frames).transpose(0, 3, 2, 1)[:, :, ::-1, :]
                            to_log[f"media/{name}"] = wandb.Video((frames * 255).astype(np.uint8), fps=15)

                    wandb.log(to_log)

                rng, _rng = jax.random.split(rng)
                to_log_videos = jax.lax.cond(
                    update_step % config["eval_freq"] == 0, _real_eval, _fake_eval, _rng, update_step
                )
                jax.debug.callback(callback, metric, traj_batch.info, loss_info, update_step, to_log_videos)

            # UPDATE RUNNER STATE
            runner_state = (train_state, env_state, obsv, dones, hstate, rng, update_step + 1)
            return runner_state, metric

        # INIT RUNNER STATE
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["num_train_envs"]), dtype=bool),
            hstate,
            rng,
            0,
        )
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["num_updates"])
        return {"runner_state": runner_state, "metric": metric}

    return train


@hydra.main(version_base=None, config_path="../configs", config_name="grpo")
def main(config):
    config = normalise_config(OmegaConf.to_container(config), "GRPO")
    env_params, static_env_params = generate_params_from_config(config)
    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)

    if config["use_wandb"]:
        run = init_wandb(config, "GRPO")

    rng = jax.random.PRNGKey(config["seed"])
    rng, _rng = jax.random.split(rng)
    train_jit = jax.jit(make_train(config, env_params, static_env_params))

    out = train_jit(_rng)

    if config["save_policy"]:
        train_state = out["runner_state"][0]
        save_model(train_state, config["total_timesteps"], config, save_to_wandb=config["use_wandb"])


if __name__ == "__main__":
    main()
