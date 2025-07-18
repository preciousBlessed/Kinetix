"GRPO With KL Divergence Penalty--Beta"

import os
import time
from typing import Any, NamedTuple

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
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
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

    def linear_warmup_cosine_decay_schedule(count):
        frac = (count // (config["num_minibatches"] * config["update_epochs"])) / config[
            "num_updates"
        ]  # between 0 and 1
        delta = config["peak_lr"] - config["initial_lr"]
        frac_diff_max = 1.0 - config["warmup_frac"]
        frac_cosine = (frac - config["warmup_frac"]) / frac_diff_max

        return jax.lax.select(
            frac < config["warmup_frac"],
            config["initial_lr"] + delta * frac / config["warmup_frac"],
            config["peak_lr"] * jnp.maximum(0.0, 0.5 * (1.0 + jnp.cos(jnp.pi * ((frac_cosine) % 1.0)))),
        )

    time_start = time.time()

    def train(rng):
        last_time = time.time()
        # INIT NETWORK
        network = make_network_from_config(env, env_params, config)
        rng, _rng = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset, (0, None))(jax.random.split(_rng, config["num_train_envs"]), env_params)
        dones = jnp.zeros((config["num_train_envs"]), dtype=jnp.bool_)
        rng, _rng = jax.random.split(rng)
        init_hstate = ScannedRNN.initialize_carry(config["num_train_envs"])
        init_x = jax.tree.map(lambda x: x[None, ...], (obsv, dones))
        network_params = network.init(_rng, init_hstate, init_x)

        # ==== GRPO: snapshot initial policy as frozen reference ====
        ref_params = network_params
        # ==========================================================

        param_count = sum(x.size for x in jax.tree_util.tree_leaves(network_params))
        obs_size = sum(x.size for x in jax.tree_util.tree_leaves(obsv)) // config["num_train_envs"]

        print("Number of parameters", param_count, "size of obs: ", obs_size)
        if config["anneal_lr"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        elif config["warmup_lr"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adamw(learning_rate=linear_warmup_cosine_decay_schedule, eps=1e-5),
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
        if config["load_from_checkpoint"] != None:
            print("LOADING from", config["load_from_checkpoint"], "with only params =", config["load_only_params"])
            train_state = load_train_state_from_wandb_artifact_path(
                train_state, config["load_from_checkpoint"], load_only_params=config["load_only_params"]
            )
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset, (0, None))(jax.random.split(_rng, config["num_train_envs"]), env_params)
        init_hstate = ScannedRNN.initialize_carry(config["num_train_envs"])
        render_static_env_params = eval_env.static_env_params.replace(downscale=4)
        pixel_renderer = jax.jit(make_render_pixels(env_params, render_static_env_params))
        pixel_render_fn = lambda x: pixel_renderer(x) / 255.0

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
            # COLLECT TRAJECTORIES (unchanged)
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    hstate,
                    rng,
                    update_step,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                ac_in = (jax.tree.map(lambda x: x[np.newaxis, :], last_obs), last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    jax.random.split(_rng, config["num_train_envs"]), env_state, action, env_params
                )
                transition = Transition(last_done, action, value, reward, log_prob, last_obs, info)
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done,
                    hstate,
                    rng,
                    update_step,
                )
                return runner_state, transition

            initial_hstate = runner_state[-3]
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
            ac_in = (jax.tree.map(lambda x: x[np.newaxis, :], last_obs), last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["gamma"] * next_value * (1 - next_done) - value
                    gae = delta + config["gamma"] * config["gae_lambda"] * (1 - next_done) * gae
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        """GRPO loss function with group-normalized advantages and KL penalty."""
                        # Rerun network for current policy
                        _, pi, value = network.apply(params, init_hstate[0], (traj_batch.obs, traj_batch.done))
                        log_prob = pi.log_prob(traj_batch.action)

                        # --- VALUE LOSS (PPO-style clipped) ---
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["clip_eps"], config["clip_eps"]
                        )
                        v_loss1 = jnp.square(value - targets)
                        v_loss2 = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(v_loss1, v_loss2).mean()

                        # --- GROUP-NORMALIZED ADVANTAGES ---
                        group_size = config.get("group_size", 64)
                        eps = config.get("group_norm_epsilon", 1e-8)

                        flat_gae = gae.flatten()
                        pad_size = (-flat_gae.shape[0]) % group_size
                        padded_gae = jnp.concatenate([flat_gae, jnp.zeros(pad_size)])
                        groups = padded_gae.reshape(-1, group_size)

                        means = groups.mean(axis=1, keepdims=True)
                        stds = groups.std(axis=1, keepdims=True) + eps
                        normed_groups = (groups - means) / stds

                        normed_gae = normed_groups.flatten()[: flat_gae.shape[0]].reshape(gae.shape)

                        # --- POLICY SURROGATE LOSS (clipped) ---
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        loss_actor1 = ratio * normed_gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["clip_eps"], 1.0 + config["clip_eps"]) * normed_gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        # --- ENTROPY BONUS ---
                        entropy = pi.entropy().mean()

                        # --- KL PENALTY AGAINST FROZEN REFERENCE POLICY ---
                        # (Assuming ref_params was snapshotted at init)
                        _, ref_pi, _ = network.apply(ref_params, init_hstate[0], (traj_batch.obs, traj_batch.done))
                        ref_log_prob = ref_pi.log_prob(traj_batch.action)
                        kl = (log_prob - ref_log_prob).mean()

                        # --- TOTAL GRPO LOSS ---
                        total_loss = (
                            loss_actor + 0 * value_loss - config["ent_coef"] * entropy + config["grpo_beta"] * kl
                        )

                        return total_loss, (value_loss, loss_actor, entropy, kl)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["num_train_envs"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["update_epochs"])
            train_state = update_state[0]
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum() / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )
            rng = update_state[-1]

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
                        lambda: (_fake_video()),
                    )
                    return (first, should_log_videos, True, *to_log_videos[1:])

                def _fake_eval(rng, update_step):
                    return (
                        _fake_video(),
                        False,
                        False,
                        jnp.zeros((config["num_eval_levels"],), jnp.int32),  # lengths
                        jnp.zeros((config["num_eval_levels"],), jnp.float32),  # returns for video
                        jnp.zeros((config["num_eval_levels"],), jnp.float32),  # returns avg
                        jnp.zeros((config["num_eval_levels"],), jnp.float32),  # ep lengths avg
                        jnp.zeros((config["num_eval_levels"],), jnp.float32),  # solve avg
                    )

                rng, _rng = jax.random.split(rng)
                to_log_videos = jax.lax.cond(
                    update_step % config["eval_freq"] == 0, _real_eval, _fake_eval, _rng, update_step
                )

                def callback(metric, raw_info, loss_info, update_step, to_log_videos):
                    nonlocal last_time
                    time_now = time.time()
                    delta_time = time_now - last_time
                    last_time = time_now
                    dones = raw_info["returned_episode"]
                    to_log = {
                        "episode_return": (raw_info["returned_episode_returns"] * dones).sum()
                        / jnp.maximum(1, dones.sum()),
                        "episode_solved": (raw_info["returned_episode_solved"] * dones).sum()
                        / jnp.maximum(1, dones.sum()),
                        "episode_length": (raw_info["returned_episode_lengths"] * dones).sum()
                        / jnp.maximum(1, dones.sum()),
                        "num_completed_episodes": dones.sum(),
                        # Add GRPO-specific logging
                        "grpo/group_size": config.get("group_size", 64),
                        "grpo/norm_epsilon": config.get("group_norm_epsilon", 1e-8),
                    }
                    to_log["timing/num_updates"] = update_step
                    to_log["timing/num_env_steps"] = (
                        int(update_step) * int(config["num_steps"]) * int(config["num_train_envs"])
                    )
                    to_log["timing/sps"] = (config["num_steps"] * config["num_train_envs"]) / delta_time
                    to_log["timing/sps_agg"] = (to_log["timing/num_env_steps"]) / (time_now - time_start)
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
                        to_log["eval/mean_eval_return"] = eval_return_mean.mean()
                        to_log["eval/mean_eval_eplen"] = eval_eplen_mean.mean()
                        to_log["eval/mean_eval_solve"] = eval_solverate_mean.mean()
                        for i, eval_name in enumerate(config["eval_levels"]):
                            return_on_video = eval_return_vid[i]
                            to_log[f"eval_video/return_{eval_name}"] = return_on_video
                            to_log[f"eval_video/len_{eval_name}"] = idx_vid[i]
                            to_log[f"eval_avg/return_{eval_name}"] = eval_return_mean[i]
                            to_log[f"eval_avg/solve_rate_{eval_name}"] = eval_solverate_mean[i]

                    if should_log_videos:
                        for i, eval_name in enumerate(config["eval_levels"]):
                            obs_to_use = obs_vid[: idx_vid[i], i]
                            obs_to_use = np.asarray(obs_to_use).transpose(0, 3, 2, 1)[:, :, ::-1, :]
                            to_log[f"media/eval_video_{eval_name}"] = wandb.Video(
                                (obs_to_use * 255).astype(np.uint8), fps=15
                            )

                    wandb.log(to_log)

                jax.debug.callback(callback, metric, traj_batch.info, loss_info, update_step, to_log_videos)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["num_train_envs"]), dtype=bool),
            init_hstate,
            _rng,
            0,
        )
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["num_updates"])
        return {"runner_state": runner_state, "metric": metric}

    return train


@hydra.main(version_base=None, config_path="../configs", config_name="grpo")  # Changed to grpo
def main(config):
    config = normalise_config(OmegaConf.to_container(config), "GRPO")  # Changed to GRPO
    env_params, static_env_params = generate_params_from_config(config)
    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)

    if config["use_wandb"]:
        run = init_wandb(config, "GRPO")  # Changed to GRPO

    rng = jax.random.PRNGKey(config["seed"])
    rng, _rng = jax.random.split(rng)
    train_jit = jax.jit(make_train(config, env_params, static_env_params))

    out = train_jit(_rng)

    if config["save_policy"]:
        train_state = jax.tree.map(lambda x: x, out["runner_state"][0])
        save_model(train_state, config["total_timesteps"], config, save_to_wandb=config["use_wandb"])


if __name__ == "__main__":
    main()
