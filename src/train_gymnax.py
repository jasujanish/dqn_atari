import argparse
import collections
import csv
import json
import os
import random
import shutil
import time

import flax.linen as nn
from flax.training import train_state
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.auto import tqdm


class ReplayBuffer:
    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer = collections.deque(maxlen=max_size)

    def sample_batch(self):
        if len(self.buffer) < self.batch_size:
            return None
        return random.sample(self.buffer, self.batch_size)

    def append(self, transition):
        self.buffer.append(transition)

    def __len__(self):
        return len(self.buffer)


class MinAtarDQN(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        return nn.Dense(features=self.num_actions)(x)


def create_train_state(num_actions, obs_shape, lr, seed):
    model = MinAtarDQN(num_actions)
    variables = model.init(
        jax.random.key(seed),
        jnp.ones((1, *obs_shape), dtype=jnp.float32),
    )
    tx = optax.adam(lr)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )

@jax.jit
def train_step(agent_state, obs, actions, rewards, next_obs, dones, gamma):
    def loss_fn(params):
        q_vals = agent_state.apply_fn({"params": params}, obs)
        action_q_vals = jnp.take_along_axis(
            q_vals,
            actions[:, None],
            axis=1,
        ).squeeze(axis=1)

        next_q_vals = agent_state.apply_fn({"params": params}, next_obs)
        next_max_q_vals = jnp.max(next_q_vals, axis=1)
        targets = rewards + gamma * (1.0 - dones) * next_max_q_vals

        losses = optax.l2_loss(action_q_vals, jax.lax.stop_gradient(targets))
        return jnp.mean(losses)

    loss, grads = jax.value_and_grad(loss_fn)(agent_state.params)
    agent_state = agent_state.apply_gradients(grads=grads)
    return agent_state, loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent with Gymnax.")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of training episodes.")
    parser.add_argument("--env_name", type=str, default="SpaceInvaders-MinAtar", help="Gymnax environment name.")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Adam learning rate.")
    parser.add_argument("--eps", type=float, default=1.0, help="Starting epsilon for epsilon-greedy exploration.")
    parser.add_argument("--eps_end", type=float, default=0.1, help="Final epsilon after annealing.")
    parser.add_argument("--random_anneal", type=int, default=100, help="Episodes used to anneal epsilon.")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer capacity.")
    parser.add_argument("--batch_size", type=int, default=32, help="Replay minibatch size.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--learning_starts", type=int, default=1000, help="Env steps before gradient updates.")
    parser.add_argument("--train_freq", type=int, default=1, help="Run one update every N env steps.")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional per-episode step cap.")
    parser.add_argument("--output_dir", type=str, default="train_outputs_gymnax", help="Output directory.")
    return parser.parse_args()


def prepare_output_dir(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)


def write_metrics_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_line_graph(path, title, x_label, y_label, xs, ys):
    if not xs or not ys:
        return

    width = 900
    height = 520
    margin_left = 72
    margin_right = 24
    margin_top = 56
    margin_bottom = 64
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min -= 1
        y_max += 1

    def x_pos(x):
        return margin_left + ((x - x_min) / (x_max - x_min)) * plot_width

    def y_pos(y):
        return margin_top + (1.0 - ((y - y_min) / (y_max - y_min))) * plot_height

    points = " ".join(f"{x_pos(x):.2f},{y_pos(y):.2f}" for x, y in zip(xs, ys))
    x_ticks = [x_min + (x_max - x_min) * i / 4 for i in range(5)]
    y_ticks = [y_min + (y_max - y_min) * i / 4 for i in range(5)]

    x_tick_svg = "\n".join(
        f'<line x1="{x_pos(x):.2f}" y1="{margin_top + plot_height}" x2="{x_pos(x):.2f}" y2="{margin_top + plot_height + 6}" stroke="#333" />'
        f'<text x="{x_pos(x):.2f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif">{x:.0f}</text>'
        for x in x_ticks
    )
    y_tick_svg = "\n".join(
        f'<line x1="{margin_left - 6}" y1="{y_pos(y):.2f}" x2="{margin_left}" y2="{y_pos(y):.2f}" stroke="#333" />'
        f'<text x="{margin_left - 10}" y="{y_pos(y) + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial, sans-serif">{y:.2f}</text>'
        for y in y_ticks
    )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-family="Arial, sans-serif">{title}</text>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333" />
<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#333" />
{x_tick_svg}
{y_tick_svg}
<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{points}" />
<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="14" font-family="Arial, sans-serif">{x_label}</text>
<text x="18" y="{height / 2}" text-anchor="middle" font-size="14" font-family="Arial, sans-serif" transform="rotate(-90 18 {height / 2})">{y_label}</text>
</svg>
'''

    with open(path, "w") as f:
        f.write(svg)


def save_training_outputs(
    output_dir,
    episode_rows,
    update_rows,
    config,
    total_reward,
    global_step,
    update_step,
    elapsed_seconds,
):
    write_metrics_csv(os.path.join(output_dir, "episodes.csv"), episode_rows)
    write_metrics_csv(os.path.join(output_dir, "updates.csv"), update_rows)

    completed_episodes = len(episode_rows)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump({
            **config,
            "episodes_completed": completed_episodes,
            "mean_raw_reward": total_reward / completed_episodes if completed_episodes else 0,
            "total_env_steps": global_step,
            "total_updates": update_step,
            "elapsed_seconds": elapsed_seconds,
            "avg_seconds_per_episode": elapsed_seconds / completed_episodes if completed_episodes else 0,
            "avg_env_steps_per_second": global_step / elapsed_seconds if elapsed_seconds > 0 else 0,
        }, f, indent=2)

    episodes = [row["episode"] for row in episode_rows]
    save_line_graph(
        os.path.join(output_dir, "episode_raw_reward.svg"),
        "Episode Raw Reward",
        "Episode",
        "Raw Reward",
        episodes,
        [row["raw_reward"] for row in episode_rows],
    )
    save_line_graph(
        os.path.join(output_dir, "episode_clipped_reward.svg"),
        "Episode Clipped Reward",
        "Episode",
        "Clipped Reward",
        episodes,
        [row["clipped_reward"] for row in episode_rows],
    )
    save_line_graph(
        os.path.join(output_dir, "episode_length.svg"),
        "Episode Length",
        "Episode",
        "Steps",
        episodes,
        [row["length"] for row in episode_rows],
    )
    save_line_graph(
        os.path.join(output_dir, "loss.svg"),
        "Training Loss",
        "Update",
        "Loss",
        [row["update"] for row in update_rows],
        [row["loss"] for row in update_rows],
    )


def epsilon_for_episode(episode_idx, eps_start, eps_end, random_anneal):
    if random_anneal <= 0:
        return eps_end
    anneal_fraction = min(episode_idx / random_anneal, 1.0)
    return eps_start + anneal_fraction * (eps_end - eps_start)


def train(
    num_episodes,
    env_name,
    buffer_size,
    batch_size,
    lr,
    eps,
    eps_end,
    random_anneal,
    gamma,
    seed,
    learning_starts,
    train_freq,
    max_steps,
    output_dir,
):
    if num_episodes <= 0:
        raise ValueError("--num_episodes must be positive")
    if batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if buffer_size < batch_size:
        raise ValueError("--buffer_size must be at least --batch_size")
    if learning_starts < 0:
        raise ValueError("--learning_starts must be non-negative")
    if train_freq <= 0:
        raise ValueError("--train_freq must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("--max_steps must be positive when provided")

    prepare_output_dir(output_dir)

    env, env_params = gymnax.make(env_name)
    action_space = env.action_space(env_params)
    num_actions = action_space.n
    key = jax.random.key(seed)
    key, key_reset = jax.random.split(key)
    first_obs, _ = env.reset(key_reset, env_params)

    agent_state = create_train_state(
        num_actions=num_actions,
        obs_shape=first_obs.shape,
        lr=lr,
        seed=seed,
    )
    replay_buffer = ReplayBuffer(max_size=buffer_size, batch_size=batch_size)

    total_reward = 0.0
    global_step = 0
    update_step = 0
    episode_rows = []
    update_rows = []
    train_start_time = time.perf_counter()
    config = {
        "env_name": env_name,
        "num_episodes": num_episodes,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "lr": lr,
        "epsilon_start": eps,
        "epsilon_end": eps_end,
        "random_anneal": random_anneal,
        "gamma": gamma,
        "seed": seed,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "max_steps": max_steps,
        "obs_shape": list(first_obs.shape),
        "num_actions": int(num_actions),
    }

    for episode_idx in tqdm(range(num_episodes)):
        episode_eps = epsilon_for_episode(episode_idx, eps, eps_end, random_anneal)
        episode_start_time = time.perf_counter()

        key, key_reset = jax.random.split(key)
        obs, env_state = env.reset(key_reset, env_params)
        done = False
        episode_reward = 0.0
        episode_clipped_reward = 0.0
        episode_length = 0
        episode_losses = []

        while not done:
            if max_steps is not None and episode_length >= max_steps:
                done = True
                break

            key, key_eps, key_action, key_step = jax.random.split(key, 4)
            if float(jax.random.uniform(key_eps)) < episode_eps:
                action = action_space.sample(key_action)
            else:
                q_vals = agent_state.apply_fn(
                    {"params": agent_state.params},
                    obs[None, ...],
                )
                action = jnp.argmax(q_vals[0]).astype(jnp.int32)

            next_obs, next_env_state, reward, env_done, info = env.step(
                key_step,
                env_state,
                action,
                env_params,
            )

            reward_value = float(reward)
            clipped_reward = float(np.sign(reward_value))
            done = bool(env_done)
            replay_buffer.append((
                np.asarray(obs),
                int(action),
                clipped_reward,
                np.asarray(next_obs),
                float(done),
            ))

            obs = next_obs
            env_state = next_env_state
            episode_reward += reward_value
            episode_clipped_reward += clipped_reward
            episode_length += 1
            global_step += 1

            if (
                global_step >= learning_starts
                and global_step % train_freq == 0
                and len(replay_buffer) >= batch_size
            ):
                batch = replay_buffer.sample_batch()
                obs_batch, actions, rewards, next_obs_batch, dones = zip(*batch)

                obs_batch = jnp.asarray(np.stack(obs_batch), dtype=jnp.float32)
                actions = jnp.asarray(actions, dtype=jnp.int32)
                rewards = jnp.asarray(rewards, dtype=jnp.float32)
                next_obs_batch = jnp.asarray(np.stack(next_obs_batch), dtype=jnp.float32)
                dones = jnp.asarray(dones, dtype=jnp.float32)

                agent_state, loss = train_step(
                    agent_state,
                    obs_batch,
                    actions,
                    rewards,
                    next_obs_batch,
                    dones,
                    gamma,
                )
                update_step += 1
                loss_value = float(loss)
                episode_losses.append(loss_value)
                update_rows.append({
                    "update": update_step,
                    "global_step": global_step,
                    "episode": episode_idx + 1,
                    "loss": loss_value,
                    "replay_size": len(replay_buffer),
                })

        total_reward += episode_reward
        episode_seconds = time.perf_counter() - episode_start_time
        episode_rows.append({
            "episode": episode_idx + 1,
            "global_step": global_step,
            "raw_reward": episode_reward,
            "clipped_reward": episode_clipped_reward,
            "length": episode_length,
            "seconds": episode_seconds,
            "mean_loss": float(np.mean(episode_losses)) if episode_losses else "",
            "updates": len(episode_losses),
            "replay_size": len(replay_buffer),
            "epsilon": episode_eps,
        })

        if (episode_idx + 1) % 10 == 0:
            save_training_outputs(
                output_dir,
                episode_rows,
                update_rows,
                config,
                total_reward,
                global_step,
                update_step,
                time.perf_counter() - train_start_time,
            )

    save_training_outputs(
        output_dir,
        episode_rows,
        update_rows,
        config,
        total_reward,
        global_step,
        update_step,
        time.perf_counter() - train_start_time,
    )


if __name__ == "__main__":
    args = parse_args()
    train(
        num_episodes=args.num_episodes,
        env_name=args.env_name,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        lr=args.lr,
        eps=args.eps,
        eps_end=args.eps_end,
        random_anneal=args.random_anneal,
        gamma=args.gamma,
        seed=args.seed,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )
