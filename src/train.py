import random
from network import DQN
from flax.training import train_state
import collections
import csv
import json
import os
import shutil
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
import argparse
import gymnasium as gym
import ale_py
from tqdm import tqdm

class ReplayBuffer():
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

def create_train_state(num_actions, shape, lr):
    model = DQN(num_actions)
    variables = model.init(jax.random.key(0), jnp.ones(shape))

    tx = optax.adam(lr)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables['params'], tx=tx
    )
    return state

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
        f'<text x="{x_pos(x):.2f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-size="12">{x:.0f}</text>'
        for x in x_ticks
    )
    y_tick_svg = "\n".join(
        f'<line x1="{margin_left - 6}" y1="{y_pos(y):.2f}" x2="{margin_left}" y2="{y_pos(y):.2f}" stroke="#333" />'
        f'<text x="{margin_left - 10}" y="{y_pos(y) + 4:.2f}" text-anchor="end" font-size="12">{y:.2f}</text>'
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

@jax.jit
def train_step(state, obs, actions, rewards, next_obs, dones, gamma):
    def loss_fn(params):
        q_vals = state.apply_fn({'params': params}, obs)
        action_q_vals = jnp.take_along_axis(
            q_vals,
            actions[:, None],
            axis=1,
        ).squeeze(axis=1)

        next_q_vals = state.apply_fn({'params': params}, next_obs)
        next_max_q_vals = jnp.max(next_q_vals, axis=1)
        targets = rewards + gamma * (1.0 - dones) * next_max_q_vals

        loss = optax.l2_loss(action_q_vals, jax.lax.stop_gradient(targets))
        return jnp.mean(loss)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DQN agent.')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--env_name', type=str, default='ALE/SpaceInvaders-v5', help='Environment name')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='')
    parser.add_argument('--eps', type=float, default=1.0, help='')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--gamma', type=float, default=0.99, help='')
    parser.add_argument('--output_dir', type=str, default='train_outputs', help='')
    return parser.parse_args()

def preprocess(obs):
    """Convert one raw Atari RGB frame to an 84x84 grayscale uint8 frame."""
    obs = jnp.asarray(obs)

    if obs.ndim == 3 and obs.shape[-1] == 3:
        obs = (
            0.299 * obs[..., 0]
            + 0.587 * obs[..., 1]
            + 0.114 * obs[..., 2]
        )

    # Resize from roughly 210x160 to 110x84, preserving Atari aspect ratio better.
    obs = jax.image.resize(obs, (110, 84), method="linear")

    # Crop vertical center/play area down to 84x84.
    obs = obs[18:102, :]

    obs = jnp.clip(obs, 0, 255).astype(jnp.uint8)
    return np.asarray(obs)

def train(num_episodes, env_name, buffer_size, batch_size, lr, eps, gamma, output_dir):
    prepare_output_dir(output_dir)
    gym.register_envs(ale_py)
    env = gym.make(env_name)
    train_state = create_train_state(env.action_space.n, (1,84,84,4), lr)
    replay_buffer = ReplayBuffer(max_size=buffer_size, batch_size=batch_size)
    total_reward = 0
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
        "epsilon": eps,
        "gamma": gamma,
    }
    for i in tqdm(range(num_episodes)):
        episode_start_time = time.perf_counter()
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_clipped_reward = 0
        episode_length = 0
        episode_losses = []
        frame = preprocess(obs)
        stacked_obs = np.stack([frame] * 4, axis=-1)

        while not done:
            # select action with eps greedy
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                q_vals = train_state.apply_fn({'params': train_state.params}, stacked_obs[None, ...])
                action = int(jnp.argmax(q_vals[0]))

            next_obs, reward, terminated, truncated, info = env.step(action)
            clipped_reward = np.sign(reward)
            episode_reward += reward
            episode_clipped_reward += clipped_reward
            episode_length += 1
            global_step += 1
            done = terminated or truncated
            next_frame = preprocess(next_obs)
            next_stacked_obs = np.concatenate([stacked_obs[:, :, 1:], next_frame[:, :, None]],axis=-1,)

            # add experience to replay buffer
            replay_buffer.append((stacked_obs, action, clipped_reward, next_stacked_obs, done))
            stacked_obs = next_stacked_obs

            # sample random minibatch, perform update step
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample_batch()
                obs_batch, actions, rewards, next_obs_batch, dones = zip(*batch)

                obs_batch = jnp.asarray(np.stack(obs_batch))
                actions = jnp.asarray(actions, dtype=jnp.int32)
                rewards = jnp.asarray(rewards, dtype=jnp.float32)
                next_obs_batch = jnp.asarray(np.stack(next_obs_batch))
                dones = jnp.asarray(dones, dtype=jnp.float32)

                train_state, loss = train_step(
                    train_state,
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
                    "episode": i + 1,
                    "loss": loss_value,
                    "replay_size": len(replay_buffer),
                })

        total_reward += episode_reward
        episode_seconds = time.perf_counter() - episode_start_time
        episode_rows.append({
            "episode": i + 1,
            "global_step": global_step,
            "raw_reward": float(episode_reward),
            "clipped_reward": float(episode_clipped_reward),
            "length": episode_length,
            "seconds": episode_seconds,
            "mean_loss": float(np.mean(episode_losses)) if episode_losses else "",
            "updates": len(episode_losses),
            "replay_size": len(replay_buffer),
            "epsilon": eps,
        })

        if (i + 1) % 10 == 0:
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

    env.close()

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

if '__main__' == __name__:
    args = parse_args()
    train(args.num_episodes, args.env_name, args.buffer_size, args.batch_size, args.lr, args.eps, args.gamma, args.output_dir)
