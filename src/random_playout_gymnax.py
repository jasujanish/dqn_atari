import argparse
import statistics

from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Run random Gymnax playouts.")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of random episodes to run.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="SpaceInvaders-MinAtar",
        help="Gymnax environment name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional per-episode step cap. Episodes hitting this cap are stopped.",
    )
    return parser.parse_args()


def random_playout(num_episodes, env_name, seed, max_steps=None):
    import gymnax
    import jax

    if num_episodes <= 0:
        raise ValueError("--num_episodes must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("--max_steps must be positive when provided")

    env, env_params = gymnax.make(env_name)
    action_space = env.action_space(env_params)
    key = jax.random.key(seed)

    episode_rewards = []
    episode_lengths = []
    done_count = 0
    max_steps_count = 0

    for _ in tqdm(range(num_episodes)):
        key, key_reset = jax.random.split(key)
        obs, info = env.reset(key_reset, env_params)

        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            if max_steps is not None and episode_length >= max_steps:
                max_steps_count += 1
                break

            key, key_policy, key_step = jax.random.split(key, 3)
            action = action_space.sample(key_policy)
            obs, info, reward, done, _ = env.step(
                key_step,
                info,
                action,
                env_params,
            )

            episode_reward += float(reward)
            episode_length += 1
            done = bool(done)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        done_count += int(done)

    return {
        "env_name": env_name,
        "num_episodes": num_episodes,
        "seed": seed,
        "max_steps": max_steps,
        "reward_mean": statistics.fmean(episode_rewards),
        "reward_median": statistics.median(episode_rewards),
        "reward_min": min(episode_rewards),
        "reward_max": max(episode_rewards),
        "reward_std": statistics.pstdev(episode_rewards),
        "length_mean": statistics.fmean(episode_lengths),
        "length_median": statistics.median(episode_lengths),
        "length_min": min(episode_lengths),
        "length_max": max(episode_lengths),
        "length_std": statistics.pstdev(episode_lengths),
        "done_count": done_count,
        "max_steps_count": max_steps_count,
    }


def print_summary(stats):
    print(f"Environment: {stats['env_name']}")
    print(f"Episodes: {stats['num_episodes']}")
    print(f"Seed: {stats['seed']}")
    if stats["max_steps"] is not None:
        print(f"Max steps: {stats['max_steps']}")
    print(
        "Reward: "
        f"mean={stats['reward_mean']:.2f}, "
        f"median={stats['reward_median']:.2f}, "
        f"std={stats['reward_std']:.2f}, "
        f"min={stats['reward_min']:.2f}, "
        f"max={stats['reward_max']:.2f}"
    )
    print(
        "Length: "
        f"mean={stats['length_mean']:.2f}, "
        f"median={stats['length_median']:.2f}, "
        f"std={stats['length_std']:.2f}, "
        f"min={stats['length_min']}, "
        f"max={stats['length_max']}"
    )
    print(
        "Episode endings: "
        f"done={stats['done_count']}, "
        f"max_steps={stats['max_steps_count']}"
    )


if __name__ == "__main__":
    args = parse_args()
    summary = random_playout(
        num_episodes=args.num_episodes,
        env_name=args.env_name,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    print_summary(summary)
