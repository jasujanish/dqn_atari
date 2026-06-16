import gymnasium as gym
import ale_py
import argparse
import statistics
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DQN agent.')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--env_name', type=str, default='ALE/SpaceInvaders-v5', help='Environment name')
    return parser.parse_args()

def random_playout(num_episodes, env_name):
    gym.register_envs(ale_py)
    env = gym.make(env_name)
    episode_rewards = []
    episode_lengths = []
    terminated_count = 0
    truncated_count = 0

    # play `num_episodes` trajectories 
    for episode in tqdm(range(num_episodes)):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            obs = next_obs

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        terminated_count += int(terminated)
        truncated_count += int(truncated)

    env.close()

    return {
        'env_name': args.env_name,
        'num_episodes': args.num_episodes,
        'reward_mean': statistics.fmean(episode_rewards),
        'reward_median': statistics.median(episode_rewards),
        'reward_min': min(episode_rewards),
        'reward_max': max(episode_rewards),
        'reward_std': statistics.pstdev(episode_rewards),
        'length_mean': statistics.fmean(episode_lengths),
        'length_median': statistics.median(episode_lengths),
        'length_min': min(episode_lengths),
        'length_max': max(episode_lengths),
        'length_std': statistics.pstdev(episode_lengths),
        'terminated_count': terminated_count,
        'truncated_count': truncated_count,
    }

def print_summary(stats):
    print(f"Environment: {stats['env_name']}")
    print(f"Episodes: {stats['num_episodes']}")
    print(
        'Reward: '
        f"mean={stats['reward_mean']:.2f}, "
        f"median={stats['reward_median']:.2f}, "
        f"std={stats['reward_std']:.2f}, "
        f"min={stats['reward_min']:.2f}, "
        f"max={stats['reward_max']:.2f}"
    )
    print(
        'Length: '
        f"mean={stats['length_mean']:.2f}, "
        f"median={stats['length_median']:.2f}, "
        f"std={stats['length_std']:.2f}, "
        f"min={stats['length_min']}, "
        f"max={stats['length_max']}"
    )
    print(
        'Episode endings: '
        f"terminated={stats['terminated_count']}, "
        f"truncated={stats['truncated_count']}"
    )

if '__main__' == __name__:
    args = parse_args()
    summary = random_playout(num_episodes=args.num_episodes, env_name=args.env_name)
    print_summary(summary)

