from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import quadruped_mujoco_rl  # noqa: F401
from quadruped_mujoco_rl.utils.paths import resolve_project_path


@dataclass
class EpisodeStats:
    episode: int
    total_reward: float
    length: int
    distance: float
    mean_forward_velocity: float
    mean_base_height: float
    mean_abs_roll: float
    mean_abs_pitch: float
    mean_action_abs: float
    terminated: bool
    truncated: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO quadruped policy.")
    parser.add_argument(
        "--checkpoint",
        default="runs/ppo/checkpoints/final_model.zip",
        help="Path to a saved PPO checkpoint.",
    )
    parser.add_argument("--config", default="configs/env_flat.yaml", help="Environment YAML config.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to evaluate.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run evaluation without opening the MuJoCo viewer.",
    )
    return parser.parse_args()


def evaluate_episode(env: gym.Env, model: PPO, episode: int, render: bool) -> EpisodeStats:
    obs, info = env.reset()
    start_x = float(env.unwrapped.data.qpos[0])
    total_reward = 0.0
    length = 0
    terminated = False
    truncated = False

    forward_velocities = []
    base_heights = []
    abs_rolls = []
    abs_pitches = []
    action_abs = []

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        length += 1
        forward_velocities.append(float(info["forward_velocity"]))
        base_heights.append(float(info["base_height"]))
        abs_rolls.append(abs(float(info["roll"])))
        abs_pitches.append(abs(float(info["pitch"])))
        action_abs.append(float(np.mean(np.abs(action))))

        if render:
            time.sleep(env.unwrapped.control_timestep)

    distance = float(env.unwrapped.data.qpos[0] - start_x)
    return EpisodeStats(
        episode=episode,
        total_reward=total_reward,
        length=length,
        distance=distance,
        mean_forward_velocity=float(np.mean(forward_velocities)),
        mean_base_height=float(np.mean(base_heights)),
        mean_abs_roll=float(np.mean(abs_rolls)),
        mean_abs_pitch=float(np.mean(abs_pitches)),
        mean_action_abs=float(np.mean(action_abs)),
        terminated=terminated,
        truncated=truncated,
    )


def print_episode_stats(stats: EpisodeStats) -> None:
    end_reason = "terminated" if stats.terminated else "truncated"
    print(
        f"episode={stats.episode} "
        f"reward={stats.total_reward:.2f} "
        f"length={stats.length} "
        f"distance={stats.distance:.2f}m "
        f"mean_vx={stats.mean_forward_velocity:.2f}m/s "
        f"mean_height={stats.mean_base_height:.2f}m "
        f"mean_abs_roll={stats.mean_abs_roll:.3f}rad "
        f"mean_abs_pitch={stats.mean_abs_pitch:.3f}rad "
        f"mean_abs_action={stats.mean_action_abs:.3f} "
        f"end={end_reason}"
    )


def print_summary(all_stats: list[EpisodeStats]) -> None:
    print("\nSummary")
    print("-------")
    print(f"episodes={len(all_stats)}")
    print(f"mean_reward={np.mean([stats.total_reward for stats in all_stats]):.2f}")
    print(f"mean_length={np.mean([stats.length for stats in all_stats]):.1f}")
    print(f"mean_distance={np.mean([stats.distance for stats in all_stats]):.2f}m")
    print(f"mean_forward_velocity={np.mean([stats.mean_forward_velocity for stats in all_stats]):.2f}m/s")
    print(f"mean_base_height={np.mean([stats.mean_base_height for stats in all_stats]):.2f}m")
    print(f"terminated_count={sum(stats.terminated for stats in all_stats)}")
    print(f"truncated_count={sum(stats.truncated for stats in all_stats)}")


def main() -> None:
    args = parse_args()
    checkpoint = resolve_project_path(args.checkpoint)

    render_mode = None if args.no_render else "human"
    env = gym.make("QuadrupedFlat-v0", config_path=args.config, render_mode=render_mode)
    model = PPO.load(checkpoint)

    try:
        all_stats = []
        for episode in range(1, args.episodes + 1):
            stats = evaluate_episode(env, model, episode=episode, render=not args.no_render)
            all_stats.append(stats)
            print_episode_stats(stats)

        print_summary(all_stats)
    finally:
        env.close()


if __name__ == "__main__":
    main()
