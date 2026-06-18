from __future__ import annotations

import argparse
import time

import gymnasium as gym
from stable_baselines3 import PPO

import quadruped_mujoco_rl  # noqa: F401
from quadruped_mujoco_rl.utils.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO quadruped policy.")
    parser.add_argument(
        "--checkpoint",
        default="runs/ppo/checkpoints/final_model.zip",
        help="Path to a saved PPO checkpoint.",
    )
    parser.add_argument("--config", default="configs/env_flat.yaml", help="Environment YAML config.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to watch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = resolve_project_path(args.checkpoint)

    env = gym.make("QuadrupedFlat-v0", config_path=args.config, render_mode="human")
    model = PPO.load(checkpoint)

    try:
        for _ in range(args.episodes):
            obs, _ = env.reset()
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                time.sleep(env.unwrapped.control_timestep)
    finally:
        env.close()


if __name__ == "__main__":
    main()
