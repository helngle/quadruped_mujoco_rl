from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import quadruped_mujoco_rl  # noqa: F401
from quadruped_mujoco_rl.utils.config import load_yaml
from quadruped_mujoco_rl.utils.paths import resolve_project_path


def make_env(env_id: str, config_path: str, seed: int, rank: int):
    def _init():
        env = gym.make(env_id, config_path=config_path)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def build_model(env, config: dict[str, Any], log_dir: Path) -> PPO:
    ppo_config = config["ppo"]
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=float(ppo_config["learning_rate"]),
        n_steps=int(ppo_config["n_steps"]),
        batch_size=int(ppo_config["batch_size"]),
        n_epochs=int(ppo_config["n_epochs"]),
        gamma=float(ppo_config["gamma"]),
        gae_lambda=float(ppo_config["gae_lambda"]),
        clip_range=float(ppo_config["clip_range"]),
        ent_coef=float(ppo_config["ent_coef"]),
        vf_coef=float(ppo_config["vf_coef"]),
        max_grad_norm=float(ppo_config["max_grad_norm"]),
        tensorboard_log=str(log_dir),
        verbose=1,
        seed=int(config["seed"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO quadruped policy.")
    parser.add_argument("--config", default="configs/train_ppo.yaml", help="Training YAML config.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total_timesteps from the YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    seed = int(config["seed"])
    total_timesteps = args.total_timesteps or int(config["total_timesteps"])
    log_dir = resolve_project_path(config["log_dir"])
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_config = config["env"]
    env = DummyVecEnv(
        [
            make_env(
                env_id=env_config["id"],
                config_path=env_config["config"],
                seed=seed,
                rank=rank,
            )
            for rank in range(int(env_config["num_envs"]))
        ]
    )

    try:
        model = build_model(env, config, log_dir)
        model.learn(total_timesteps=total_timesteps, tb_log_name="ppo")

        final_path = checkpoint_dir / "final_model"
        model.save(final_path)
        print(f"Saved model to {final_path.with_suffix('.zip')}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
