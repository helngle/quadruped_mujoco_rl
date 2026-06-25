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
        device=str(config.get("device", "auto")),
    )


def transfer_policy_weights(source: PPO, target: PPO) -> tuple[int, int, list[str]]:
    """Copy a policy into a compatible policy with extra observation inputs."""
    source_state = source.policy.state_dict()
    target_state = target.policy.state_dict()
    expanded_input_keys = {
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.value_net.0.weight",
    }
    copied = 0
    expanded = 0
    skipped = []

    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None:
            skipped.append(key)
            continue

        source_tensor = source_tensor.to(
            device=target_tensor.device,
            dtype=target_tensor.dtype,
        )
        if source_tensor.shape == target_tensor.shape:
            target_state[key] = source_tensor.clone()
            copied += 1
            continue

        can_expand_input = (
            key in expanded_input_keys
            and source_tensor.ndim == 2
            and target_tensor.ndim == 2
            and source_tensor.shape[0] == target_tensor.shape[0]
            and source_tensor.shape[1] < target_tensor.shape[1]
        )
        if can_expand_input:
            expanded_tensor = target_tensor.clone()
            expanded_tensor[:, : source_tensor.shape[1]] = source_tensor
            expanded_tensor[:, source_tensor.shape[1] :] = 0.0
            target_state[key] = expanded_tensor
            expanded += 1
            continue

        skipped.append(key)

    target.policy.load_state_dict(target_state)
    return copied, expanded, skipped


def initialize_policy_from_checkpoint(model: PPO, checkpoint: str, device: str) -> None:
    source = PPO.load(resolve_project_path(checkpoint), device=device)
    if source.action_space.shape != model.action_space.shape:
        raise ValueError(
            "Source and target action spaces must match: "
            f"{source.action_space.shape} != {model.action_space.shape}"
        )

    copied, expanded, skipped = transfer_policy_weights(source, model)
    print(
        f"Initialized policy from {resolve_project_path(checkpoint)} "
        f"(copied={copied}, expanded_inputs={expanded}, skipped={len(skipped)})"
    )
    if skipped:
        print(f"Skipped incompatible policy tensors: {', '.join(skipped)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO quadruped policy.")
    parser.add_argument("--config", default="configs/train_ppo.yaml", help="Training YAML config.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total_timesteps from the YAML config.",
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume",
        default=None,
        help="Resume PPO weights and optimizer state from a checkpoint.",
    )
    checkpoint_group.add_argument(
        "--init-from",
        default=None,
        help="Initialize compatible policy weights without restoring optimizer state.",
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
        if args.resume:
            model = PPO.load(
                resolve_project_path(args.resume),
                env=env,
                tensorboard_log=str(log_dir),
                device=str(config.get("device", "auto")),
            )
        else:
            model = build_model(env, config, log_dir)
            if args.init_from:
                initialize_policy_from_checkpoint(
                    model,
                    checkpoint=args.init_from,
                    device=str(config.get("device", "auto")),
                )
        model.learn(
            total_timesteps=total_timesteps,
            tb_log_name="ppo",
            reset_num_timesteps=not bool(args.resume),
        )

        final_path = checkpoint_dir / "final_model"
        model.save(final_path)
        print(f"Saved model to {final_path.with_suffix('.zip')}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
