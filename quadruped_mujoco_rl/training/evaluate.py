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
    lateral_drift: float
    yaw_drift: float
    mean_forward_velocity: float
    mean_lateral_velocity: float
    mean_world_vx: float
    mean_world_vy: float
    mean_body_vx: float
    mean_body_vy: float
    mean_yaw_rate: float
    mean_abs_vx_error: float
    mean_abs_yaw_rate_error: float
    mean_base_height: float
    mean_abs_roll: float
    mean_abs_pitch: float
    mean_abs_yaw: float
    mean_abs_joint_velocity: float
    mean_action_abs: float
    mean_ctrl_abs: float
    foot_contact_ratios: dict[str, float]
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
    parser.add_argument("--target-vx", type=float, default=None, help="Fixed forward command.")
    parser.add_argument(
        "--target-yaw-rate",
        type=float,
        default=None,
        help="Fixed yaw-rate command.",
    )
    return parser.parse_args()


def evaluate_episode(
    env: gym.Env,
    model: PPO,
    episode: int,
    render: bool,
    command: tuple[float, float] | None,
) -> EpisodeStats:
    options = {"command": command} if command is not None else None
    obs, info = env.reset(options=options)
    start_x = float(env.unwrapped.data.qpos[0])
    start_y = float(env.unwrapped.data.qpos[1])
    start_yaw = float(info["yaw"])
    total_reward = 0.0
    length = 0
    terminated = False
    truncated = False

    forward_velocities = []
    lateral_velocities = []
    world_velocities_x = []
    world_velocities_y = []
    body_velocities_x = []
    body_velocities_y = []
    yaw_rates = []
    abs_vx_errors = []
    abs_yaw_rate_errors = []
    base_heights = []
    abs_rolls = []
    abs_pitches = []
    abs_yaws = []
    abs_joint_velocities = []
    action_abs = []
    ctrl_abs = []
    foot_contact_counts = {name: 0 for name in env.unwrapped.foot_geom_names}

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        length += 1
        forward_velocities.append(float(info["forward_velocity"]))
        lateral_velocities.append(float(info["lateral_velocity"]))
        world_velocities_x.append(float(info["world_vx"]))
        world_velocities_y.append(float(info["world_vy"]))
        body_velocities_x.append(float(info["body_vx"]))
        body_velocities_y.append(float(info["body_vy"]))
        yaw_rates.append(float(info["yaw_rate"]))
        abs_vx_errors.append(abs(float(info["forward_velocity"] - info["target_vx"])))
        abs_yaw_rate_errors.append(abs(float(info["yaw_rate"] - info["target_yaw_rate"])))
        base_heights.append(float(info["base_height"]))
        abs_rolls.append(abs(float(info["roll"])))
        abs_pitches.append(abs(float(info["pitch"])))
        abs_yaws.append(abs(float(info["yaw"])))
        abs_joint_velocities.append(
            float(np.mean(np.abs(env.unwrapped.data.qvel[env.unwrapped.actuator_qvel_ids])))
        )
        action_abs.append(float(np.mean(np.abs(action))))
        ctrl_abs.append(float(np.mean(np.abs(env.unwrapped.data.ctrl))))
        for name, in_contact in env.unwrapped.foot_contacts().items():
            foot_contact_counts[name] += int(in_contact)
        if render:
            time.sleep(env.unwrapped.control_timestep)

    distance = float(env.unwrapped.data.qpos[0] - start_x)
    lateral_drift = float(env.unwrapped.data.qpos[1] - start_y)
    yaw_drift = wrap_to_pi(float(info["yaw"] - start_yaw))
    return EpisodeStats(
        episode=episode,
        total_reward=total_reward,
        length=length,
        distance=distance,
        lateral_drift=lateral_drift,
        yaw_drift=yaw_drift,
        mean_forward_velocity=float(np.mean(forward_velocities)),
        mean_lateral_velocity=float(np.mean(lateral_velocities)),
        mean_world_vx=float(np.mean(world_velocities_x)),
        mean_world_vy=float(np.mean(world_velocities_y)),
        mean_body_vx=float(np.mean(body_velocities_x)),
        mean_body_vy=float(np.mean(body_velocities_y)),
        mean_yaw_rate=float(np.mean(yaw_rates)),
        mean_abs_vx_error=float(np.mean(abs_vx_errors)),
        mean_abs_yaw_rate_error=float(np.mean(abs_yaw_rate_errors)),
        mean_base_height=float(np.mean(base_heights)),
        mean_abs_roll=float(np.mean(abs_rolls)),
        mean_abs_pitch=float(np.mean(abs_pitches)),
        mean_abs_yaw=float(np.mean(abs_yaws)),
        mean_abs_joint_velocity=float(np.mean(abs_joint_velocities)),
        mean_action_abs=float(np.mean(action_abs)),
        mean_ctrl_abs=float(np.mean(ctrl_abs)),
        foot_contact_ratios={
            name: count / length for name, count in foot_contact_counts.items()
        },
        terminated=terminated,
        truncated=truncated,
    )


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def print_episode_stats(stats: EpisodeStats) -> None:
    end_reason = "terminated" if stats.terminated else "truncated"
    contact_ratios = ",".join(
        f"{name}:{ratio:.2f}" for name, ratio in stats.foot_contact_ratios.items()
    )
    print(
        f"episode={stats.episode} "
        f"reward={stats.total_reward:.2f} "
        f"length={stats.length} "
        f"distance={stats.distance:.2f}m "
        f"lateral_drift={stats.lateral_drift:.2f}m "
        f"yaw_drift={stats.yaw_drift:.3f}rad "
        f"mean_vx={stats.mean_forward_velocity:.2f}m/s "
        f"mean_vy={stats.mean_lateral_velocity:.2f}m/s "
        f"world_v=({stats.mean_world_vx:.2f},{stats.mean_world_vy:.2f})m/s "
        f"body_v=({stats.mean_body_vx:.2f},{stats.mean_body_vy:.2f})m/s "
        f"mean_yaw_rate={stats.mean_yaw_rate:.2f}rad/s "
        f"mean_abs_vx_error={stats.mean_abs_vx_error:.2f}m/s "
        f"mean_abs_yaw_rate_error={stats.mean_abs_yaw_rate_error:.2f}rad/s "
        f"mean_height={stats.mean_base_height:.2f}m "
        f"mean_abs_roll={stats.mean_abs_roll:.3f}rad "
        f"mean_abs_pitch={stats.mean_abs_pitch:.3f}rad "
        f"mean_abs_yaw={stats.mean_abs_yaw:.3f}rad "
        f"mean_abs_joint_vel={stats.mean_abs_joint_velocity:.2f}rad/s "
        f"mean_abs_action={stats.mean_action_abs:.3f} "
        f"mean_abs_ctrl={stats.mean_ctrl_abs:.2f} "
        f"foot_contacts=[{contact_ratios}] "
        f"end={end_reason}"
    )


def print_summary(all_stats: list[EpisodeStats]) -> None:
    print("\nSummary")
    print("-------")
    print(f"episodes={len(all_stats)}")
    print(f"mean_reward={np.mean([stats.total_reward for stats in all_stats]):.2f}")
    print(f"mean_length={np.mean([stats.length for stats in all_stats]):.1f}")
    print(f"mean_distance={np.mean([stats.distance for stats in all_stats]):.2f}m")
    print(f"mean_lateral_drift={np.mean([stats.lateral_drift for stats in all_stats]):.2f}m")
    print(f"mean_abs_lateral_drift={np.mean([abs(stats.lateral_drift) for stats in all_stats]):.2f}m")
    print(f"mean_yaw_drift={np.mean([stats.yaw_drift for stats in all_stats]):.3f}rad")
    print(f"mean_abs_yaw_drift={np.mean([abs(stats.yaw_drift) for stats in all_stats]):.3f}rad")
    print(f"mean_forward_velocity={np.mean([stats.mean_forward_velocity for stats in all_stats]):.2f}m/s")
    print(f"mean_lateral_velocity={np.mean([stats.mean_lateral_velocity for stats in all_stats]):.2f}m/s")
    print(f"mean_world_vx={np.mean([stats.mean_world_vx for stats in all_stats]):.2f}m/s")
    print(f"mean_world_vy={np.mean([stats.mean_world_vy for stats in all_stats]):.2f}m/s")
    print(f"mean_body_vx={np.mean([stats.mean_body_vx for stats in all_stats]):.2f}m/s")
    print(f"mean_body_vy={np.mean([stats.mean_body_vy for stats in all_stats]):.2f}m/s")
    print(f"mean_yaw_rate={np.mean([stats.mean_yaw_rate for stats in all_stats]):.2f}rad/s")
    print(f"mean_abs_vx_error={np.mean([stats.mean_abs_vx_error for stats in all_stats]):.2f}m/s")
    print(
        "mean_abs_yaw_rate_error="
        f"{np.mean([stats.mean_abs_yaw_rate_error for stats in all_stats]):.2f}rad/s"
    )
    print(f"mean_base_height={np.mean([stats.mean_base_height for stats in all_stats]):.2f}m")
    print(f"mean_abs_joint_velocity={np.mean([stats.mean_abs_joint_velocity for stats in all_stats]):.2f}rad/s")
    print(f"mean_abs_action={np.mean([stats.mean_action_abs for stats in all_stats]):.3f}")
    print(f"mean_abs_ctrl={np.mean([stats.mean_ctrl_abs for stats in all_stats]):.2f}")
    for name in all_stats[0].foot_contact_ratios:
        mean_ratio = np.mean([stats.foot_contact_ratios[name] for stats in all_stats])
        print(f"mean_contact_ratio_{name}={mean_ratio:.3f}")
    print(f"terminated_count={sum(stats.terminated for stats in all_stats)}")
    print(f"truncated_count={sum(stats.truncated for stats in all_stats)}")


def main() -> None:
    args = parse_args()
    checkpoint = resolve_project_path(args.checkpoint)

    render_mode = None if args.no_render else "human"
    env = gym.make("QuadrupedFlat-v0", config_path=args.config, render_mode=render_mode)
    model = PPO.load(checkpoint, device="cpu")
    if (args.target_vx is None) != (args.target_yaw_rate is None):
        raise ValueError("--target-vx and --target-yaw-rate must be provided together")
    command = (
        (args.target_vx, args.target_yaw_rate)
        if args.target_vx is not None and args.target_yaw_rate is not None
        else None
    )

    try:
        all_stats = []
        for episode in range(1, args.episodes + 1):
            stats = evaluate_episode(
                env,
                model,
                episode=episode,
                render=not args.no_render,
                command=command,
            )
            all_stats.append(stats)
            print_episode_stats(stats)

        print_summary(all_stats)
    finally:
        env.close()


if __name__ == "__main__":
    main()
