import time
import argparse

from quadruped_mujoco_rl.envs.quadruped_env import QuadrupedEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch random actions in the quadruped env.")
    parser.add_argument("--config", default="configs/env_flat.yaml", help="Environment YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = QuadrupedEnv(config_path=args.config, render_mode="human")
    env.reset(seed=0)

    try:
        while True:
            for _ in range(100):
                _, _, terminated, truncated, _ = env.step(env.action_space.low * 0.0)
                time.sleep(env.control_timestep)

                if terminated or truncated:
                    break

            for _ in range(200):
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                time.sleep(env.control_timestep)

                if terminated or truncated:
                    break

            env.reset()
            time.sleep(0.5)
    finally:
        env.close()


if __name__ == "__main__":
    main()
