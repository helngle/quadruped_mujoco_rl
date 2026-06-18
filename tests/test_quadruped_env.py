import gymnasium as gym
import numpy as np

import quadruped_mujoco_rl  # noqa: F401
from quadruped_mujoco_rl.envs.quadruped_env import QuadrupedEnv


def test_env_reset_returns_observation() -> None:
    env = QuadrupedEnv()

    obs, info = env.reset(seed=123)

    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert info["base_height"] > 0.0
    env.close()


def test_env_steps_with_random_actions() -> None:
    env = QuadrupedEnv()
    obs, _ = env.reset(seed=123)

    for _ in range(25):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "forward_velocity" in info
        if terminated or truncated:
            break

    env.close()


def test_env_can_be_created_from_gym_registry() -> None:
    env = gym.make("QuadrupedFlat-v0")
    obs, _ = env.reset(seed=123)

    assert obs.shape == env.observation_space.shape
    env.close()


def test_go2_config_resets_and_steps() -> None:
    env = QuadrupedEnv(config_path="configs/env_go2.yaml")
    obs, info = env.reset(seed=123)

    assert obs.shape == env.observation_space.shape
    assert info["base_height"] > 0.0

    for _ in range(10):
        obs, reward, terminated, truncated, _ = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(reward)
        assert not truncated
        if terminated:
            break

    env.close()
