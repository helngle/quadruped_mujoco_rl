import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3 import PPO

import quadruped_mujoco_rl  # noqa: F401
from quadruped_mujoco_rl.envs.quadruped_env import QuadrupedEnv
from quadruped_mujoco_rl.training.train_ppo import transfer_policy_weights


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


def test_go2_quality_config_resets_and_steps() -> None:
    env = QuadrupedEnv(config_path="configs/env_go2_quality.yaml")
    obs, _ = env.reset(seed=123)

    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(reward)
        assert "lateral_velocity" in info
        assert "yaw_rate" in info
        assert set(env.foot_contacts()) == {"FL", "FR", "RL", "RR"}
        assert not truncated
        if terminated:
            break

    env.close()


def test_go2_quality_v2_adds_gait_phase_observation() -> None:
    env = QuadrupedEnv(config_path="configs/env_go2_quality_v2.yaml")
    obs, _ = env.reset(seed=123)

    assert obs.shape == (51,)
    assert env.observation_space.shape == (51,)
    assert np.allclose(obs[-2:], [0.0, 1.0])

    obs, reward, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert obs.shape == (51,)
    assert np.isfinite(reward)
    env.close()


def test_go2_command_config_observes_fixed_command() -> None:
    env = QuadrupedEnv(config_path="configs/env_go2_command.yaml")
    obs, _ = env.reset(seed=123, options={"command": [0.8, -0.3]})

    assert obs.shape == (53,)
    assert env.observation_space.shape == (53,)
    assert np.allclose(obs[-2:], [0.8, -0.3])

    obs, reward, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert obs.shape == (53,)
    assert np.isfinite(reward)
    assert info["target_vx"] == 0.8
    assert info["target_yaw_rate"] == -0.3
    env.close()


def test_go2_command_v2_uses_body_velocity_frame() -> None:
    env = QuadrupedEnv(config_path="configs/env_go2_command_v2.yaml")
    obs, _ = env.reset(seed=123, options={"command": [0.7, 0.4]})

    assert env.velocity_frame == "body"
    assert obs.shape == (53,)
    obs, reward, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert np.isfinite(reward)
    assert info["forward_velocity"] == info["body_vx"]
    assert info["lateral_velocity"] == info["body_vy"]
    assert info["target_vx"] == 0.7
    assert info["target_yaw_rate"] == 0.4
    env.close()


def test_go2_command_v3_phase1_samples_only_straight_motion() -> None:
    env = QuadrupedEnv(config_path="configs/env_go2_command_v3_phase1.yaml")

    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        assert 0.5 <= obs[-2] <= 1.0
        assert obs[-1] == 0.0

    env.close()


def test_policy_transfer_expands_command_inputs_with_zero_weights() -> None:
    source_env = QuadrupedEnv(config_path="configs/env_go2_quality_v2.yaml")
    target_env = QuadrupedEnv(config_path="configs/env_go2_command_v3_phase1.yaml")
    source = PPO("MlpPolicy", source_env, n_steps=8, batch_size=8, seed=1, device="cpu")
    target = PPO("MlpPolicy", target_env, n_steps=8, batch_size=8, seed=2, device="cpu")

    source_state = source.policy.state_dict()
    source_state["mlp_extractor.policy_net.0.weight"].fill_(0.125)
    source.policy.load_state_dict(source_state)

    copied, expanded, skipped = transfer_policy_weights(source, target)
    target_weight = target.policy.state_dict()["mlp_extractor.policy_net.0.weight"]

    assert copied > 0
    assert expanded == 2
    assert skipped == []
    assert np.allclose(target_weight[:, :51].cpu().numpy(), 0.125)
    assert np.allclose(target_weight[:, 51:].cpu().numpy(), 0.0)
    source_env.close()
    target_env.close()


def test_go2_command_v4_uses_canonical_locomotion_observation_and_reward() -> None:
    env = QuadrupedEnv(config_path="configs/env_go2_command_v4_canonical.yaml")
    obs, _ = env.reset(seed=123, options={"command": [0.75, 0.0]})

    assert env.observation_type == "locomotion"
    assert env.reward_type == "canonical"
    assert obs.shape == (47,)
    assert np.allclose(obs[6:9], [0.0, 0.0, -1.0])
    assert np.allclose(obs[9:11], [1.5, 0.0])

    env.data.qpos[0] += 10.0
    mujoco.mj_forward(env.model, env.data)
    assert np.allclose(env._get_obs(), obs)

    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert np.isfinite(reward)
    assert set(info["reward_terms"]) == {
        "linear_velocity_tracking",
        "yaw_rate_tracking",
        "vertical_velocity",
        "angular_velocity_xy",
        "flat_orientation",
        "joint_torque",
        "action_rate",
        "joint_acceleration",
        "joint_posture",
        "foot_slip",
        "foot_clearance",
        "feet_air_time",
        "undesired_contacts",
        "base_height",
        "termination",
    }
    env.close()


def test_go2_command_v4_1_preserves_observation_and_enables_gait_costs() -> None:
    env = QuadrupedEnv(config_path="configs/env_go2_command_v4_1_gait.yaml")
    obs, _ = env.reset(seed=123, options={"command": [0.75, 0.0]})

    assert obs.shape == (47,)
    assert env.reward_config["feet_air_time_threshold"] == 0.05
    assert env.reward_config["foot_slip_penalty"] == 0.5
    assert env.reward_config["foot_clearance_penalty"] == 5.0

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, reward, _, _, info = env.step(action)
    assert np.isfinite(reward)
    assert np.isfinite(info["reward_terms"]["joint_acceleration"])
    assert np.isfinite(info["reward_terms"]["foot_clearance"])
    env.close()
