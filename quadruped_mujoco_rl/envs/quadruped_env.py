from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from quadruped_mujoco_rl.utils.config import load_yaml
from quadruped_mujoco_rl.utils.paths import resolve_project_path


DEFAULT_CONFIG_PATH = "configs/env_flat.yaml"


class QuadrupedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        render_mode: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = load_yaml(config_path) if config is None else config
        self.render_mode = render_mode

        model_path = resolve_project_path(self.config["model_path"])
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.episode_length = int(self.config.get("episode_length", 1000))
        self.control_timestep = float(self.config.get("control_timestep", 0.02))
        self.frame_skip = max(1, round(self.control_timestep / self.model.opt.timestep))
        self.action_config = self.config.get("action", {})
        self.action_type = str(self.action_config.get("type", "joint_position"))
        self.action_scale = float(self.action_config.get("scale", 1.0))
        self.default_joint_qpos = np.asarray(
            self.config.get("default_joint_qpos", np.zeros(self.model.nu)),
            dtype=np.float64,
        )
        if self.default_joint_qpos.shape != (self.model.nu,):
            raise ValueError(
                f"default_joint_qpos must have shape ({self.model.nu},), "
                f"got {self.default_joint_qpos.shape}"
            )
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.actuator_qpos_ids, self.actuator_qvel_ids, self.actuator_joint_ids = (
            self._build_actuator_joint_maps()
        )
        self.joint_low = self.model.jnt_range[self.actuator_joint_ids, 0].copy()
        self.joint_high = self.model.jnt_range[self.actuator_joint_ids, 1].copy()
        self.kp = float(self.action_config.get("kp", 30.0))
        self.kd = float(self.action_config.get("kd", 1.0))
        self.reward_config = self.config.get("reward", {})
        self.termination_config = self.config.get("termination", {})

        self.step_count = 0
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)
        self._viewer = None
        self._renderer = None

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._observation_size(),),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        keyframe_name = self.config.get("initial_keyframe")
        if keyframe_name:
            keyframe_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_KEY,
                str(keyframe_name),
            )
            if keyframe_id < 0:
                raise ValueError(f"Unknown keyframe: {keyframe_name}")
            mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[2] = float(self.config.get("initial_base_height", 0.32))
            self.data.qpos[3] = 1.0

        if options and "qpos" in options:
            self.data.qpos[:] = np.asarray(options["qpos"], dtype=np.float64)
        if options and "qvel" in options:
            self.data.qvel[:] = np.asarray(options["qvel"], dtype=np.float64)

        self.data.qpos[7:] = self.default_joint_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self._action_to_ctrl(np.zeros(self.model.nu, dtype=np.float32))
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.last_action.fill(0.0)

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        x_before = float(self.data.qpos[0])
        self.data.ctrl[:] = self._action_to_ctrl(action)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        x_after = float(self.data.qpos[0])
        self.step_count += 1

        forward_velocity = (x_after - x_before) / (self.frame_skip * self.model.opt.timestep)
        terminated = self._is_terminated()
        truncated = self.step_count >= self.episode_length
        reward = self._compute_reward(action, forward_velocity, terminated)
        self.last_action = action.copy()

        info = self._get_info()
        info["forward_velocity"] = forward_velocity

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

        if self.render_mode == "human":
            from mujoco import viewer

            if self._viewer is None:
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None

        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.data.qpos.astype(np.float32),
                self.data.qvel.astype(np.float32),
                self.last_action.astype(np.float32),
            ]
        )

    def _observation_size(self) -> int:
        return self.model.nq + self.model.nv + self.model.nu

    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        target_qpos = self.default_joint_qpos + action * self.action_scale
        target_qpos = np.clip(target_qpos, self.joint_low, self.joint_high)

        if self.action_type == "pd_position":
            joint_qpos = self.data.qpos[self.actuator_qpos_ids]
            joint_qvel = self.data.qvel[self.actuator_qvel_ids]
            torque = self.kp * (target_qpos - joint_qpos) - self.kd * joint_qvel
            return np.clip(torque, self.ctrl_low, self.ctrl_high)

        if self.action_type == "joint_position":
            return np.clip(target_qpos, self.ctrl_low, self.ctrl_high)

        raise ValueError(f"Unsupported action type: {self.action_type}")

    def _build_actuator_joint_maps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        joint_ids = []
        qpos_ids = []
        qvel_ids = []

        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            joint_ids.append(joint_id)
            qpos_ids.append(int(self.model.jnt_qposadr[joint_id]))
            qvel_ids.append(int(self.model.jnt_dofadr[joint_id]))

        return (
            np.asarray(qpos_ids, dtype=np.int32),
            np.asarray(qvel_ids, dtype=np.int32),
            np.asarray(joint_ids, dtype=np.int32),
        )

    def _compute_reward(
        self,
        action: np.ndarray,
        forward_velocity: float,
        terminated: bool,
    ) -> float:
        upright = float(self.data.xmat[1, 8])
        energy = float(np.sum(np.square(action)))
        smoothness = float(np.sum(np.square(action - self.last_action)))

        reward = 0.0
        reward += float(self.reward_config.get("forward_velocity", 1.0)) * forward_velocity
        reward += float(self.reward_config.get("upright", 0.0)) * upright
        reward -= float(self.reward_config.get("energy_penalty", 0.0)) * energy
        reward -= float(self.reward_config.get("action_smoothness", 0.0)) * smoothness
        if terminated:
            reward -= float(self.reward_config.get("fall_penalty", 0.0))
        return float(reward)

    def _is_terminated(self) -> bool:
        base_height = float(self.data.qpos[2])
        min_base_height = float(self.termination_config.get("min_base_height", 0.18))
        if base_height < min_base_height:
            return True

        roll, pitch, _ = self._base_euler_xyz()
        max_roll_pitch = float(self.termination_config.get("max_roll_pitch", 0.9))
        return bool(abs(roll) > max_roll_pitch or abs(pitch) > max_roll_pitch)

    def _base_euler_xyz(self) -> np.ndarray:
        quat = self.data.qpos[3:7]
        mat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, quat)
        return self._mat_to_euler_xyz(mat.reshape(3, 3))

    @staticmethod
    def _mat_to_euler_xyz(mat: np.ndarray) -> np.ndarray:
        pitch = np.arcsin(np.clip(mat[0, 2], -1.0, 1.0))
        roll = np.arctan2(-mat[1, 2], mat[2, 2])
        yaw = np.arctan2(-mat[0, 1], mat[0, 0])
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def _get_info(self) -> dict[str, float | int]:
        roll, pitch, yaw = self._base_euler_xyz()
        return {
            "step_count": self.step_count,
            "base_height": float(self.data.qpos[2]),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
        }
