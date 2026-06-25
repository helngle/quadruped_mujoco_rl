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
        self.velocity_frame = str(self.config.get("velocity_frame", "world"))
        if self.velocity_frame not in {"world", "body"}:
            raise ValueError("velocity_frame must be 'world' or 'body'")
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
        self.reward_type = str(self.reward_config.get("type", "legacy"))
        if self.reward_type not in {"legacy", "canonical"}:
            raise ValueError("reward.type must be 'legacy' or 'canonical'")
        self.observation_config = self.config.get("observation", {})
        self.observation_type = str(self.observation_config.get("type", "legacy"))
        if self.observation_type not in {"legacy", "locomotion"}:
            raise ValueError("observation.type must be 'legacy' or 'locomotion'")
        self.termination_config = self.config.get("termination", {})
        self.gait_config = self.config.get("gait", {})
        self.use_gait_phase = bool(self.gait_config.get("enabled", False))
        self.command_config = self.config.get("command", {})
        self.use_commands = bool(self.command_config.get("enabled", False))
        self.target_vx = float(self.reward_config.get("target_forward_velocity", 0.0))
        self.target_yaw_rate = 0.0
        self.command_is_fixed = False
        self.foot_geom_names = list(self.config.get("foot_geom_names", []))
        self.foot_geom_ids = self._resolve_geom_ids(self.foot_geom_names)
        base_body_name = str(self.config.get("base_body_name", "base"))
        self.base_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            base_body_name,
        )
        if self.base_body_id < 0:
            raise ValueError(f"Unknown base body: {base_body_name}")
        ground_geom_name = self.config.get("ground_geom_name")
        self.ground_geom_id = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, ground_geom_name)
            if ground_geom_name
            else -1
        )
        if ground_geom_name and self.ground_geom_id < 0:
            raise ValueError(f"Unknown ground geom: {ground_geom_name}")

        self.step_count = 0
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)
        self.foot_air_time = np.zeros(len(self.foot_geom_ids), dtype=np.float64)
        self.previous_foot_contacts = np.zeros(len(self.foot_geom_ids), dtype=bool)
        self.previous_joint_velocity = np.zeros(self.model.nu, dtype=np.float64)
        self.last_reward_terms: dict[str, float] = {}
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
        self.foot_air_time.fill(0.0)
        self.previous_foot_contacts = self._foot_contact_array()
        self.previous_joint_velocity = self.data.qvel[self.actuator_qvel_ids].copy()
        self.last_reward_terms = {}
        command = options.get("command") if options else None
        if command is not None:
            self._set_command(command)
            self.command_is_fixed = True
        elif self.use_commands:
            self._sample_command()
            self.command_is_fixed = False

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        x_before = float(self.data.qpos[0])
        y_before = float(self.data.qpos[1])
        yaw_before = float(self._base_euler_xyz()[2])
        self.data.ctrl[:] = self._action_to_ctrl(action)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        x_after = float(self.data.qpos[0])
        y_after = float(self.data.qpos[1])
        yaw_after = float(self._base_euler_xyz()[2])
        self.step_count += 1

        elapsed = self.frame_skip * self.model.opt.timestep
        world_vx = (x_after - x_before) / elapsed
        world_vy = (y_after - y_before) / elapsed
        cos_yaw = float(np.cos(yaw_after))
        sin_yaw = float(np.sin(yaw_after))
        body_vx = cos_yaw * world_vx + sin_yaw * world_vy
        body_vy = -sin_yaw * world_vx + cos_yaw * world_vy
        if self.velocity_frame == "body":
            forward_velocity = body_vx
            lateral_velocity = body_vy
        else:
            forward_velocity = world_vx
            lateral_velocity = world_vy
        yaw_rate = self._wrap_to_pi(yaw_after - yaw_before) / elapsed
        terminated = self._is_terminated()
        truncated = self.step_count >= self.episode_length
        reward = self._compute_reward(
            action=action,
            forward_velocity=forward_velocity,
            lateral_velocity=lateral_velocity,
            yaw_rate=yaw_rate,
            terminated=terminated,
        )
        self.previous_joint_velocity = self.data.qvel[self.actuator_qvel_ids].copy()
        self.last_action = action.copy()

        info = self._get_info()
        info["forward_velocity"] = forward_velocity
        info["lateral_velocity"] = lateral_velocity
        info["yaw_rate"] = yaw_rate
        info["world_vx"] = world_vx
        info["world_vy"] = world_vy
        info["body_vx"] = body_vx
        info["body_vy"] = body_vy
        info["target_vx"] = self.target_vx
        info["target_yaw_rate"] = self.target_yaw_rate
        info["reward_terms"] = self.last_reward_terms.copy()

        if self._should_resample_command():
            self._sample_command()

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
        if self.observation_type == "locomotion":
            scales = self.observation_config.get("scales", {})
            body_linear_velocity = self._body_linear_velocity()
            body_angular_velocity = self._body_angular_velocity()
            projected_gravity = self._projected_gravity()
            command = np.asarray([self.target_vx, self.target_yaw_rate], dtype=np.float64)
            joint_position_error = (
                self.data.qpos[self.actuator_qpos_ids] - self.default_joint_qpos
            )
            joint_velocity = self.data.qvel[self.actuator_qvel_ids]
            return np.concatenate(
                [
                    body_linear_velocity * float(scales.get("linear_velocity", 2.0)),
                    body_angular_velocity * float(scales.get("angular_velocity", 0.25)),
                    projected_gravity,
                    command
                    * np.asarray(
                        [
                            float(scales.get("command_velocity", 2.0)),
                            float(scales.get("command_yaw_rate", 0.25)),
                        ]
                    ),
                    joint_position_error * float(scales.get("joint_position", 1.0)),
                    joint_velocity * float(scales.get("joint_velocity", 0.05)),
                    self.last_action,
                ]
            ).astype(np.float32)

        parts = [
            self.data.qpos.astype(np.float32),
            self.data.qvel.astype(np.float32),
            self.last_action.astype(np.float32),
        ]
        if self.use_gait_phase:
            parts.append(self._gait_phase_features())
        if self.use_commands:
            parts.append(np.asarray([self.target_vx, self.target_yaw_rate], dtype=np.float32))
        return np.concatenate(parts)

    def _observation_size(self) -> int:
        if self.observation_type == "locomotion":
            return 3 + 3 + 3 + 2 + self.model.nu * 3
        gait_phase_size = 2 if self.use_gait_phase else 0
        command_size = 2 if self.use_commands else 0
        return self.model.nq + self.model.nv + self.model.nu + gait_phase_size + command_size

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
        lateral_velocity: float,
        yaw_rate: float,
        terminated: bool,
    ) -> float:
        if self.reward_type == "canonical":
            return self._compute_canonical_reward(
                action=action,
                forward_velocity=forward_velocity,
                lateral_velocity=lateral_velocity,
                yaw_rate=yaw_rate,
                terminated=terminated,
            )

        self.last_reward_terms = {}
        upright = float(self.data.xmat[1, 8])
        energy = float(np.sum(np.square(action)))
        smoothness = float(np.sum(np.square(action - self.last_action)))
        joint_velocity = float(np.mean(np.square(self.data.qvel[self.actuator_qvel_ids])))
        joint_posture = float(
            np.mean(
                np.square(
                    self.data.qpos[self.actuator_qpos_ids] - self.default_joint_qpos
                )
            )
        )
        body_angular_velocity = float(np.mean(np.square(self.data.qvel[3:6])))
        _, pitch, yaw = self._base_euler_xyz()
        yaw_target = float(self.reward_config.get("yaw_heading_target", 0.0))
        yaw_error = self._wrap_to_pi(yaw - yaw_target)
        pitch_target = float(self.reward_config.get("pitch_target", 0.0))
        pitch_error = self._wrap_to_pi(float(pitch) - pitch_target)
        foot_slip = self._foot_slip_cost()
        ctrl_effort = float(np.mean(np.square(self.data.ctrl)))
        base_height_target = self.reward_config.get("base_height_target")
        base_height_error = 0.0
        if base_height_target is not None:
            base_height_error = float((self.data.qpos[2] - float(base_height_target)) ** 2)

        base_height_tracking = 0.0
        if base_height_target is not None:
            height_sigma = max(
                float(self.reward_config.get("base_height_tracking_sigma", 0.04)), 1e-6
            )
            normalized_height_error = (
                self.data.qpos[2] - float(base_height_target)
            ) / height_sigma
            base_height_tracking = float(np.exp(-0.5 * normalized_height_error**2))

        velocity_tracking = 0.0
        forward_velocity_error = 0.0
        target_forward_velocity = (
            self.target_vx
            if self.use_commands
            else self.reward_config.get("target_forward_velocity")
        )
        if target_forward_velocity is not None:
            sigma = max(float(self.reward_config.get("velocity_tracking_sigma", 0.5)), 1e-6)
            forward_velocity_error = forward_velocity - float(target_forward_velocity)
            normalized_velocity_error = forward_velocity_error / sigma
            velocity_tracking = float(np.exp(-0.5 * normalized_velocity_error**2))

        yaw_rate_tracking = 0.0
        if self.use_commands:
            yaw_sigma = max(
                float(self.reward_config.get("yaw_rate_tracking_sigma", 0.4)), 1e-6
            )
            normalized_yaw_rate_error = (yaw_rate - self.target_yaw_rate) / yaw_sigma
            yaw_rate_tracking = float(np.exp(-0.5 * normalized_yaw_rate_error**2))

        command_stillness = 0.0
        if self.use_commands and abs(self.target_vx) < 0.05 and abs(self.target_yaw_rate) < 0.05:
            stillness_sigma = max(
                float(self.reward_config.get("command_stillness_sigma", 0.15)), 1e-6
            )
            stillness_error = (
                forward_velocity**2 + lateral_velocity**2 + yaw_rate**2
            ) / stillness_sigma**2
            command_stillness = float(np.exp(-0.5 * stillness_error))

        reward = 0.0
        reward += float(self.reward_config.get("forward_velocity", 1.0)) * forward_velocity
        reward += (
            float(self.reward_config.get("forward_velocity_tracking", 0.0))
            * velocity_tracking
        )
        reward += float(self.reward_config.get("yaw_rate_tracking", 0.0)) * yaw_rate_tracking
        reward += (
            float(self.reward_config.get("command_tracking_reward", 0.0))
            * velocity_tracking
            * yaw_rate_tracking
        )
        reward -= (
            float(self.reward_config.get("forward_velocity_error_penalty", 0.0))
            * abs(forward_velocity_error)
        )
        reward += (
            float(self.reward_config.get("command_stillness_reward", 0.0))
            * command_stillness
        )
        reward += float(self.reward_config.get("upright", 0.0)) * upright
        reward += (
            float(self.reward_config.get("base_height_tracking", 0.0))
            * base_height_tracking
        )
        reward -= float(self.reward_config.get("energy_penalty", 0.0)) * energy
        reward -= float(self.reward_config.get("action_smoothness", 0.0)) * smoothness
        reward -= float(self.reward_config.get("lateral_velocity_penalty", 0.0)) * abs(
            lateral_velocity
        )
        reward -= float(self.reward_config.get("yaw_rate_penalty", 0.0)) * abs(yaw_rate)
        reward -= float(self.reward_config.get("joint_velocity_penalty", 0.0)) * joint_velocity
        reward -= float(self.reward_config.get("joint_posture_penalty", 0.0)) * joint_posture
        reward -= (
            float(self.reward_config.get("body_angular_velocity_penalty", 0.0))
            * body_angular_velocity
        )
        reward -= float(self.reward_config.get("yaw_heading_penalty", 0.0)) * yaw_error**2
        reward -= float(self.reward_config.get("pitch_penalty", 0.0)) * pitch_error**2
        reward -= float(self.reward_config.get("foot_slip_penalty", 0.0)) * foot_slip
        reward -= float(self.reward_config.get("ctrl_effort_penalty", 0.0)) * ctrl_effort
        movement_command = abs(self.target_vx) + abs(self.target_yaw_rate)
        gait_scale = float(movement_command > 0.1)
        reward += (
            float(self.reward_config.get("gait_contact_reward", 0.0))
            * self._gait_contact_score()
            * gait_scale
        )
        reward -= float(self.reward_config.get("base_height_penalty", 0.0)) * base_height_error
        if terminated:
            reward -= float(self.reward_config.get("fall_penalty", 0.0))
        return float(reward)

    def _compute_canonical_reward(
        self,
        action: np.ndarray,
        forward_velocity: float,
        lateral_velocity: float,
        yaw_rate: float,
        terminated: bool,
    ) -> float:
        linear_sigma = max(
            float(self.reward_config.get("linear_velocity_tracking_sigma", 0.5)),
            1e-6,
        )
        linear_error = (forward_velocity - self.target_vx) ** 2 + lateral_velocity**2
        linear_tracking = float(np.exp(-linear_error / linear_sigma**2))

        yaw_sigma = max(
            float(self.reward_config.get("yaw_rate_tracking_sigma", 0.5)),
            1e-6,
        )
        yaw_error = (yaw_rate - self.target_yaw_rate) ** 2
        yaw_tracking = float(np.exp(-yaw_error / yaw_sigma**2))

        body_linear_velocity = self._body_linear_velocity()
        body_angular_velocity = self._body_angular_velocity()
        projected_gravity = self._projected_gravity()
        vertical_velocity = float(body_linear_velocity[2] ** 2)
        angular_velocity_xy = float(np.sum(np.square(body_angular_velocity[:2])))
        flat_orientation = float(np.sum(np.square(projected_gravity[:2])))
        torque = float(np.sum(np.square(self.data.ctrl)))
        action_rate = float(np.sum(np.square(action - self.last_action)))
        joint_velocity = self.data.qvel[self.actuator_qvel_ids]
        joint_acceleration = float(
            np.sum(
                np.square(
                    (joint_velocity - self.previous_joint_velocity) / self.control_timestep
                )
            )
        )
        joint_posture = float(
            np.sum(
                np.square(
                    self.data.qpos[self.actuator_qpos_ids] - self.default_joint_qpos
                )
            )
        )
        foot_slip = self._foot_slip_cost()
        foot_clearance = self._foot_clearance_cost()
        feet_air_time = self._feet_air_time_reward()
        undesired_contacts = float(self._undesired_ground_contact_count())
        base_height_target = float(
            self.reward_config.get("base_height_target", self.config.get("initial_base_height", 0.27))
        )
        base_height_error = float((self.data.qpos[2] - base_height_target) ** 2)

        weighted_terms = {
            "linear_velocity_tracking": float(
                self.reward_config.get("linear_velocity_tracking", 1.5)
            )
            * linear_tracking,
            "yaw_rate_tracking": float(self.reward_config.get("yaw_rate_tracking", 0.75))
            * yaw_tracking,
            "vertical_velocity": -float(
                self.reward_config.get("vertical_velocity_penalty", 2.0)
            )
            * vertical_velocity,
            "angular_velocity_xy": -float(
                self.reward_config.get("angular_velocity_xy_penalty", 0.05)
            )
            * angular_velocity_xy,
            "flat_orientation": -float(
                self.reward_config.get("flat_orientation_penalty", 1.0)
            )
            * flat_orientation,
            "joint_torque": -float(self.reward_config.get("joint_torque_penalty", 2e-5))
            * torque,
            "action_rate": -float(self.reward_config.get("action_rate_penalty", 0.01))
            * action_rate,
            "joint_acceleration": -float(
                self.reward_config.get("joint_acceleration_penalty", 0.0)
            )
            * joint_acceleration,
            "joint_posture": -float(
                self.reward_config.get("joint_posture_penalty", 0.0)
            )
            * joint_posture,
            "foot_slip": -float(self.reward_config.get("foot_slip_penalty", 0.1))
            * foot_slip,
            "foot_clearance": -float(
                self.reward_config.get("foot_clearance_penalty", 0.0)
            )
            * foot_clearance,
            "feet_air_time": float(self.reward_config.get("feet_air_time_reward", 0.1))
            * feet_air_time,
            "undesired_contacts": -float(
                self.reward_config.get("undesired_contact_penalty", 1.0)
            )
            * undesired_contacts,
            "base_height": -float(self.reward_config.get("base_height_penalty", 0.0))
            * base_height_error,
            "termination": -float(self.reward_config.get("fall_penalty", 20.0))
            * float(terminated),
        }
        self.last_reward_terms = weighted_terms
        return float(sum(weighted_terms.values()))

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

    def _base_rotation_matrix(self) -> np.ndarray:
        return self.data.xmat[self.base_body_id].reshape(3, 3)

    def _body_linear_velocity(self) -> np.ndarray:
        return self._base_rotation_matrix().T @ self.data.qvel[:3]

    def _body_angular_velocity(self) -> np.ndarray:
        return self.data.qvel[3:6].copy()

    def _projected_gravity(self) -> np.ndarray:
        gravity_world = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        return self._base_rotation_matrix().T @ gravity_world

    @staticmethod
    def _mat_to_euler_xyz(mat: np.ndarray) -> np.ndarray:
        pitch = np.arcsin(np.clip(mat[0, 2], -1.0, 1.0))
        roll = np.arctan2(-mat[1, 2], mat[2, 2])
        yaw = np.arctan2(-mat[0, 1], mat[0, 0])
        return np.array([roll, pitch, yaw], dtype=np.float32)

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _resolve_geom_ids(self, geom_names: list[str]) -> np.ndarray:
        geom_ids = []
        for name in geom_names:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id < 0:
                raise ValueError(f"Unknown geom: {name}")
            geom_ids.append(geom_id)
        return np.asarray(geom_ids, dtype=np.int32)

    def _foot_slip_cost(self) -> float:
        if self.foot_geom_ids.size == 0:
            return 0.0

        contacting_feet = self._contacting_foot_ids()
        if not contacting_feet:
            return 0.0

        slip_costs = []
        for geom_id in contacting_feet:
            velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_id,
                velocity,
                0,
            )
            slip_costs.append(float(np.dot(velocity[3:5], velocity[3:5])))
        return float(np.mean(slip_costs))

    def _foot_clearance_cost(self) -> float:
        if self.foot_geom_ids.size == 0:
            return 0.0

        contacts = self._foot_contact_array()
        target_height = float(self.reward_config.get("foot_clearance_target", 0.06))
        costs = []
        for index, geom_id in enumerate(self.foot_geom_ids):
            if contacts[index]:
                continue
            velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_GEOM,
                int(geom_id),
                velocity,
                0,
            )
            horizontal_speed = float(np.linalg.norm(velocity[3:5]))
            height_error = float(self.data.geom_xpos[int(geom_id), 2] - target_height)
            costs.append(height_error**2 * np.sqrt(horizontal_speed))
        return float(np.sum(costs))

    def foot_contacts(self) -> dict[str, bool]:
        contacting_feet = self._contacting_foot_ids()
        return {
            name: int(geom_id) in contacting_feet
            for name, geom_id in zip(self.foot_geom_names, self.foot_geom_ids, strict=True)
        }

    def _foot_contact_array(self) -> np.ndarray:
        contacting_feet = self._contacting_foot_ids()
        return np.asarray(
            [int(geom_id) in contacting_feet for geom_id in self.foot_geom_ids],
            dtype=bool,
        )

    def _feet_air_time_reward(self) -> float:
        if self.foot_geom_ids.size == 0:
            return 0.0

        contacts = self._foot_contact_array()
        self.foot_air_time += self.control_timestep
        first_contacts = contacts & ~self.previous_foot_contacts
        threshold = float(self.reward_config.get("feet_air_time_threshold", 0.15))
        maximum = max(
            float(self.reward_config.get("feet_air_time_max", 0.5)),
            threshold,
        )
        rewarded_air_time = np.clip(
            self.foot_air_time - threshold,
            0.0,
            maximum - threshold,
        )
        reward = float(np.sum(rewarded_air_time * first_contacts))
        self.foot_air_time[contacts] = 0.0
        self.previous_foot_contacts = contacts
        movement_command = abs(self.target_vx) + abs(self.target_yaw_rate)
        return reward * float(movement_command > 0.1)

    def _undesired_ground_contact_count(self) -> int:
        if self.ground_geom_id < 0:
            return 0

        foot_ids = set(self.foot_geom_ids.tolist())
        undesired_geom_ids: set[int] = set()
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == self.ground_geom_id and geom2 not in foot_ids:
                undesired_geom_ids.add(geom2)
            elif geom2 == self.ground_geom_id and geom1 not in foot_ids:
                undesired_geom_ids.add(geom1)
        undesired_geom_ids.discard(self.ground_geom_id)
        return len(undesired_geom_ids)

    def _contacting_foot_ids(self) -> set[int]:
        foot_ids = set(self.foot_geom_ids.tolist())
        contacting_feet: set[int] = set()
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in foot_ids and (self.ground_geom_id < 0 or geom2 == self.ground_geom_id):
                contacting_feet.add(geom1)
            if geom2 in foot_ids and (self.ground_geom_id < 0 or geom1 == self.ground_geom_id):
                contacting_feet.add(geom2)
        return contacting_feet

    def _gait_phase_features(self) -> np.ndarray:
        phase = 2 * np.pi * self._gait_phase_fraction()
        return np.asarray([np.sin(phase), np.cos(phase)], dtype=np.float32)

    def _gait_phase_fraction(self) -> float:
        period = max(float(self.gait_config.get("period", 0.5)), self.control_timestep)
        return float((self.step_count * self.control_timestep / period) % 1.0)

    def _gait_contact_score(self) -> float:
        if not self.use_gait_phase or len(self.foot_geom_names) != 4:
            return 0.0

        contacts = self.foot_contacts()
        if not {"FL", "FR", "RL", "RR"}.issubset(contacts):
            return 0.0

        if self._gait_phase_fraction() < 0.5:
            expected = {"FL": True, "FR": False, "RL": False, "RR": True}
        else:
            expected = {"FL": False, "FR": True, "RL": True, "RR": False}

        matches = sum(contacts[name] == expected[name] for name in expected)
        return matches / len(expected)

    def _set_command(self, command: Any) -> None:
        command_array = np.asarray(command, dtype=np.float64)
        if command_array.shape != (2,):
            raise ValueError("command must contain [target_vx, target_yaw_rate]")
        self.target_vx = float(command_array[0])
        self.target_yaw_rate = float(command_array[1])

    def _sample_command(self) -> None:
        stand_probability = float(self.command_config.get("stand_probability", 0.15))
        turn_probability = float(self.command_config.get("turn_probability", 0.15))
        straight_probability = float(self.command_config.get("straight_probability", 0.25))
        sample = float(self.np_random.random())

        if sample < stand_probability:
            self.target_vx = 0.0
            self.target_yaw_rate = 0.0
            return

        yaw_range = self.command_config.get("yaw_rate_range", [-0.8, 0.8])
        min_abs_yaw_rate = float(self.command_config.get("min_abs_yaw_rate", 0.0))
        if sample < stand_probability + turn_probability:
            self.target_vx = 0.0
            self.target_yaw_rate = self._sample_yaw_rate(yaw_range, min_abs_yaw_rate)
            return

        vx_range = self.command_config.get("vx_range", [0.3, 1.2])
        self.target_vx = float(self.np_random.uniform(*vx_range))
        if sample < stand_probability + turn_probability + straight_probability:
            self.target_yaw_rate = 0.0
        else:
            self.target_yaw_rate = self._sample_yaw_rate(yaw_range, min_abs_yaw_rate)

    def _sample_yaw_rate(self, yaw_range: list[float], min_abs_yaw_rate: float) -> float:
        for _ in range(10):
            yaw_rate = float(self.np_random.uniform(*yaw_range))
            if abs(yaw_rate) >= min_abs_yaw_rate:
                return yaw_rate
        sign = -1.0 if float(self.np_random.random()) < 0.5 else 1.0
        return sign * min_abs_yaw_rate

    def _should_resample_command(self) -> bool:
        if not self.use_commands or self.command_is_fixed:
            return False
        resample_time = float(self.command_config.get("resample_time", 4.0))
        resample_steps = max(1, round(resample_time / self.control_timestep))
        return self.step_count > 0 and self.step_count % resample_steps == 0

    def _get_info(self) -> dict[str, float | int]:
        roll, pitch, yaw = self._base_euler_xyz()
        return {
            "step_count": self.step_count,
            "base_height": float(self.data.qpos[2]),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
        }
