from gymnasium.envs.registration import register


def register_envs() -> None:
    register(
        id="QuadrupedFlat-v0",
        entry_point="quadruped_mujoco_rl.envs.quadruped_env:QuadrupedEnv",
    )
