# Quadruped MuJoCo RL

This project is for training quadruped locomotion policies in MuJoCo with reinforcement learning.

The experiment history, failures, decisions, and current status are maintained in
[`docs/PROJECT_JOURNAL.md`](docs/PROJECT_JOURNAL.md).

The first milestone is a minimal flat-ground pipeline:

1. Load a quadruped MuJoCo model.
2. Wrap it as a Gymnasium environment.
3. Train a baseline PPO policy.
4. Evaluate and visualize rollouts.

## Project Layout

```text
configs/                 YAML experiment and environment configs
assets/                  MuJoCo models, terrains, textures, and robot assets
quadruped_mujoco_rl/     Python source package
scripts/                 Convenience shell entrypoints
tests/                   Smoke tests and behavior checks
```

## Setup

```bash
conda create -n quadruped-mujoco-rl python=3.10 -y
conda activate quadruped-mujoco-rl
pip install -e ".[dev]"
```

## Commands

```bash
python scripts/check_mujoco_model.py
python scripts/watch_random_env.py
python scripts/watch_random_env.py --config configs/env_go2.yaml
python scripts/start_tensorboard.py --logdir runs
pytest
```

The Gymnasium environment can be created with:

```python
import gymnasium as gym
import quadruped_mujoco_rl

env = gym.make("QuadrupedFlat-v0")
obs, info = env.reset()
```

Planned training commands:

```bash
python -m quadruped_mujoco_rl.training.train_ppo --config configs/train_ppo.yaml
python -m quadruped_mujoco_rl.training.train_ppo --config configs/train_ppo_go2.yaml
python -m quadruped_mujoco_rl.training.train_ppo --config configs/train_ppo_go2_lr1e-4.yaml
python -m quadruped_mujoco_rl.training.train_ppo --config configs/train_ppo_go2_stable.yaml
python -m quadruped_mujoco_rl.training.train_ppo --config configs/train_ppo_go2_quality.yaml
python -m quadruped_mujoco_rl.training.train_ppo --config configs/train_ppo_go2_quality_v2.yaml
python -m quadruped_mujoco_rl.training.train_ppo --config configs/train_ppo_go2_command.yaml
python -m quadruped_mujoco_rl.training.train_ppo --config configs/train_ppo_go2_command_v2.yaml
python -m quadruped_mujoco_rl.training.train_ppo \
  --config configs/train_ppo_go2_command_v3_phase1.yaml \
  --init-from runs/ppo_go2_quality_v2/checkpoints/go2_1m_quality_v2.zip
python -m quadruped_mujoco_rl.training.train_ppo \
  --config configs/train_ppo_go2_command_v4_canonical.yaml
python -m quadruped_mujoco_rl.training.train_ppo \
  --config configs/train_ppo_go2_command_v4_1_gait.yaml \
  --init-from runs/ppo_go2_command_v4_canonical/checkpoints/go2_1m_command_v4_canonical.zip
python -m quadruped_mujoco_rl.training.evaluate --checkpoint runs/ppo/checkpoints/final_model.zip
python -m quadruped_mujoco_rl.training.evaluate --config configs/env_go2.yaml --checkpoint runs/ppo_go2/checkpoints/go2_1m_baseline.zip --episodes 3 --no-render
```

For a quick smoke run before a long training job:

```bash
python -m quadruped_mujoco_rl.training.train_ppo --config configs/train_ppo.yaml --total-timesteps 512
```
