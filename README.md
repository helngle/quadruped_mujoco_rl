# Quadruped MuJoCo RL

This project is for training quadruped locomotion policies in MuJoCo with reinforcement learning.

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
python -m quadruped_mujoco_rl.training.evaluate --checkpoint runs/ppo/checkpoints/final_model.zip
```

For a quick smoke run before a long training job:

```bash
python -m quadruped_mujoco_rl.training.train_ppo --config configs/train_ppo.yaml --total-timesteps 512
```
