# 四足机器人强化学习项目日志

最后更新：2026-06-25

本文档记录项目从初始化到当前阶段的完整演进，包括目标、实现、实验结果、失败原因和后续决策。它是一份持续维护的工程日志，不是只展示成功结果的说明书。

## 当前状态

- 当前机器人：Unitree Go2 MuJoCo Menagerie 模型。
- 当前算法：Stable-Baselines3 PPO。
- 当前控制方式：策略输出关节目标位置偏移，PD 控制器转换为电机 torque。
- 已完成：平地前进、步态质量改进、评估指标、TensorBoard、checkpoint 迁移。
- 最新完成：`command_v4_canonical` 训练 1M timesteps 并完成三档速度评估。
- v4 已经学会根据 command 调速，且机身姿态稳定；当前主要问题变为航向缓慢漂移。
- 最新准备：`command_v4_1_gait` 已完成实现和 smoke test，等待正式微调。
- 最新路线变化：保留本 SB3 项目不动，另建 `~/projects/mujoco_playground` 验证 Google DeepMind MuJoCo Playground / MJX GPU 并行训练路线。
- 最新验证：`~/projects/mujoco_playground` 中 JAX GPU、`mujoco_playground` 导入、官方 `Go1JoystickFlatTerrain` 环境加载和 Go1 PPO 短训练/checkpoint 保存均已通过；视频渲染仍被本机 headless OpenGL/EGL 配置阻塞。
- 尚未完成：可靠的速度命令跟踪、转向、复杂地形、视觉跟随、避障、D1 Ultra sim-to-real。

## 项目最终方向

最初目标是在 MuJoCo 中训练四足机器人，并考虑未来部署到 D1 Ultra。由于暂未找到经过验证的 D1 Ultra 官方 URDF/MJCF 和完整执行器参数，当前先使用资料和模型更成熟的 Unitree Go2 完成训练方法验证。

长期希望形成以下能力：

1. 稳定、自然的四足移动。
2. 根据速度命令前进和转向。
3. 在坡面、台阶和障碍物等复杂环境中运动。
4. 接入视觉检测，实现目标跟随和避障。
5. 获取 D1 Ultra 的可靠模型与硬件接口后，再评估 sim-to-real。

## 阶段一：工程初始化

项目最初建立了 Python 包、配置、模型资源、脚本和测试目录：

```text
configs/                 环境和训练实验配置
assets/robots/           MuJoCo 机器人模型
quadruped_mujoco_rl/     环境、训练、评估和工具代码
scripts/                 模型查看和辅助脚本
tests/                   环境行为与兼容性测试
runs/                    TensorBoard 日志和训练模型
```

### 环境安装问题

早期直接执行 `pip install` 时遇到 PEP 668 提示，因为系统 Python 受 Debian 管理。同时 `which python` 指向 Conda 环境，而 `which pip` 指向 `~/.local/bin/pip`，说明 Python 和 pip 没有来自同一个环境。

最终做法是使用独立 Conda 环境，并通过环境中的 Python 调用 pip：

```bash
conda activate quadruped-mujoco-rl
python -m pip install -e ".[dev]"
```

### 初始机器人模型

项目最早使用自制的简化四足 MJCF，用于验证 MuJoCo 加载、`reset()`、`step()` 和可视化流程。这个模型的默认姿态和物理参数不够可靠，初始形态甚至无法自然站立，因此不适合继续承担步态训练。

### 切换 Unitree Go2

随后引入 MuJoCo Menagerie 的 Unitree Go2 模型。Go2 提供了更可信的几何结构、质量、关节限制和执行器信息，项目由此从“验证代码能运行”进入“研究 locomotion 训练”的阶段。

## 阶段二：基础环境与第一批 PPO

环境被封装为 Gymnasium `QuadrupedFlat-v0`，核心接口为：

- `reset()`：恢复初始姿态并返回 observation。
- `step(action)`：执行动作、推进 MuJoCo、计算 reward 和终止条件。
- action：12 个关节目标位置偏移。
- PD 控制：`torque = kp * (q_target - q) - kd * qvel`。
- observation：早期为 `qpos + qvel + last_action`。

同时加入 PPO 训练、checkpoint 保存、MuJoCo 可视化评估和 TensorBoard。

## 阶段三：前进策略实验

### Baseline 1M

配置：`configs/train_ppo_go2.yaml`

模型：`runs/ppo_go2/checkpoints/go2_1m_baseline.zip`

结果：

| 指标 | 结果 |
|---|---:|
| 前进距离 | 25.05 m |
| 平均前进速度 | 1.25 m/s |
| 横向漂移绝对值 | 5.65 m |

结论：机器人学会了快速向前，但路线明显偏斜，动作质量不是优化重点。

### Lower LR 1M

配置：`configs/train_ppo_go2_lr1e-4.yaml`

模型：`runs/ppo_go2_lr1e-4/checkpoints/go2_1m_lr1e-4.zip`

把学习率从 `3e-4` 降到 `1e-4` 后：

| 指标 | 结果 |
|---|---:|
| 前进距离 | 27.40 m |
| 平均前进速度 | 1.37 m/s |
| 横向漂移绝对值 | 3.60 m |

结论：速度和距离有所提升，漂移减少，但仍不是直线、自然的步态。

### Stable 1M

配置：`configs/train_ppo_go2_stable.yaml`

模型：`runs/ppo_go2_stable/checkpoints/go2_1m_stable.zip`

增加横向运动和姿态约束后：

| 指标 | 结果 |
|---|---:|
| 前进距离 | 约 24.70 m |
| 平均前进速度 | 约 1.24 m/s |
| 横向漂移绝对值 | 约 0.25 m |

结论：直线稳定性显著改善，证明 reward 能改变行为风格。但“稳定”不等于“自然”，肉眼仍能看到头部和身体姿态不理想。

## 阶段四：步态质量实验

### Quality v1

配置：`configs/train_ppo_go2_quality.yaml`

模型：`runs/ppo_go2_quality/checkpoints/go2_1m_quality.zip`

这一版增加目标速度、姿态、动作平滑、足端打滑等约束。结果速度降到约 `0.32 m/s`，机身偏低，而且四足使用极不均衡：

| 足端 | 接触比例 |
|---|---:|
| FL | 0.997 |
| FR | 0.992 |
| RL | 0.178 |
| RR | 0.716 |

结论：reward 过度限制动作，策略找到了一种低速、偏斜、少用某条腿的局部最优解。这是典型 reward hacking：数值可能变好，但行为不是想要的。

### Quality v2

配置：`configs/train_ppo_go2_quality_v2.yaml`

模型：`runs/ppo_go2_quality_v2/checkpoints/go2_1m_quality_v2.zip`

加入 gait phase 和对角腿接触参考后：

| 指标 | 结果 |
|---|---:|
| 平均速度 | 约 1.04 m/s |
| 平均机身高度 | 约 0.27 m |
| 平均 pitch | 约 0.057 rad |
| 横向漂移 | 约 1.81 m |
| FL/FR/RL/RR 接触比例 | 0.545 / 0.622 / 0.624 / 0.640 |

结论：四足使用明显均衡，速度恢复，成为当时最可靠的基础步态。但硬编码接触相位仍可能让策略为了匹配时序而产生不自然姿态。

## 阶段五：速度命令策略

目标从“固定向前跑”升级为：策略接收 `target_vx` 和 `target_yaw_rate`，根据命令调整动作。

### Command v1

配置：`configs/train_ppo_go2_command.yaml`

模型：`runs/ppo_go2_command/checkpoints/go2_2m_command.zip`

训练 2M timesteps。主要问题是速度使用世界坐标系，机器人自身朝向变化后，reward 中的“向前”与机器人机身前方不一致。

结论：命令控制概念已经接入，但坐标系设计错误会直接污染学习目标。

### Command v2

配置：`configs/train_ppo_go2_command_v2.yaml`

模型：`runs/ppo_go2_command_v2/checkpoints/go2_100k_command_v2.zip`

改为机身坐标系速度后训练 100k。策略没有学会前进，而是趋向静止。

结论：仅有速度跟踪奖励时，随机初始化的 PPO 仍可能发现“站着风险更小”。训练步数少、奖励差异不够明确、单环境采样效率低共同造成了这个问题。

### Command v3 Phase 1，从零训练

配置：`configs/train_ppo_go2_command_v3_phase1.yaml`

备份模型：`runs/ppo_go2_command_v3_phase1/checkpoints/go2_100k_command_v3_phase1.zip`

这一版只采样 `0.5-1.0 m/s` 的直行命令，并加入显式速度误差扣分。100k 后仍然失败：

- 三档目标速度下实际速度都约 `0.01 m/s`。
- 平均机身高度约 `0.20 m`。
- pitch 约 `0.31 rad`。
- 策略选择趴低、静止并承受负 reward。

结论：增加“不动扣分”能够让 reward 正确表达不满意，却不会自动提供一条从随机动作到稳定步态的学习路径。

### Command v3 Phase 1，Quality v2 权重迁移

训练入口新增 `--init-from`。迁移时复制 quality v2 的网络权重，并把新增的两个 command 输入列初始化为零，从而保留已学会的步态。

迁移后再训练 100k 的结果：

| 目标速度 | 实际速度 |
|---|---:|
| 0.50 m/s | 约 0.96 m/s |
| 0.75 m/s | 约 0.96 m/s |
| 1.00 m/s | 约 0.96 m/s |

其他表现：

- 机身高度约 `0.27 m`。
- pitch 约 `0.04 rad`。
- 四足接触比例约 `0.54-0.66`。
- 步态成功保留，但三档命令输出几乎相同。

网络权重检查显示，critic 已经使用 command 信息，而 actor 对 command 两列的平均权重只有约 `0.0016`。也就是说，网络知道不同命令会影响回报，却没有学会用命令改变动作。

结论：迁移解决了“从零学不会走”的问题，但没有解决命令条件化。继续堆训练步数或临时调权重不是可靠路线。

## 阶段六：Canonical v4

配置：

- `configs/env_go2_command_v4_canonical.yaml`
- `configs/train_ppo_go2_command_v4_canonical.yaml`

输出目录：`runs/ppo_go2_command_v4_canonical/`

这一版不覆盖 v3，旧配置和旧 checkpoint 都被保留。环境通过配置选择 legacy 或 canonical 分支。

### Observation 重构

旧 observation 使用原始 `qpos + qvel + last_action + command`，包含全局位置和原始四元数。v4 改为标准 47 维 locomotion observation：

```text
body linear velocity       3
body angular velocity      3
projected gravity          3
velocity command           2
joint position error      12
joint velocity            12
last action               12
总计                       47
```

全局 XY 位置不再进入策略，因此机器人跑到场地不同位置不会改变动作判断。

### Reward 重构

v4 去掉了固定世界 `yaw=0` 惩罚和硬编码对角腿接触奖励，改为主流 locomotion 结构：

正向目标：

- 机身坐标系 XY 速度命令跟踪。
- yaw 角速度命令跟踪。
- 足端合理腾空时间。

动作与稳定性约束：

- 垂直速度。
- roll/pitch 方向角速度。
- projected gravity 水平姿态误差。
- 电机 torque。
- action rate。
- 默认关节姿态偏差。
- 接触地面时的足端滑移。
- 非足部触地。
- 机身高度偏差和摔倒。

每个 canonical reward 分量都会写入 `info["reward_terms"]`，便于后续判断究竟是哪一项主导策略，而不只看总 reward。

### 训练与评估结果

v4 从零训练 1M timesteps，不从 quality v2 迁移，因为新旧 observation 的维度和语义均不同。

训练完成于 2026-06-24，最终模型另行保留为：

`runs/ppo_go2_command_v4_canonical/checkpoints/go2_1m_command_v4_canonical.zip`

三档固定直行命令的评估结果：

| 目标速度 | 实际机身前进速度 | 平均绝对速度误差 | yaw 漂移 | 横向位移 |
|---:|---:|---:|---:|---:|
| 0.50 m/s | 0.51 m/s | 0.05 m/s | -0.593 rad | -3.12 m |
| 0.75 m/s | 0.73 m/s | 0.05 m/s | -0.215 rad | -1.76 m |
| 1.00 m/s | 0.92 m/s | 0.08 m/s | -0.370 rad | -3.39 m |

共同表现：

- 所有 episode 均达到 1,000 steps，没有摔倒。
- 平均机身高度约 `0.26-0.27 m`。
- 平均绝对 roll 约 `0.014-0.052 rad`。
- 平均绝对 pitch 约 `0.015-0.027 rad`。
- 机身坐标系横向速度接近 `0 m/s`。
- 四足接触比例约 `0.56-0.74`，没有再次出现某条腿几乎不用的情况。

结论：canonical v4 首次同时实现了稳定步态和明显的 command 调速。机器人机身本身已经较水平，早期“头朝下、身体明显歪”的问题大幅改善。

当前的横向位移主要来自航向缓慢旋转，而不是机身坐标系中的横向滑行。例如 `0.5 m/s` 时平均 yaw rate 只有约 `-0.03 rad/s`，但在 20 秒 episode 内会累积成约 `-0.59 rad`，因此世界坐标轨迹呈弧线。

训练末尾 `approx_kl` 约 `0.09`、`clip_fraction` 约 `0.56`，说明最后阶段的 PPO 更新偏大。后续微调航向时应考虑降低学习率，而不是继续使用相同参数长时间训练。

### v4 步态专项复核

速度跟踪成功后，进一步对 `0.75 m/s` 的足端时序、腾空高度、滑移和动作对称性进行了专项分析。

结果：

- 每条腿在 20 秒内约发生 `206` 次接触状态切换，对应约 `5.2 Hz` 的快速步频。
- 约 `63.6%` 的时间有三条腿同时着地，`27.7%` 的时间有两条腿着地。
- 严格的两组对角腿交替支撑比例接近 `0`，没有形成清晰 trot。
- 单次着地中位时间约 `0.12-0.14 s`，腾空中位时间约 `0.06-0.08 s`。
- 足端最大腾空高度约 `0.039-0.051 m`。
- 接触期间足端平均水平速度约 `0.18-0.33 m/s`，存在明显滑动。
- RR 接触比例约 `0.73`，高于其他腿，仍存在一定左右/前后不对称。

reward 分量检查显示：

- `linear_velocity_tracking` 总贡献约 `1475`。
- `yaw_rate_tracking` 总贡献约 `746`。
- `foot_slip` 总惩罚约 `-21`。
- `feet_air_time` 总奖励仅约 `0.001`，事实上没有发挥作用。

结论：v4 是稳定且能调速的功能性步态，但还不能称为自然、标准的四足小跑。当前策略主要采用高频、低腾空、三脚支撑的小碎步，并伴随足端滑动。机身姿态正常不代表腿部时序正常。

形成这种行为的直接原因是速度跟踪 reward 占绝对主导，而实际腾空时间低于 `feet_air_time_threshold=0.15 s`，导致腾空奖励几乎始终为零；滑移惩罚相对速度奖励也偏弱。

## 阶段七：Canonical v4.1 Gait

配置：

- `configs/env_go2_command_v4_1_gait.yaml`
- `configs/train_ppo_go2_command_v4_1_gait.yaml`

输出目录：`runs/ppo_go2_command_v4_1_gait/`

这一版保持 v4 的 47 维 observation 和网络结构不变，从 v4 1M checkpoint 迁移全部策略权重。目标是保留已经成功的速度控制，仅微调腿部动作质量。

主要变化：

- `feet_air_time_threshold` 从 `0.15 s` 降到 `0.05 s`，让当前短腾空动作也能获得非零学习信号。
- `feet_air_time_reward` 从 `0.1` 提高到 `0.5`。
- `foot_slip_penalty` 从 `0.1` 提高到 `0.5`。
- 新增连续 `foot_clearance` 代价，摆动腿目标足端高度为 `0.06 m`。
- `action_rate_penalty` 从 `0.01` 提高到 `0.02`。
- 新增 joint acceleration 惩罚，抑制高频关节抖动。
- 学习率降到 `1e-4`，正式微调长度设为 200k timesteps。

在尚未微调、直接把 v4 策略放入 v4.1 reward 时，单个 episode 的主要加权分量为：

| Reward 分量 | 总贡献 |
|---|---:|
| linear velocity tracking | 约 1475 |
| action rate | 约 -85 |
| joint acceleration | 约 -65 |
| foot slip | 约 -105 |
| foot clearance | 约 -7 |
| feet air time | 约 8.1 |

与 v4 相比，feet-air-time 不再为零，滑移和高频动作也获得了足够明显但没有压过速度目标的权重。256 timesteps PPO smoke test 已通过，尚未进行正式 200k 微调。

## 阶段八：MuJoCo Playground / MJX 路线验证

由于单环境 Stable-Baselines3 PPO 加 classic MuJoCo 的采样效率较低，四足步态 reward 调试周期过长，项目路线新增一条独立验证分支：先不改动当前 `quadruped_mujoco_rl`，而是在 `~/projects/mujoco_playground` 中验证官方 MuJoCo Playground / MJX GPU 并行训练是否能在本机跑通。

环境隔离原则：

- 当前 SB3 项目继续使用 Conda 环境 `quadruped-mujoco-rl`。
- MuJoCo Playground 单独使用 `~/projects/mujoco_playground/.venv`，由 `uv` 管理。
- 不把 Playground 装进原 SB3 项目的 Conda 环境。

2026-06-25 完成三项 smoke test：

| 检查项 | 命令 | 结果 |
|---|---|---|
| JAX GPU | `uv run python -c "import jax; print(jax.__version__); print(jax.default_backend()); print(jax.devices())"` | JAX `0.6.2`，backend 为 `gpu`，设备为 `CudaDevice(id=0)` |
| Playground 导入 | `uv --no-config run python -c "import mujoco_playground; print('Success')"` | 输出 `Success` |
| 官方 Go1 环境 | `uv run python -c "from mujoco_playground import locomotion; env = locomotion.load('Go1JoystickFlatTerrain'); print(env)"` | 成功创建 `mujoco_playground._src.locomotion.go1.joystick.Joystick` 环境 |

Go1 环境加载时 Warp 初始化到 `cuda:0`，GPU 为 `NVIDIA GeForce RTX 5060 Laptop GPU`，显存 `8 GiB`。这说明当前机器已经具备继续测试官方 Go1 训练和评估 demo 的基本条件。

随后运行官方 Go1 PPO 训练入口的短训练 smoke test：

```bash
uv --no-config run train-jax-ppo --env_name Go1JoystickFlatTerrain --num_timesteps=4096 --num_envs=64 --num_eval_envs=4 --num_evals=1 --episode_length=100 --unroll_length=10 --num_minibatches=4 --num_updates_per_batch=1 --batch_size=128 --num_videos=1 --suffix=smoke --logdir=/home/jensen/projects/mujoco_playground/logs_smoke
```

结果：

- 训练入口、JAX/MJX 编译、PPO 更新和 checkpoint 保存成功。
- 实际保存 checkpoint：`logs_smoke/Go1JoystickFlatTerrain-20260625-103339-smoke/checkpoints/000000051200`。
- 首次 JIT compile 用时约 `135 s`。
- 短训练只用于验证链路，不代表策略质量；输出 reward 为 `0.000`。

训练后脚本进入自动推理和视频渲染阶段，但渲染失败：

- 默认路径出现 GLX framebuffer 警告，并在 `mujoco.Renderer` 初始化时报 `gladLoadGL error`。
- 显式设置 `MUJOCO_GL=egl` 后，checkpoint 恢复和推理前半段正常，但渲染时报 `Cannot initialize a EGL device display`，提示 EGL driver 不支持创建 headless rendering context 所需的 `PLATFORM_DEVICE` 扩展。
- 显式设置 `MUJOCO_GL=osmesa` 后，MuJoCo/OpenGL 导入阶段失败，`OpenGL` 没有可用的 `glGetError`，说明当前环境没有可用的 OSMesa 软件渲染后端。

结论：MuJoCo Playground / MJX 路线的本机训练和 checkpoint 基础链路已通过初步验证；阻塞点转移到 headless 渲染/视频导出配置。下一步先修复或绕过渲染后端，再运行更接近官方默认参数的 Go1 训练/评估 demo，确认吞吐和可视化评估流程。

## 已确认的主要问题

### 1. 模型问题与算法问题必须分开

最初简化模型站不稳，继续调 PPO 没有意义。换用 Go2 后，物理模型基础更可靠，策略问题才值得分析。

### 2. 坐标系错误会让 reward 目标失真

速度命令应该在机器人机身坐标系中解释。使用世界 X 速度时，机器人一旦转向，任务定义就发生变化。

### 3. Reward 数值提高不代表动作正确

趴着、少用一条腿、斜着跑都可能是数学上更容易的局部最优。必须同时检查视频、速度误差、姿态、接触比例和足端滑移。

### 4. Reward 项不是越多越好

多个弱约束可能彼此冲突。旧版同时使用固定 yaw、yaw rate、横向速度和硬编码 gait contact，策略可能通过身体扭斜来折中这些目标。

### 5. 迁移学习只能迁移兼容的能力

Quality v2 的步态可以迁移到增加 command 输入的网络，但零初始化的新 command 权重也意味着初始策略会忽略命令。迁移保住了步态，却不会自动产生调速能力。

### 6. 单环境 PPO 的采样效率有限

当前使用一个 MuJoCo 环境，训练速度和 command 多样性都低于常见的数千并行环境方案。后续若项目继续扩大，需要评估并行 MuJoCo、MJX 或 Isaac Lab。

### 7. Sim-to-real 仍缺少关键条件

Go2 策略不能直接部署到 D1 Ultra。机器人尺寸、质量、关节顺序、限位、减速器、电机 torque、PD 参数、通信频率和传感器都必须匹配。当前项目首先验证训练方法，不宣称已经完成 D1 Ultra sim-to-real。

## 评估原则

后续每个模型至少需要检查：

1. `mean_forward_velocity` 与目标速度误差。
2. `mean_lateral_velocity` 和横向漂移。
3. roll、pitch、yaw rate 和机身高度。
4. 四足接触比例、足端滑移和腾空时间。
5. action、torque 和关节速度。
6. terminated 次数。
7. MuJoCo 中的实际动作观感。

只看 `ep_rew_mean` 不足以判断模型是否可用，因为 reward 定义本身会随版本变化。

## 当前下一步

1. 保留 canonical v4 作为“稳定调速基线”，不覆盖 1M checkpoint。
2. 暂缓继续在单环境 SB3 中长时间调 reward，优先修复或绕过 MuJoCo Playground 的 headless 渲染问题。
3. 渲染链路处理后，运行更接近官方默认参数的 Go1 训练/评估 demo，确认并行训练吞吐、checkpoint 恢复和可视化评估流程。
4. 如果 Playground Go1 demo 完整跑通，再评估 Go2 迁移方式：直接改官方 Go1/Go2 locomotion 环境，或把当前 SB3 项目的 observation、reward、评估指标逐步移植到 Playground。
5. 若继续推进原 SB3 分支，则从 v4 checkpoint 正式训练 v4.1 gait 200k timesteps，并重新统计支撑模式、步频、腾空时间、滑移和动作对称性。
6. 步态质量达标后，再增加相对航向目标并处理累计 yaw 漂移；随后加入正负 yaw-rate 命令，最后进入复杂地形与视觉避障。

## 文档维护规则

以后每次发生以下事件，都同步更新本文档：

- 新增或修改环境、observation、reward、控制器。
- 开始新的正式训练版本。
- 训练结束并完成定量评估。
- 某条方案被证明有效或失败。
- checkpoint 被保留、替换或废弃。
- 项目的近期计划发生变化。

更新时保留失败记录，不用后来的结论覆盖早期过程；这样可以解释项目为什么演进到当前设计。

## 更新记录

### 2026-06-25

- 决定保留 `quadruped_mujoco_rl` 原 SB3 项目不动，另建 `~/projects/mujoco_playground` 验证官方 MuJoCo Playground / MJX GPU 并行训练路线。
- 明确环境隔离：SB3 项目继续使用 Conda `quadruped-mujoco-rl`，Playground 使用独立 `uv` `.venv`。
- 完成 JAX GPU smoke test：JAX `0.6.2`，默认 backend 为 `gpu`，设备为 `CudaDevice(id=0)`。
- 完成 `mujoco_playground` 导入 smoke test。
- 完成官方 `Go1JoystickFlatTerrain` 环境加载 smoke test，Warp 成功初始化到 RTX 5060 Laptop GPU。
- 完成官方 Go1 PPO 短训练 smoke test，确认训练入口、JAX/MJX 编译、PPO 更新和 checkpoint 保存链路可用。
- 保存 smoke checkpoint：`logs_smoke/Go1JoystickFlatTerrain-20260625-103339-smoke/checkpoints/000000051200`。
- 记录渲染失败：默认 GLX 报 `gladLoadGL error`，`MUJOCO_GL=egl` 报 EGL device display 初始化失败，`MUJOCO_GL=osmesa` 在 OpenGL 导入阶段失败。
- 更新近期路线：先处理 Playground headless 渲染/视频导出问题，再运行更完整的官方 Go1 训练/评估 demo，最后决定 Go2 或本项目经验的迁移方式。

### 2026-06-24

- 创建项目开发日志。
- 整理从简化模型、Go2 baseline 到 canonical v4 的完整过程。
- canonical v4 完成 1M timesteps 训练并保存独立 checkpoint。
- 完成 `0.5 / 0.75 / 1.0 m/s` 三档评估，确认调速和姿态控制成功。
- 将下一阶段问题定位为小幅 yaw-rate 偏置造成的累计航向漂移。
- 完成 v4 步态专项复核，确认当前行为是高频三脚支撑小碎步，而非自然 trot。
- 发现 feet-air-time reward 实际贡献接近零，更新下一阶段优先级为步态质量修正。
- 新增 canonical v4.1 gait，强化腾空、足端高度、滑移、action rate 和关节加速度信号。
- 完成 v4 到 v4.1 的完整权重迁移与 256 timesteps smoke test，等待正式 200k 微调。
