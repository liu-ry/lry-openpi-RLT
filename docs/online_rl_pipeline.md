# 在线强化学习（Online RL）完整流程

本文档基于仓库当前状态，描述从训练好的 RLT/openpi 基础模型出发，执行在线 RL 微调的完整 pipeline。

---

## 概览：双机架构

在线 RL 系统分为两台机器：

```
┌─────────────────────────────────────────────────────────────┐
│  Machine A（高算力推理服务器）                                │
│  ● 运行冻结的 openpi/RLT 策略服务 (serve_rlt_policy.py)      │
│  ● 每次 query 返回：                                          │
│      - z_rl：紧凑 RL-token 特征向量（2048 维）                │
│      - ref_chunk：VLA 参考动作块（chunk_len × action_dim）    │
└────────────────────────────┬────────────────────────────────┘
                             │ WebSocket (ws://)
┌────────────────────────────▼────────────────────────────────┐
│  Machine B（在线 RL 训练机器）                                │
│  ● B1 actor_service   轻量 ChunkActor 推理（HTTP）            │
│  ● B2 learner_service  Actor+Critic 训练，发布 snapshot        │
│  ● B3 replay_manager   Replay 缓冲区 + append-only journal    │
│  ● B4 EnvDriver (robot rollout)  连接 ROS / 环境              │
└─────────────────────────────────────────────────────────────┘
```

---

## 一、前置准备

### 1.1 训练基础 RLT 模型

在进行在线 RL 之前，需要先使用离线数据训练好 openpi/RLT 模型：

```bash
# 在仓库根目录
python scripts/train_rlt.py \
  --config rlt_pi05_agilexbag_image_delta_joint \
  --exp-name my_rlt_run
```

产出：`checkpoint-dir/` 下的 JAX 检查点。

### 1.2 环境配置

**Machine B 独立 Python 3.10 环境：**

```bash
cd openpi-RLT/rlt_online_rl
conda create -y -n rlt_online_rl310 python=3.10 pip
conda activate rlt_online_rl310
pip install --upgrade pip setuptools wheel
pip install -e ../packages/openpi-client
pip install -e .
# 可选：W&B 监控
pip install -e '.[monitor]'
```

**ROS（机器人 rollout 节点需要）：**

```bash
source /opt/ros/humble/setup.bash
```

### 1.3 任务配置文件

当前公开任务配置：`rlt_online_rl/configs/tasks/agilex_ethernet/online_rl.yaml`

核心参数（Ethernet 任务默认值）：

| 参数 | 值 | 含义 |
|------|----|------|
| `action_dim` | 7 | 动作维度 |
| `chunk_len` | 10 | 每个 chunk 的步数 |
| `z_dim` | 2048 | RL-token 特征维度 |
| `proprio_dim` | 7 | 本体感知维度 |
| `action_representation` | `delta_chunk` | 动作表示方式 |
| `reference_dropout_prob` | 0.5 | 参考动作 dropout 概率 |
| `warmup_min_size` | 600 | Warmup 最小 Replay 大小 |
| `warmup_post_collect_updates` | 20000 | Warmup 后额外训练步数 |
| `grad_updates_per_cycle` | 5 | 每次新数据的梯度更新比 |
| `control_frequency_hz` | 20.0 | 控制频率 |

---

## 二、启动服务

### 2.1 启动 Machine A（冻结 VLA 策略服务）

在 Machine A 上，从仓库根目录：

```bash
conda activate <openpi_env>
python scripts/serve_rlt_policy.py \
  --config rlt_pi05_agilexbag_image_delta_joint \
  --checkpoint-dir <checkpoint-dir> \
  --port 8000 \
  --shared-prefix-inference   # 可选：启用共享前缀推理优化延迟
```

`--shared-prefix-inference` 会让服务器对 `z_rl` 提取和 `ref_chunk` 生成复用同一次 VLA prefix 前向，不影响训练权重或归一化。

**本地 debug 时（无真实 VLA 服务器）：**

```bash
cd rlt_online_rl
python launch/fake_machine_a.py
```

### 2.2 启动 Machine B 服务（`launch_machine_b.py`）

```bash
cd openpi-RLT/rlt_online_rl
conda activate rlt_online_rl310
python launch/launch_machine_b.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml
```

这会按顺序以独立进程启动：
1. **B3 replay_manager** (port 9102) — 首先启动，等待 0.5s
2. **B2 learner_service** (GPU 进程) — 启动，等待 0.5s
3. **B1 actor_service** (port 9101) — 启动，等待其 `/healthz` 就绪

进程间通信使用 HTTP + msgpack-numpy 序列化。

### 2.3 启动机器人 Rollout（`launch_robot_rollout.py`）

```bash
cd openpi-RLT/rlt_online_rl
source /opt/ros/humble/setup.bash
conda activate rlt_online_rl310
python launch/launch_robot_rollout.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml \
  --machine_a_ws_url ws://MACHINE_A_IP:8000
```

Rollout 会等待：
- Machine A WebSocket 服务就绪
- Machine B actor_service 就绪（`actor_param_version >= 0`）

### 2.4 启动训练键盘客户端

```bash
python keyboard_toggle_teleop_record_reward_isolation.py
```

常用按键：

| 键 | 功能 |
|----|------|
| `o` | 开始新 episode |
| `c` | 标记进入 critical phase（full_task 模式）|
| `s` | 记录成功并结束 episode |
| `f` | 记录失败并结束 episode |
| `t` | 切换 teleop 人类接管 |

---

## 三、在线 RL 训练循环详解

### 3.1 Chunk 级执行路径

每到 chunk 边界时（每 `chunk_exec_horizon` 个控制 tick）：

```
机器人观测 obs
    │
    ▼
Machine A (WebSocket)
    ├── z_rl  [2048]        ← 压缩后的 RL-token 特征
    └── ref_chunk [10×7]    ← VLA 冻结参考动作块
    │
    ▼
从 obs.state 提取 proprio [7]
    │
    ├─ [Warmup / full_task 非关键段]
    │       直接执行 ref_chunk（BASE 模式）
    │
    └─ [Online 关键段]
            ▼
        Machine B actor_service (HTTP)
        输入: z_rl, proprio, ref_chunk
        输出: refined_chunk [10×7]（RL 精炼后）
            │
            ▼
        执行 refined_chunk（RL 模式）
    │
    ▼
记录 RawEpisodeStep（观测、动作、奖励、source 标记）
```

### 3.2 三个训练阶段

#### 阶段 1：Warmup（预热）

- 机器人完全执行 `ref_chunk`（来自冻结 VLA），source = `BASE`
- 每完成一个 episode，将 replay transitions 推送到 B3
- 当 `replay_size >= warmup_min_size`（默认 600）时，learner 开始训练
- 如果设置了 `warmup_post_collect_updates`，learner 会在此步数完成后才允许 online 控制
- Warmup 阶段使用 `warmup_bc_weight / warmup_q_weight`（默认均为 1.0）

#### 阶段 2：Warmup Wait Online（等待 Online 就绪）

- 等待以下两个条件**同时满足**：
  - `learner_status.ready_for_online == true`
  - `actor_param_version >= rollout_threshold`
- 等待发生在 episode 之间，不在 episode 中途切换

#### 阶段 3：Online（在线学习）

- 关键段由 ChunkActor（B1）控制，source = `RL`
- 非关键段前缀仍用 VLA ref（full_task 模式下），不写入 replay
- 使用 `online_bc_weight / online_q_weight`（可与 warmup 不同）
- 训练 eval：可设置 `actor_deterministic=True` 使用确定性 actor 均值

### 3.3 Episode 结束时的 Replay 构建

每个 episode 结束后，`EnvDriver` 从原始步轨迹（`RawEpisodeTrace`）构建 replay transitions：

**模式 A：`step_trace_stride = 0`（当前 Ethernet 配置默认）**
- 只在 chunk 边界构建 transitions
- 添加 policy restart anchors（人工接管后恢复策略时）
- 可能追加一个 terminal aligned 末尾 window
- 对缺失的 Machine A anchors 进行回填（backfill）

**模式 B：`step_trace_stride > 0`（如设为 2，对应 RLT 论文设置）**
- 以步长为 stride 构建密集 replay windows
- 使用批量 Machine A 特征回填

每个 `RLTTransition` 包含：
```
z_rl, proprio, ref_chunk, action_chunk,
rewards, done,
next_z_rl, next_proprio, next_ref_chunk,
source, source_chunk, collection_phase,
success, intervention_flag,
episode_id, step_id
```

### 3.4 Learner 训练目标

Learner（B2）从 replay 采样批次，执行 JAX JIT 编译的 `train_step`：

**Critic 更新（每步）：**
$$\mathcal{L}_{\text{critic}} = \text{TD-error on twin Q-networks (clipped double Q)}$$

**Actor 更新（每 `actor_update_period` 步，默认 2）：**
$$\mathcal{L}_{\text{actor}} = \lambda_{\text{bc}} \cdot \mathcal{L}_{\text{bc}} - \lambda_q \cdot Q(s, \pi(s)) + \lambda_\delta \cdot \mathcal{L}_{\delta}$$

其中：
- $\mathcal{L}_{\text{bc}}$：与 BC target 的均方误差
  - `HUMAN/MIXED` 步骤 → target = 人工执行的 `action_chunk`
  - `BASE/RL` 步骤 → target = VLA 的 `ref_chunk`
- $Q(s, \pi(s))$：Actor Q 值（最大化）
- $\mathcal{L}_{\delta}$：step-to-step delta 平滑正则项（前六个关节）

**Replay 采样策略（stratified 模式）：**
- `recent_online_ratio`：近期 online episodes 占比
- `warmup_demo_ratio`：warmup 演示数据占比
- `human_intervention_ratio`：人工干预数据占比

### 3.5 Actor Snapshot 热更新

```
Learner 训练 → 每 push_actor_interval_steps 步（默认 50）
    → 写入 actor_snapshot.pkl（原子替换）
    → Actor Service 轮询检测（默认 0.25s 间隔）
    → 加载新参数 → warmup JIT 编译
    → 热更新 actor_params（加锁）
```

---

## 四、Eval-only 模式

仅运行 actor 推理，不启动 learner 和 replay：

```bash
# 启动 actor service
python scripts/run_online_rl.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml \
  --system.role actor_service \
  --system.actor_service.snapshot_path <actor_snapshot.pkl>

# 启动 eval rollout
python launch/launch_actor_eval.py \
  --config configs/tasks/agilex_ethernet/online_rl.yaml \
  --machine_a_ws_url ws://MACHINE_A_IP:8000

# Eval 键盘客户端
python keyboard_actor_eval.py
```

Eval rollout 强制使用确定性 actor 均值（`deterministic=True`）。

---

## 五、工具与监控

### 检查 Replay Journal

```bash
python scripts/tools/inspect_replay_journal.py \
  runs/agilex_ethernet/replay/replay_journal.pkl
```

### 绘制 Learner 指标曲线

```bash
python scripts/tools/plot_learner_metrics.py \
  --run_dir runs/agilex_ethernet
```

### W&B 实时 streaming

```bash
python scripts/stream_learner_metrics_to_wandb.py \
  --run_dir runs/agilex_ethernet
```

### 离线 Replay 训练

参见 `rlt_online_rl/scripts/offline/README.md`。

---

## 六、数据流总结图

```
┌──────────────────────────────────────────────────────────┐
│                    Robot Rollout (B4)                     │
│  obs → Machine A → (z_rl, ref_chunk)                      │
│                           │                               │
│          ┌────────────────┴──────────────────┐            │
│          │ warmup/full-task prefix            │ online     │
│          │ execute ref_chunk (BASE)           │ query B1   │
│          └────────────────┬──────────────────┘            │
│                           │ executed_chunk                 │
│                           ▼                               │
│              robot.step(chunk) → reward, done             │
│                           │                               │
│                   RawEpisodeStep 记录                      │
│                           │ episode end                   │
│                           ▼                               │
│              构建 RLTTransition → B3 replay_manager        │
└──────────────────────────────────────────────────────────┘
                            │
                     add_transitions
                            │
┌───────────────────────────▼──────────────────────────────┐
│                   B3 ReplayManager                        │
│  append-only journal + 分层采样缓冲区                      │
└───────────────────────────┬──────────────────────────────┘
                            │ sample_batch
                            │
┌───────────────────────────▼──────────────────────────────┐
│                   B2 LearnerService                       │
│  train_step (JAX JIT):                                    │
│    update_critic → update_actor (每 2 步)                  │
│    soft_update target networks (τ=0.005)                  │
│    → export actor_snapshot.pkl                            │
└───────────────────────────┬──────────────────────────────┘
                            │ poll snapshot (0.25s)
                            │
┌───────────────────────────▼──────────────────────────────┐
│                   B1 ActorService                         │
│  hot-load new actor_params → serve /infer                 │
└──────────────────────────────────────────────────────────┘
```

---

## 七、常见问题

| 现象 | 原因 | 解决方法 |
|------|------|----------|
| Learner 一直等待 warmup | `replay_size < warmup_min_size` | 多做几个 warmup episode |
| Online 模式迟迟不切换 | warmup updates 未完成 | 查看 `learner_status.json` 中 `ready_for_online` |
| Actor 推理超时回退 `ref_chunk` | actor_service 未加载 snapshot | 等待第一次 snapshot 写入 |
| full_task 前缀不写 replay | 设计如此 | 只有 critical phase 数据进 replay |
| eval 时 `s` 键不计入奖励 | eval rollout 不训练 | `s` 仅用于结束 episode |
