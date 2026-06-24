# Dobot UMI 客户端硬件接入完整链路

本文档描述从服务端模型训练完成后，客户端介入机械臂、UMI 等硬件的完整链路，涵盖连通性测试、rollout 执行、人工介入、数据落盘等全流程。

---

## 总体架构

```
[Machine A] 训练好的 VLA 策略服务（serve_rlt_policy.py）
     │  WebSocket ws://MACHINE_A_IP:8000
     │  返回：z_rl (2048D) + ref_chunk (10×7)
     ▼
[Machine B / 客户端机器]
  B1 actor_service :9101  ← 轻量 RL actor 推理
  B2 learner_service      ← JAX 训练（Critic + Actor）
  B3 replay_manager :9102 ← Replay 缓冲区 + journal 落盘
  B4 dobot_umi_ros.py     ← 机械臂/夹爪/相机/UMI 硬件驱动 + rollout
```

---

## 阶段一：硬件连通性验证

### 1.1 物理接线确认

| 硬件 | 连接方式 | 默认地址/设备 |
|------|----------|--------------|
| Dobot 机械臂 | 以太网 TCP/IP | `192.168.5.1` dashboard `:29999` / feedback `:30004` |
| 知行夹爪 | RS-485 串口 | `/dev/ttyUSB0` baud=115200 |
| RealSense 正面相机 | USB3 → ROS | `/cam_front/color/image_raw` |
| RealSense 腕部相机 | USB3 → ROS | `/cam_wrist/color/image_raw` |
| UMI 示教设备 | ROS 话题（可选） | `/umi1/vio/pose` |

### 1.2 运行一键连通检测

```bash
cd /path/to/lry-openpi-RLT
source /opt/ros/humble/setup.bash
conda activate rlt_online_rl310

# 基础连通（不动夹爪）
python examples/dobot_umi/check_hardware.py

# 含夹爪开合测试
python examples/dobot_umi/check_hardware.py --gripper_test

# 跳过 UMI（UMI 设备未接时）
python examples/dobot_umi/check_hardware.py --skip_umi
```

**预期通过标准：**

- `✓ Dashboard 连接成功 192.168.5.1:29999`
- `✓ FeedBack 关节角（°）: [...]`（6 个非零数）
- `✓ /cam_front/color/image_raw 收到帧`
- `✓ /cam_wrist/color/image_raw 收到帧`
- 夹爪：`✓ 读取位置成功`

**常见问题排查：**

```bash
# 机械臂 IP 不通
ping 192.168.5.1
ip addr show

# 夹爪串口权限
ls -la /dev/ttyUSB*
sudo chmod 666 /dev/ttyUSB0
# 或加入 dialout 组（重登生效）
sudo usermod -aG dialout $USER

# 相机话题验证
ros2 topic list | grep cam
ros2 topic hz /cam_front/color/image_raw   # 应约 30Hz

# UMI VIO 话题
ros2 topic echo /umi1/vio/pose --once
```

### 1.3 修改默认 IP / 串口（如需）

编辑 `examples/dobot_umi/constants.py`：

```python
DOBOT_IP            = "192.168.5.1"   # ← 改为你的实际 IP
GRIPPER_SERIAL_PORT = "/dev/ttyUSB0"  # ← 改为你的实际串口
```

---

## 阶段二：配置文件适配

### 2.1 检查 dobot_umi 任务配置

```bash
cat rlt_online_rl/configs/tasks/dobot_umi/online_rl.yaml
```

重点核对以下字段并按实际任务调整：

```yaml
experiment:
  rl:
    action_dim: 7          # 6关节 + 1夹爪，固定
    chunk_len: 10          # 每个 chunk 步数
    proprio_dim: 7         # 与 action_dim 一致
    warmup_min_size: 600   # warmup 需要多少条 transition
    warmup_post_collect_updates: 20000  # warmup 后额外训练步数，可调小加速调试
```

### 2.2 生成归一化统计量

如果已有采集数据（LeRobot 格式），先生成 norm stats：

```bash
conda activate rlt_online_rl310
cd rlt_online_rl
python scripts/offline/compute_delta_norm_stats.py \
  --dataset-dir /path/to/your/lerobot_dataset \
  --output-path configs/tasks/dobot_umi/stats/norm_stats_delta.json
```

如果暂无数据，可先用 `agilex_ethernet` 的 stats 占位，后续更新。

---

## 阶段三：确认 Machine A 服务端就绪

确认 Machine A 上的策略服务已在运行：

```bash
# 在 Machine A 上（已训练好的检查点）
conda activate <openpi_env>
python scripts/serve_rlt_policy.py \
  --config rlt_pi05_vitai_lora_finetune \
  --checkpoint-dir checkpoints/pi05_vitai_lora_finetune/vitai_test_1 \
  --port 8000 \
  --shared-prefix-inference

# 从客户端机器验证连通
curl http://MACHINE_A_IP:8000/healthz
```

**本地 debug 无真实 Machine A 时，启动 fake 服务：**

```bash
cd rlt_online_rl
python launch/fake_machine_a.py   # 返回随机 z_rl + ref_chunk
```

---

## 阶段四：启动 Machine B 在线 RL 服务

```bash
cd rlt_online_rl
conda activate rlt_online_rl310

python launch/launch_machine_b.py \
  --config configs/tasks/dobot_umi/online_rl.yaml
```

启动顺序（自动串行）：

1. **B3 replay_manager** (`:9102`) — 先起，等 journal 目录就绪
2. **B2 learner_service** — GPU 进程，等待 replay 数据
3. **B1 actor_service** (`:9101`) — 等待 `/healthz` 就绪后才返回

验证各服务：

```bash
curl http://127.0.0.1:9101/healthz   # actor service
curl http://127.0.0.1:9102/healthz   # replay manager
curl http://127.0.0.1:9101/version   # actor 参数版本号（初始 = -1）
```

---

## 阶段五：启动机器人 Rollout（B4）

```bash
cd rlt_online_rl
source /opt/ros/humble/setup.bash
conda activate rlt_online_rl310

python launch/launch_robot_rollout.py \
  --config configs/tasks/dobot_umi/online_rl.yaml \
  --ros_script dobot_umi \
  --machine_a_ws_url ws://MACHINE_A_IP:8000
```

`--ros_script dobot_umi` 指定使用 `train_deploy_alignment/dobot_umi_ros.py` 作为硬件适配器。

**rollout 脚本会自动等待：**

- Machine A WebSocket 可达
- actor_service `param_version >= 0`（即第一个 snapshot 写入后）

---

## 阶段六：键盘控制 & 人工介入

另开一个终端：

```bash
cd rlt_online_rl
conda activate rlt_online_rl310
python keyboard_toggle_teleop_record_reward_isolation.py
```

### 键位说明

| 键 | 功能 | 使用时机 |
|----|------|----------|
| `o` | **开始新 episode** | 机械臂复位后按 |
| `t` | **切换 teleop 人工接管** | 策略执行失败时接管控制 |
| `c` | 标记进入 critical phase | `full_task` 模式下告知何时开始记 RL 数据 |
| `s` | **标记成功并结束** | 任务完成，reward=1 写入 replay |
| `f` | **标记失败并结束** | 任务失败，reward=0 写入 replay |

### 人工介入（UMI teleop）流程

```
策略执行中
    │
    │ 发现策略即将失败
    ▼
按 [t] → 切换到人工接管
    │  UMI 设备发布 /umi1/vio/pose
    │  dobot_umi_ros.py 的 UMIHumanActionRecorder 接收并转换
    │  机械臂执行人工动作
    ▼
按 [t] → 切回策略控制
    │  intervention_flag=True 的 steps 被单独标记
    │  写入 replay 时 source=HUMAN，BC target=人工动作
    ▼
按 [s] 或 [f] 结束 episode
```

> **注意：** 人工接管期间，replay 记录的是当前控制帧采样到的人工动作，而不是原始 teleop 事件流。人工数据的 BC target 是执行的 `action_chunk`，而非替换 `ref_chunk`，使 actor 学习如何在 VLA 参考基础上做修正。

---

## 阶段七：训练循环 & 数据落盘

### 数据流（episode 结束时自动触发）

```
episode 结束
    │
    ▼
EnvDriver 构建 RLTTransition
    每条包含：z_rl, proprio, ref_chunk, action_chunk,
             reward, done, source, intervention_flag,
             next_z_rl, next_proprio, next_ref_chunk,
             episode_id, step_id, success, collection_phase
    │
    ▼
HTTP POST → B3 replay_manager :9102
    落盘到 runs/dobot_umi/replay/replay_journal.pkl
    （append-only，断点续传安全）
    │
    ▼
replay_size >= warmup_min_size (600)?
    │
    ├─ No  → 继续 warmup episode（纯 ref_chunk 执行）
    │
    └─ Yes → B2 learner 开始训练
                每 500 步写 actor_snapshot.pkl
                B1 actor_service 热加载新参数（0.25s 轮询）
                → rollout 在 critical phase 切入 RL 控制模式
```

### 三个训练阶段说明

| 阶段 | 触发条件 | 控制来源 | 数据写 replay |
|------|----------|----------|--------------|
| **Warmup** | 初始 | 冻结 VLA `ref_chunk` (BASE) | ✅ 全部写入 |
| **Warmup Wait Online** | replay 满 + learner 训练中 | 仍执行 `ref_chunk` | ✅ 继续收集 |
| **Online** | `ready_for_online=true` + `actor_version >= threshold` | critical phase 由 ChunkActor 控制 (RL) | ✅ critical phase 写入 |

> `full_task` 模式下，按 `c` 之前的前缀段执行 `ref_chunk` 但**不写 replay**；`critical_phase` 模式下直接从头进入关键段。

### 落盘目录结构

```
rlt_online_rl/runs/dobot_umi/
├── actor_snapshot/
│   └── actor_snapshot.pkl          ← 最新 actor 参数（热更新用）
├── checkpoints/
│   ├── step_001000/                ← 每 1000 step 存盘
│   └── step_002000/
└── replay/
    └── replay_journal.pkl          ← 全量 transition 流水账（append-only）
```

---

## 阶段八：监控 & 工具

### 检查落盘数据

```bash
# 查看 replay journal 统计
python scripts/tools/inspect_replay_journal.py \
  runs/dobot_umi/replay/replay_journal.pkl

# 查看 learner 训练指标
python scripts/tools/plot_learner_metrics.py \
  --run_dir runs/dobot_umi

# 查看 learner 当前状态
cat runs/dobot_umi/learner_status.json
# 关注：ready_for_online, total_train_steps, replay_size
```

### W&B 实时推送

将 `online_rl.yaml` 中 `wandb_mode` 改为 `online` 后：

```bash
python scripts/stream_learner_metrics_to_wandb.py \
  --run_dir runs/dobot_umi
```

### 离线回放训练（用已有 replay 数据预热）

```bash
python scripts/offline/offline_train_from_replay.py \
  --replay-path runs/dobot_umi/replay/replay_journal.pkl \
  --rl-config-path configs/tasks/dobot_umi/dobot_umi.yaml \
  --steps 5000 \
  --batch-size 128 \
  --bc-weight 2.0 \
  --q-weight 0.1 \
  --output-dir runs/dobot_umi/offline_train \
  --phase all \
  --source all
```

---

## 快速启动顺序速查

```bash
# Terminal 1 — Machine A（策略服务，已训练好的检查点）
python scripts/serve_rlt_policy.py \
  --config rlt_pi05_vitai_lora_finetune \
  --checkpoint-dir checkpoints/pi05_vitai_lora_finetune/vitai_test_1 \
  --port 8000

# Terminal 2 — Machine B 后端服务
cd rlt_online_rl
conda activate rlt_online_rl310
python launch/launch_machine_b.py \
  --config configs/tasks/dobot_umi/online_rl.yaml

# Terminal 3 — 机器人 rollout（ROS 环境）
source /opt/ros/humble/setup.bash
conda activate rlt_online_rl310
python launch/launch_robot_rollout.py \
  --config configs/tasks/dobot_umi/online_rl.yaml \
  --ros_script dobot_umi \
  --machine_a_ws_url ws://MACHINE_A_IP:8000

# Terminal 4 — 键盘控制 & 人工介入
python keyboard_toggle_teleop_record_reward_isolation.py
```

**按键顺序（每个 episode）：**

1. 确认机械臂已复位到初始姿态
2. `o` → 开始新 episode
3. （`full_task` 模式）到达关键操作段前按 `c`
4. 如策略失败，按 `t` 人工接管，纠正后再按 `t` 交还控制权
5. `s` 成功 / `f` 失败 → 结束本 episode，数据自动落盘

---

## 关键判断点汇总

| 状态 | 判断依据 | 处理 |
|------|----------|------|
| 硬件全通 | `check_hardware.py` 全绿 | 继续 |
| Machine A 就绪 | `curl .../healthz` 返回 200 | 继续 |
| actor_service 就绪 | `/version` 返回 `param_version >= 0` | 等待 learner 写出第一个 snapshot |
| warmup 完成 | `learner_status.json` → `ready_for_online: true` | 系统自动切换到 RL 模式 |
| 数据落盘正常 | `replay_journal.pkl` 持续增大 | 正常 |
| 策略无改善 | W&B actor Q 值不增长 | 检查奖励函数 / `s`/`f` 键是否正确使用 |

---

## 调试提速建议

### warmup 阶段加速（前期调试）

临时调小 `online_rl.yaml` 中的值：

```yaml
warmup_min_size: 100              # 100 条就开始训练（默认 600）
warmup_post_collect_updates: 500  # 减少等待步数（默认 20000）
```

### 纯 warmup 推演验证（不进 RL 阶段）

```bash
# 使用 eval-only 模式，只跑 actor 推理，不启动 learner
python launch/launch_actor_eval.py \
  --config configs/tasks/dobot_umi/online_rl.yaml \
  --machine_a_ws_url ws://MACHINE_A_IP:8000

# 对应键盘客户端
python keyboard_actor_eval.py
```

Eval 键盘额外按键：

| 键 | 功能 |
|----|------|
| `a` | 下一个 episode 的 critical phase 使用 **actor** |
| `b` | 下一个 episode 的 critical phase 使用 **base（VLA ref）** |
| `s` | 结束/重置 episode（不计入训练奖励） |
