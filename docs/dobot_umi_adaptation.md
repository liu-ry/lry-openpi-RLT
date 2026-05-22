# 越疆 Dobot + 知行夹爪 + 双 RealSense + UMI 适配记录

> 文档日期：2026-05-23  
> 分支：main  
> 仓库：lry-openpi-RLT

---

## 一、背景与硬件配置

### 1.1 目标

将越疆（Dobot）CR 系列机械臂 + 知行（Zhixing）电动夹爪 + 两台 Intel RealSense 相机 + UMI 示教设备适配到 `lry-openpi-RLT` 框架，支持：

- **推理部署**：运行 Pi0/Pi0.5 策略控制机械臂；
- **在线 RLT 强化学习**：使用 `rlt_online_rl` 模块在真机上做在线微调。

### 1.2 硬件清单

| 设备 | 型号 / 接口 | 接入方式 |
|------|------------|---------|
| 机械臂 | 越疆 Dobot CR/MG 系列 | **SDK 直驱**（TCP/IP 以太网） |
| 夹爪 | 知行电动夹爪 | **SDK 直驱**（RS-485 串口 `/dev/ttyUSB0`） |
| 正面相机 | Intel RealSense D4xx | ROS 2 话题 `/cam_front/color/image_raw` |
| 腕部相机 | Intel RealSense D4xx | ROS 2 话题 `/cam_wrist/color/image_raw` |
| 示教设备 | UMI 遥操作手柄 | ROS 2 话题 `/umi/human_action` |

### 1.3 混合接入架构说明

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python 进程                               │
│                                                                   │
│  ┌─────────────────────┐     ┌───────────────────────────────┐  │
│  │  DobotSDKArm        │     │  DobotImageRecorder (ROS Node)│  │
│  │  TCP/IP 直驱         │     │  订阅两路 RealSense 话题       │  │
│  │  Dashboard :29999   │     │  /cam_front  /cam_wrist       │  │
│  │  FeedBack  :30004   │     └───────────────────────────────┘  │
│  └─────────┬───────────┘                                         │
│            │ 关节角（弧度）                                        │
│  ┌─────────▼───────────┐     ┌───────────────────────────────┐  │
│  │  ZhixingSDKGripper  │     │  UMIHumanActionRecorder (ROS) │  │
│  │  RS-485 串口直驱     │     │  订阅 /umi/human_action        │  │
│  │  /dev/ttyUSB0       │     └───────────────────────────────┘  │
│  └─────────────────────┘                                         │
└─────────────────────────────────────────────────────────────────┘
          SDK 直驱（无 ROS）          ROS 2 通信（相机 + UMI）
```

**关键设计原则**：
- 机械臂和夹爪**绕过 ROS 驱动**，直接用厂商 SDK 控制（低延迟、减少依赖）；
- 相机和 UMI 设备继续走 ROS 话题（生态成熟、驱动稳定）。

---

## 二、SDK 文件位置

| 文件 | 用途 |
|------|------|
| `third_party/dobot_umi_sdk/dobot_sdk/dobot_api.py` | 越疆官方 TCP/IP SDK（`DobotApiDashboard`, `DobotApiFeedBack`） |
| `third_party/dobot_umi_sdk/adaptive_sdk/changingtek_p_rtu_Servo.py` | 知行夹爪 RS-485 SDK（`MotorController`） |

SDK 文件由 `/home/lry/src/tactile-closed-loop-manipulation` 复制而来，不作修改，仅在运行时注入 `sys.path`。

---

## 三、新增 / 修改文件列表

| 文件路径 | 类型 | 说明 |
|---------|------|------|
| `examples/dobot_umi/constants.py` | 新增 | 全局常量：SDK 连接参数、夹爪行程、ROS 话题名 |
| `examples/dobot_umi/robot_utils.py` | 新增 | 核心硬件抽象：`DobotSDKArm`、`ZhixingSDKGripper`、`DobotImageRecorder`、`UMIHumanActionRecorder` |
| `examples/dobot_umi/real_env.py` | 新增 | `DobotUMIRealEnv`：dm_env 风格真实环境，整合机械臂+夹爪+相机 |
| `examples/dobot_umi/main.py` | 新增 | 推理部署主循环入口 |
| `examples/dobot_umi/env.py` | 新增 | 策略观测/动作的格式化适配（供 openpi-server 调用） |
| `rlt_online_rl/train_deploy_alignment/dobot_umi_ros.py` | 新增 | RLT 在线 RL 机器人侧主循环 |
| `rlt_online_rl/configs/tasks/dobot_umi/online_rl.yaml` | 新增 | 在线 RL 任务配置 |
| `third_party/dobot_umi_sdk/` | 新增 | SDK 文件目录 |
| `docs/dobot_umi_adaptation.md` | 本文档 | 适配记录 |

---

## 四、核心类说明

### 4.1 `DobotSDKArm`（`robot_utils.py`）

越疆机械臂 TCP/IP 直驱控制器。

| 方法 | 说明 |
|------|------|
| `connect()` | 建立 Dashboard（29999）和 FeedBack（30004）TCP 连接，启动反馈线程 |
| `disconnect()` | 停止反馈线程，关闭 TCP 连接 |
| `enable() / disable()` | 机械臂使能/去使能 |
| `clear_error()` | 清除报警 |
| `get_joint_angles_rad()` | 读取 6 关节角（弧度）；优先使用反馈线程，回退 `GetAngle` 查询 |
| `get_robot_mode()` | 读取机器人状态码（5=空闲） |
| `servo_j(joints_rad, t, aheadtime, gain)` | `ServoJ` 伺服跟踪（非阻塞，适合实时控制，内部自动转°） |
| `move_j(joints_rad, wait)` | `MovJ` 关节运动（默认阻塞等待空闲） |
| `stop()` | 停止运动 |

反馈线程解析 `DobotApiFeedBack.feedBackData()` 结构体，持续更新 `_q_actual`（弧度）和 `_robot_mode`。

### 4.2 `ZhixingSDKGripper`（`robot_utils.py`）

知行夹爪 RS-485 串口直驱控制器。

| 方法 | 说明 |
|------|------|
| `init()` | 打开串口，设置速度/力/加减速参数 |
| `set_opening_m(distance_m)` | 设定目标开口距离（m），自动映射到内部编码器位置并触发运动 |
| `open() / close()` | 快捷方法：完全张开（0.085 m）/ 完全关闭（0 m） |
| `stop()` | 就地停止（以当前位置为目标） |
| `get_position_m()` | 读取当前开口距离（m） |
| `set_force(force_pct)` | 动态调整夹持力百分比（1–100） |
| `release()` | 释放串口资源 |

编码器映射：`pos = (1 - distance_m / 0.085) × GRIPPER_POS_CLOSE（12000）`

### 4.3 `DobotImageRecorder`（`robot_utils.py`）

ROS 2 节点，订阅两路 RealSense 相机话题，维护时间戳对齐的图像缓存。

- 订阅话题：`/cam_front/color/image_raw`、`/cam_wrist/color/image_raw`（`sensor_msgs/Image`）
- 核心方法：`get_images(resize_hw, align_timestamps)` → `{"cam_front": ndarray, "cam_wrist": ndarray}`
- `aligned_snapshot()`：取双路时间戳最接近的帧，保证同步性

### 4.4 `UMIHumanActionRecorder`（`robot_utils.py`）

ROS 2 节点，订阅 UMI 设备发布的人类示教动作。

- 订阅话题：`/umi/human_action`（`sensor_msgs/JointState`，`position` 字段为 7D：6 关节角 + 夹爪距离）
- 核心方法：`snapshot_latest()` → `(action_7d, seq_id)`

### 4.5 `DobotUMIRealEnv`（`real_env.py`）

整合所有硬件的 dm_env 风格真实环境。

```python
env = DobotUMIRealEnv()
env.connect()          # 机械臂 connect() + 夹爪 init() + 等待相机就绪
ts  = env.reset()      # move_j 到复位姿态 + 打开夹爪
while True:
    action = policy(ts.observation)   # shape (7,) [joint×6 rad, gripper_m]
    ts = env.step(action)             # servo_j + set_opening_m
env.disconnect()
```

**观测空间**：
```
obs = {
    "qpos":   ndarray (7,)    # [joint1..6 rad, gripper_m]
    "images": {
        "cam_front": ndarray (H, W, 3) uint8
        "cam_wrist": ndarray (H, W, 3) uint8
    }
}
```

**动作空间**：7D `[joint1..6 rad（绝对关节角）, gripper_m（开口距离 m）]`

---

## 五、在线 RL 适配（`dobot_umi_ros.py`）

文件：`rlt_online_rl/train_deploy_alignment/dobot_umi_ros.py`

### 5.1 复用关系

从 `pika_sync_ros.py`（Pika 机器人版本）继承的通用组件：

| 组件 | 说明 |
|------|------|
| `EnvDriver` | 主控循环，驱动观测→推理→执行 |
| `ReplayClient` | 向训练服务器推送 trajectory |
| `ActorClient` | 从策略服务器获取 chunk 动作 |
| `PikaChunkEnvAdapter` | 将 chunk 动作拆包为逐步动作 |

### 5.2 Dobot-UMI 定制类

| 类 | 说明 |
|----|------|
| `DobotSDKArm`（内嵌轻量版） | 与 `robot_utils.py` 同功能，独立实现避免循环导入 |
| `ZhixingSDKGripper`（内嵌轻量版） | 同上 |
| `DobotUMIObsBuffer` | ROS Node，仅订阅双路相机（不订阅关节状态话题） |
| `DobotUMIRobotBridge` | `get_observation()` 从 SDK 直接读关节角 |
| `UMIHumanActionRecorder` | 订阅 `/umi/human_action`（复用 `robot_utils.py` 同名类的逻辑） |
| `TeleopTriggerNode` | ROS Service Server，服务名 `/umi/teleop_trigger` |

### 5.3 启动命令

```bash
python rlt_online_rl/train_deploy_alignment/dobot_umi_ros.py \
    --config rlt_online_rl/configs/tasks/dobot_umi/online_rl.yaml \
    --dobot_ip 192.168.5.1 \
    --dobot_dashboard_port 29999 \
    --gripper_port /dev/ttyUSB0 \
    --gripper_force_pct 50
```

---

## 六、ROS 话题 / SDK 参数汇总

### 6.1 SDK 直驱参数（机械臂 + 夹爪）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DOBOT_IP` | `192.168.5.1` | 机械臂以太网 IP |
| `DOBOT_DASHBOARD_PORT` | `29999` | 控制指令端口 |
| `DOBOT_FEEDBACK_PORT` | `30004` | 反馈数据端口 |
| `GRIPPER_SERIAL_PORT` | `/dev/ttyUSB0` | 夹爪串口设备节点 |
| `GRIPPER_SLAVE_ID` | `1` | Modbus 从机 ID |
| `GRIPPER_BAUDRATE` | `115200` | RS-485 波特率 |
| `GRIPPER_SPEED_PCT` | `5` | 夹爪运动速度 % |
| `GRIPPER_FORCE_PCT` | `50` | 夹爪夹持力 % |
| `GRIPPER_POS_OPEN` | `0` | 编码器位置：完全张开 |
| `GRIPPER_POS_CLOSE` | `12000` | 编码器位置：完全闭合 |

### 6.2 ROS 订阅话题（相机 + UMI）

| 话题 | 消息类型 | 方向 | 说明 |
|------|---------|------|------|
| `/cam_front/color/image_raw` | `sensor_msgs/Image` | 订阅 | 正面 RealSense 相机 |
| `/cam_wrist/color/image_raw` | `sensor_msgs/Image` | 订阅 | 腕部 RealSense 相机 |
| `/umi/human_action` | `sensor_msgs/JointState` | 订阅 | UMI 示教动作（7D） |

### 6.3 ROS 服务（在线 RL）

| 服务 | 类型 | 说明 |
|------|------|------|
| `/umi/teleop_trigger` | `std_srvs/Trigger` | 切换策略/手动接管 |

> **注意**：机械臂关节状态和夹爪控制**不**经过 ROS，无对应话题。

---

## 七、安装与运行

### 7.1 依赖

```bash
# Python
pip install dm-env numpy opencv-python

# ROS 2（相机和 UMI 设备）
# 安装 realsense2_camera 和 cv_bridge 包

# SDK 文件（已在 third_party/dobot_umi_sdk/ 内）
# 无需额外安装，运行时自动注入 sys.path
```

### 7.2 网络配置

机械臂出厂默认 IP `192.168.5.1`，需将运行主机网卡配置到同一子网（如 `192.168.5.100/24`）。

### 7.3 串口权限

```bash
sudo usermod -aG dialout $USER
# 或临时：
sudo chmod 666 /dev/ttyUSB0
```

### 7.4 推理部署

```bash
python examples/dobot_umi/main.py \
    --policy_host <openpi-server-ip> \
    --policy_port 8000
```

---

## 八、硬件连通性检测

插上所有硬件后，**建议先运行连通性检测脚本**，确认每台设备均正常后再启动策略推理或在线 RL。

脚本路径：`examples/dobot_umi/check_hardware.py`

### 8.1 检测项目

| 序号 | 检测项 | 检测内容 | 是否移动硬件 |
|------|--------|---------|------------|
| 1 | 越疆机械臂 | Dashboard TCP 连接、RobotMode、GetAngle、FeedBack 关节角 | **否**（只读） |
| 2 | 知行夹爪 | RS-485 串口初始化、读取编码器位置 | 默认否 |
| 2+ | 知行夹爪（开合测试） | 张开 → 闭合 → 复位（需 `--gripper_test`） | **是** |
| 3 | RealSense 正面相机 | ROS 话题是否有图像帧到达，打印分辨率和编码格式 | 否 |
| 4 | RealSense 腕部相机 | 同上 | 否 |
| 5 | UMI 示教设备 | ROS 话题是否有 7D 动作帧到达（可选） | 否 |

### 8.2 常用命令

```bash
# ── 基础检测（不动任何硬件）
python examples/dobot_umi/check_hardware.py

# ── 同时做夹爪开合测试（会实际移动夹爪，请确认无遮挡）
python examples/dobot_umi/check_hardware.py --gripper_test

# ── UMI 设备未连接时，跳过 UMI 检测
python examples/dobot_umi/check_hardware.py --skip_umi

# ── 修改等待 ROS 话题的超时时间（默认 8s）
python examples/dobot_umi/check_hardware.py --ros_timeout 15

# ── 只测相机和 UMI（跳过机械臂和夹爪）
python examples/dobot_umi/check_hardware.py --skip_arm --skip_gripper

# ── 自定义连接参数
python examples/dobot_umi/check_hardware.py \
    --dobot_ip 192.168.5.1 \
    --gripper_port /dev/ttyUSB0 \
    --gripper_force_pct 30
```

### 8.3 典型输出示例

```
============================================================
  越疆 Dobot / 知行夹爪 / RealSense / UMI  硬件连通性检测
============================================================

[1] 越疆 Dobot 机械臂
   连接 Dashboard 192.168.5.1:29999 ...
✓  Dashboard 连接成功 192.168.5.1:29999
✓  机器人模式: 5 (空闲)
✓  GetAngle 响应: 0,{0.00,0.00,0.00,0.00,0.00,0.00}
   连接 FeedBack 192.168.5.1:30004 ...
✓  FeedBack 连接成功 192.168.5.1:30004
✓  FeedBack 关节角（°）: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
✓  Dobot 机械臂检测通过（未执行任何运动指令）

[2] 知行夹爪（RS-485）
   打开串口 /dev/ttyUSB0（波特率 115200）...
✓  串口 /dev/ttyUSB0 初始化成功
✓  当前编码器位置: 0  →  开口距离: 85.0 mm
⚠  跳过开合运动测试（添加 --gripper_test 可启用）
✓  知行夹爪检测通过

[3] RealSense 相机（ROS 话题）
   等待相机话题（最多 8s）...
✓  /cam_front/color/image_raw  →  640×480  encoding=rgb8
✓  /cam_wrist/color/image_raw  →  640×480  encoding=rgb8

[4] UMI 示教设备（ROS 话题）
   等待 UMI 话题 /umi/human_action（最多 8s）...
✓  /umi/human_action  →  7D 动作: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.085]

============================================================
  检测结果汇总
============================================================
  越疆机械臂  PASS
  知行夹爪    PASS
  RealSense   PASS
  UMI 设备    PASS
============================================================
  所有检测项通过，硬件就绪！
```

### 8.4 常见问题排查

| 现象 | 可能原因 | 解决方法 |
|------|---------|---------|
| Dashboard 连接失败 | 机械臂未上电 / 网线未插 / IP 不对 | 确认网线连通，主机配置 `192.168.5.x/24` 网段 |
| RobotMode = 9（报警） | 碰撞或限位触发 | 手动操作示教器 ClearError，检查姿态 |
| 串口初始化失败 | USB-Serial 设备号变化 / 权限不足 | `ls /dev/ttyUSB*` 确认设备节点；`sudo chmod 666 /dev/ttyUSB0` |
| 相机话题超时 | `realsense2_camera` 节点未启动 | `ros2 launch realsense2_camera rs_launch.py` |
| UMI 话题超时 | UMI 设备未连接或话题名不一致 | 用 `--skip_umi` 跳过，或 `ros2 topic list` 确认话题名 |

---

## 九、已知问题与待办

| 事项 | 状态 |
|------|------|
| `ServoJ` 实时跟踪：需调优 `t / aheadtime / gain` 参数 | 待测 |
| 夹爪编码器物理行程标定（`GRIPPER_POS_CLOSE=12000` 是否准确） | 待测 |
| `DobotApiFeedBack` 结构体格式随固件版本变化，需确认字段名 | 待确认 |
| 双相机时间戳同步精度（`aligned_snapshot` 逻辑） | 待验证 |
| 在线 RL 完整流程测试 | 待测 |
