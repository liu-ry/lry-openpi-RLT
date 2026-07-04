# MarvinSDK 高级 API 文档

## 简介

`MarvinSDK` 是天机-孚晞 MARVIN 机器人控制与运动学 SDK 的 Python 高级封装。它整合了机器人连接、状态查询、运动控制、阻抗/力控模式和轨迹规划，并通过简化接口降低使用门槛。

## 核心特性

- 支持左右臂分别控制：`A`, `B`, `AB`
- 自动异常抛出，失败时直接返回 `Exception`
- 内置运动学初始化与笛卡尔/关节规划
- 支持阻塞与非阻塞运动
- 提供位置模式、阻抗模式、力控模式、拖动示教、离线 PVT 等功能

## 快速开始

```python
from marvin_api import MarvinSDK

sdk = MarvinSDK(ip='192.168.1.190')

sdk.connect()

# 可选：如果需要重新切换运动学配置
sdk.init_kinematics('path/to/ccs_m6.MvKDCfg')
```

> 构造函数默认会加载包内 `ccs_m6_40.MvKDCfg`，如果需要指定其它文件请传入 `config_path`。

---

## 核心接口

### 连接与释放

- `connect()`
- `release()`

### 状态查询

- `get_info()`
- `get_current_joints(arm)`
- `get_current_tcppose(arm)`
- `fk(joints, arm='A')`
- `ik(arm, pose, ref_joints=None)`

### 运动与控制

- `soft_stop(arm='AB')`
- `get_servo_error_code(arm=None, lang='CN')`
- `servo_reset(arm=None, axis=None)`
- `set_tool(arm=None, kine_para=None, dyn_para=None)`
- `set_position_state(arm=None, vel_ratio=10, acc_ratio=10)`
- `set_imp_joint_state(arm=None, vel_ratio=10, acc_ratio=10, K=None, D=None)`
- `set_imp_cart_state(arm=None, vel_ratio=10, acc_ratio=10, K=None, D=None, rot_type=0, cart_ctrl_para=None)`
- `set_imp_force_state(arm=None, fx_dir=None, fc_adj_lmt=10.0)`
- `disable(arm=None)`
- `set_joint_position_cmd(arm, joints)`
- `set_force_cmd(arm, force)`
- `send_pvt(arm, local_file, serial)`
- `run_pvt(arm, id_)`
- `set_joint_drag(arm)`
- `set_cart_drag(arm, direction)`
- `exit_drag(arm)`
- `movej(arm, end_joints, vel_ratio=0.1, acc_ratio=0.1, blocking=True)`
- `movel(arm, end_pose, vel=10, acc=10, blocking=True)`
- `stop_pln(arm)`

---

## API 详解

### `MarvinSDK(ip, config_path=None)`

- `ip`: 机器人 IP 地址
- `config_path`: 可选配置文件路径，默认加载包内 `ccs_m6_40.MvKDCfg`
- 构造函数会自动初始化运动学模块，如加载失败会抛出异常。

### `connect()`

- 建立机器人连接
- 检查伺服错误并尝试清错
- 验证 UDP 数据通道是否正常
- 成功后自动初始化规划模块

### `release()`

- 释放机器人连接资源

### `get_info()`

- 返回当前订阅的机器人状态数据

### `get_current_joints(arm)`

- `arm`: 'A' 或 'B'
- 返回当前 7 关节角度列表

### `get_current_tcppose(arm)`

- `arm`: 'A' 或 'B'
- 返回当前末端 TCP 位姿 `[x, y, z, a, b, c]`

### `fk(joints, arm='A')`

正向运动学：从关节角度计算末端位姿

- `joints`: 关节角度列表，7 个值（度）
- `arm`: 'A' 或 'B'，默认 'A'
- 返回：末端位姿列表 `[x, y, z, a, b, c]` (mm 和度)

### `ik(pose, arm='A', ref_joints=None)`

逆向运动学：从末端位姿计算关节角度

- `pose`: 末端位姿列表 `[x, y, z, a, b, c]` (mm 和度)
- `arm`: 'A' 或 'B'，默认 'A'
- `ref_joints`: 参考关节角度，7 个值（度），可选，用于约束解的构型
- 返回：关节角度列表（7 个值，度）

### `soft_stop(arm='AB')`

- `arm`: 'A', 'B' 或 'AB'
- 执行软急停

### `get_servo_error_code(arm=None, lang='CN')`

- `arm`: 'A', 'B' 或 `None`
- `lang`: 'CN' 或 'EN'
- `None` 时返回 `{'A': ..., 'B': ...}`

### `servo_reset(arm=None, axis=None)`

- `arm`: 'A', 'B' 或 `None`
- `axis`: 0-6 或 `None`

### `set_tool(arm=None, kine_para=None, dyn_para=None)`

- `kine_para`: 长度 6，默认 `[0,0,0,0,0,0]`
- `dyn_para`: 长度 10，默认 `[0,...,0]`

### `set_position_state(arm=None, vel_ratio=10, acc_ratio=10)`

- 切换到位置控制模式

### `set_imp_joint_state(arm=None, vel_ratio=10, acc_ratio=10, K=None, D=None)`

- `K`: 7 个刚度系数，默认 `[2,2,2,1,1,1,1]`
- `D`: 7 个阻尼系数，默认 `[0.6,0.6,0.6,0.4,0.2,0.2,0.2]`

### `set_imp_cart_state(arm=None, vel_ratio=10, acc_ratio=10, K=None, D=None, rot_type=0, cart_ctrl_para=None)`

- `K`: 7 个刚度系数，默认 `[1000,1000,1000,50,50,50,10]`
- `D`: 7 个阻尼系数，默认 `[0.6,0.6,0.6,0.3,0.3,0.3,0.3]`
- `rot_type`: 0/1/2
- `cart_ctrl_para`: 7 个数值，默认全 0

### `set_imp_force_state(arm=None, fx_dir=None, fc_adj_lmt=10.0)`

- `fx_dir`: 6 维力控方向，默认 `[0,0,1,0,0,0]`
- `fc_adj_lmt`: 调节范围，单位 mm

### `disable(arm=None)`

- 对指定或两臂进行复位/下使能

### `set_joint_position_cmd(arm, joints)`

- `arm`: 'A' 或 'B'
- `joints`: 长度 7，单位度

### `set_force_cmd(arm, force)`

- `arm`: 'A' 或 'B'
- `force`: 目标力或力矩

### `send_pvt(arm, local_file, serial)`

- 上传本地 PVT 轨迹
- `serial`: 0-99

### `run_pvt(arm, id_)`

- 运行指定 PVT 轨迹 ID

### `set_joint_drag(arm)`

- 进入关节拖动示教模式

### `set_cart_drag(arm, direction)`

- `direction`: 'X', 'Y', 'Z', 或 'R'

### `exit_drag(arm)`

- 退出拖动模式

### `movej(arm, end_joints, vel_ratio=0.1, acc_ratio=0.1, blocking=True)`

- `end_joints`: 目标关节角度，长度 7
- `vel_ratio`, `acc_ratio`: 0-1，默认 0.1 = 10%
- `blocking`: 是否阻塞等待完成
- 起点由当前关节位置自动获取

### `movel(arm, end_pose, vel=10, acc=10, blocking=True)`

- `end_pose`: 目标末端位姿 `[x, y, z, a, b, c]`
- `vel`: 末端规划速度（mm/s）
- `acc`: 末端规划加速度（mm/s^2）
- 起点由当前位姿自动计算

### `stop_pln(arm)`

- 停止指定手臂的规划运动

### `init_kinematics(config_path)`

- 重新加载运动学配置并初始化规划

---

## 使用示例

### 基础关节运动

```python
sdk = MarvinSDK(ip='192.168.1.190')
sdk.connect()

sdk.set_position_state(arm='A', vel_ratio=20, acc_ratio=20)

sdk.movej(
    arm='A',
    end_joints=[30, -60, 60, 0, 0, 0, 0],
    vel_ratio=0.1,
    acc_ratio=0.1,
    blocking=True
)

sdk.release()
```

### 笛卡尔直线运动

```python
sdk = MarvinSDK(ip='192.168.1.190')
sdk.connect()

sdk.movel(
    arm='A',
    end_pose=[100, 200, 500, 0, 90, 0],
    vel=50,
    acc=50,
    blocking=True
)
```

### 力控应用

```python
sdk.set_imp_force_state(arm='A', fx_dir=[0, 0, 1, 0, 0, 0], fc_adj_lmt=10.0)
sdk.set_force_cmd(arm='A', force=10.0)
```

---

## 异常处理

所有接口在失败时会抛出 `Exception`，建议使用 `try/except` 捕获：

```python
try:
    sdk.movej(arm='A', end_joints=[...])
except Exception as e:
    print(f"操作失败: {e}")
```

---

## 参数说明

| 参数 | 说明 | 单位 |
|------|------|------|
| `vel_ratio` | 运动速度比例 | 0-1 |
| `acc_ratio` | 加速度比例 | 0-1 |
| `joints` | 关节角度列表 | 度 |
| `end_pose` | 末端位姿列表 | mm / 度 |
| `vel` | movel 速度约束 | mm/s |
| `acc` | movel 加速度约束 | mm/s^2 |
| `force` | 力控目标值 | N / N·m |

---

## 常见问题

- `connect()` 失败：检查 IP、网络、防火墙
- `movel()` 失败：确保已调用 `init_kinematics()` 或使用正确配置文件
- `movej()` 失败：检查目标关节角度是否在工作范围内
- `set_tool()` 失败：确保 `kine_para` 长度为 6，`dyn_para` 长度为 10

---

## 版本信息

- `MarvinSDK` 版本: 1.0
- 支持平台: Linux、Windows
- 依赖: `fx_robot.py`, `fx_kine.py`
