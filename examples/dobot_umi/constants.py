# constants.py — Dobot CR / MG 系列机械臂 + 知行夹爪 常量定义
# ruff: noqa

# ─────────────────────────────────────────────────────────────
# 控制周期
# ─────────────────────────────────────────────────────────────
DT = 0.02  # 50 Hz 控制频率（秒/步）

# ─────────────────────────────────────────────────────────────
# 越疆 Dobot SDK 直驱参数（TCP/IP）
# ─────────────────────────────────────────────────────────────
DOBOT_IP               = "192.168.5.1"    # 机械臂以太网 IP
DOBOT_DASHBOARD_PORT   = 29999
DOBOT_FEEDBACK_PORT    = 30004
DOBOT_USE_FEEDBACK     = True             # 是否启动反馈线程（获取关节角）

# ─────────────────────────────────────────────────────────────
# 知行夹爪 SDK 直驱参数（RS-485 串口）
# ─────────────────────────────────────────────────────────────
GRIPPER_SERIAL_PORT    = "/dev/ttyUSB1"   # USB-Serial 设备节点；请按当前实际枚举结果修改
GRIPPER_SLAVE_ID       = 1
GRIPPER_BAUDRATE       = 115200
GRIPPER_SPEED_PCT      = 5                # 运动速度百分比（1–100）
GRIPPER_FORCE_PCT      = 50              # 夹持力百分比（1–100）
# 夹爪内部编码器位置（对应物理行程）
GRIPPER_POS_OPEN       = 0               # 完全张开
GRIPPER_POS_CLOSE      = 12000           # 完全闭合

# ─────────────────────────────────────────────────────────────
# 机械臂复位姿态（关节角度，单位：弧度）
# 请根据实际任务场景调整该值
# ─────────────────────────────────────────────────────────────
DEFAULT_RESET_JOINT_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# ─────────────────────────────────────────────────────────────
# 知行夹爪（Zhixing Gripper）物理行程
# 夹爪开合距离以米（m）为单位，策略动作的第 7 维
# ─────────────────────────────────────────────────────────────
GRIPPER_OPEN_M   = 0.085   # 完全打开时的指间距离（m）
GRIPPER_CLOSE_M  = 0.000   # 完全关闭时的指间距离（m）

# 归一化函数（将物理距离映射到 [0, 1]）
GRIPPER_NORMALIZE_FN   = lambda x: (x - GRIPPER_CLOSE_M) / (GRIPPER_OPEN_M - GRIPPER_CLOSE_M)
GRIPPER_UNNORMALIZE_FN = lambda x: x * (GRIPPER_OPEN_M - GRIPPER_CLOSE_M) + GRIPPER_CLOSE_M

# ─────────────────────────────────────────────────────────────
# RealSense 相机（SDK 直驱，pyrealsense2）
# 用 rs-enumerate-devices 或 check_hardware.py 获取序列号
# ─────────────────────────────────────────────────────────────
REALSENSE_FRONT_SERIAL  = "341522300463"   # 正面相机序列号（请按实际填写）
REALSENSE_WRIST_SERIAL  = "427622270458"   # 腕部相机序列号（请按实际填写；为空时自动选第二台）
REALSENSE_WIDTH         = 640
REALSENSE_HEIGHT        = 480
REALSENSE_FPS           = 30

# ─────────────────────────────────────────────────────────────
# VitAI GF225 触觉传感器（pyvitaisdk）
# ─────────────────────────────────────────────────────────────
TACTILE_LEFT_SERIAL     = "GF22511615AAF"
TACTILE_RIGHT_SERIAL    = "GF22513812AF6"

# ─────────────────────────────────────────────────────────────
# ROS 话题名称（仅 UMI 设备走 ROS；相机已改为 SDK 直驱）
# ─────────────────────────────────────────────────────────────
# RealSense 相机话题（保留，供需要 ROS 桥接的场景使用）
CAM_FRONT_TOPIC         = "/cam_front/color/image_raw"   # 正面/全局视角
CAM_WRIST_TOPIC         = "/cam_wrist/color/image_raw"   # 腕部视角

# UMI 人为介入：VIO 末端位姿话题（PoseStamped，UMI 世界坐标系）
UMI_VIO_POSE_TOPIC      = "/umi1/vio/pose"

# UMI 人为介入：（旧版）直接发布 7D 关节动作的话题（向后兼容，不再推荐）
UMI_HUMAN_ACTION_TOPIC  = "/umi/human_action"

# UMI 触发：切换策略/手动接管的 ROS 服务名
UMI_TELEOP_TRIGGER_SERVICE = "/umi/teleop_trigger"

# ─────────────────────────────────────────────────────────────
# 策略观测输入键名（须与 Pi0 / openpi-server 端保持一致）
# ─────────────────────────────────────────────────────────────
IMAGE_KEY_FRONT = "cam_front"
IMAGE_KEY_WRIST = "cam_wrist"
IMAGE_KEY_TACTILE_LEFT = "tactile_left"
IMAGE_KEY_TACTILE_RIGHT = "tactile_right"
