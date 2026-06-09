#!/usr/bin/env python3
"""dobot_umi_ros.py — 越疆 Dobot 机械臂 + 知行夹爪 + 双 RealSense 相机 + UMI 人为介入 的 ROS 适配器。

本脚本是 pika_sync_ros.py 的硬件定制版本，主要变更：
  1. ROSObsBuffer → DobotUMIObsBuffer：仅订阅两路 RealSense 相机（cam_front / cam_wrist），
     不再订阅关节状态话题（关节角通过 SDK 反馈线程直接读取）。
  2. 越疆机械臂：SDK 直驱（TCP/IP，DobotApiDashboard + DobotApiFeedBack），
     不依赖 dobot_bringup ROS 驱动。
  3. 知行夹爪：SDK 直驱（RS-485 串口，MotorController），
     不依赖 data_msgs/Gripper ROS 话题。
  4. HumanActionRecorder → UMIHumanActionRecorder：订阅 UMI 设备发布的示教动作话题
     /umi/human_action（UMI 设备仍走 ROS 话题）。
  5. TeleopTriggerNode 默认服务名改为 /umi/teleop_trigger。
  6. get_observation() 对外统一返回 cam_front / cam_wrist 图像键。
  7. 默认配置文件指向 configs/tasks/dobot_umi/online_rl.yaml。

所有通用的 RLT 在线训练逻辑（EnvDriver、ReplayClient、ActorClient、
PikaChunkEnvAdapter 等）直接从 pika_sync_ros.py 导入复用，不重复实现。
"""
from __future__ import annotations

import argparse
from collections import deque
import dataclasses
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from geometry_msgs.msg import PoseStamped as ROSPoseStamped
from sensor_msgs.msg import Image as ROSImage
from std_srvs.srv import Trigger

try:
    from scipy.spatial.transform import Rotation as _Rotation
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    _Rotation = None  # type: ignore

# ── 将 pika_sync_ros 所在目录加入路径 ────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
ADAPTER_DIR = Path(__file__).resolve().parent
for _p in (str(SRC_ROOT), str(ADAPTER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Dobot / 知行夹爪 SDK ──────────────────────────────────────────────────────
_REPO_ROOT_OUTER = Path(__file__).resolve().parents[2]
_SDK_ROOT = _REPO_ROOT_OUTER / "third_party" / "dobot_umi_sdk"
# 将 dobot_sdk/ 和 adaptive_sdk/ 子目录直接加入 sys.path，
# 这样可以用 `from dobot_api import ...` 和 `from changingtek_p_rtu_Servo import ...` 导入。
for _p in (str(_SDK_ROOT / "dobot_sdk"), str(_SDK_ROOT / "adaptive_sdk")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── examples/dobot_umi 工具包（RealSenseImageRecorder 等）───────────────────
# robot_utils.py 内部用 `from examples.dobot_umi import constants`，
# 因此需要把仓库根目录（而不是 examples/ 子目录）加入 sys.path。
if str(_REPO_ROOT_OUTER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_OUTER))

try:
    from examples.dobot_umi.robot_utils import RealSenseImageRecorder  # type: ignore[import]
    _HAS_REALSENSE_RECORDER = True
except ImportError as _e_realsense:
    import traceback as _tb
    _tb.print_exc()
    _HAS_REALSENSE_RECORDER = False
    RealSenseImageRecorder = None  # type: ignore[assignment,misc]

try:
    from dobot_api import DobotApiDashboard, DobotApiFeedBack  # type: ignore[import]
    _HAS_DOBOT_SDK = True
except ImportError:
    _HAS_DOBOT_SDK = False

try:
    from changingtek_p_rtu_Servo import MotorController  # type: ignore[import]
    _HAS_MOTOR_SDK = True
except ImportError:
    _HAS_MOTOR_SDK = False

# ── 自动 source ROS2 工作空间（获取 common_msgs 等消息定义）────────────────
def _source_ros_workspace(setup_bash: str) -> bool:
    """运行 `bash -c 'source <setup_bash> && env -0'`，将新增的环境变量写入
    当前进程的 os.environ，并同步 PYTHONPATH 到 sys.path。
    返回 True 表示成功，False 表示文件不存在或执行失败。
    """
    import subprocess
    setup_path = Path(setup_bash).expanduser().resolve()
    if not setup_path.exists():
        return False
    try:
        result = subprocess.run(
            ["bash", "-c", f"source {setup_path} && env -0"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return False
        for item in result.stdout.split("\0"):
            if "=" in item:
                k, _, v = item.partition("=")
                if k in ("PYTHONPATH", "AMENT_PREFIX_PATH", "LD_LIBRARY_PATH",
                         "PATH", "AMENT_CURRENT_PREFIX", "ROS_DISTRO"):
                    os.environ[k] = v
        for p in os.environ.get("PYTHONPATH", "").split(":"):
            if p and p not in sys.path:
                sys.path.insert(0, p)
        return True
    except Exception as _e:
        logging.getLogger(__name__).warning("[_source_ros_workspace] 失败: %s", _e)
        return False

# 尝试导入 common_msgs；失败时按候选路径顺序自动 source 后重试。
# 候选优先级：
#   ① 仓库内自带（third_party/ros2_msgs_ws）—— 首次使用需运行：
#        bash third_party/ros2_msgs_ws/build_msgs.sh
#   ② 外部 handheld-umi_ws（兼容旧环境，不强依赖）
_UMI_WS_CANDIDATES = [
    _REPO_ROOT_OUTER / "third_party" / "ros2_msgs_ws" / "install" / "setup.bash",
    _REPO_ROOT_OUTER.parents[1] / "handheld-umi_ws" / "install" / "setup.bash",
    Path.home() / "handheld-umi_ws" / "install" / "setup.bash",
    Path("/opt/handheld-umi_ws/install/setup.bash"),
]
try:
    from common_msgs.msg import EncoderState as UMIEncoderState  # type: ignore[import]
    _HAS_ENCODER_STATE = True
except ImportError:
    _HAS_ENCODER_STATE = False
    UMIEncoderState = None  # type: ignore[assignment,misc]
    for _candidate in _UMI_WS_CANDIDATES:
        if _source_ros_workspace(str(_candidate)):
            logging.getLogger(__name__).info("[init] 已自动 source %s", _candidate)
            try:
                from common_msgs.msg import EncoderState as UMIEncoderState  # type: ignore[import]
                _HAS_ENCODER_STATE = True
            except ImportError:
                pass
            break
    if not _HAS_ENCODER_STATE:
        logging.getLogger(__name__).warning(
            "[init] common_msgs 不可用。\n"
            "  请先运行：bash third_party/ros2_msgs_ws/build_msgs.sh"
        )

import re as _re
_RESP_PAT = _re.compile(r"(-?\d+),?\{?([\d.,\-\s]*)\}?")

def _parse_dobot_resp(resp: str):
    if not resp:
        return None, None
    try:
        resp = resp.strip().rstrip(";")
        m = _RESP_PAT.match(resp)
        if m:
            code = int(m.group(1))
            vs = m.group(2)
            vals = [float(v.strip()) for v in vs.split(",") if v.strip()] if vs else None
            return code, vals
        if resp.lstrip("-").isdigit():
            return int(resp), None
    except Exception:
        pass
    return None, None

from openpi_client import image_tools  # noqa: E402

from rlt_online_rl.config import OnlineRLSystemConfig, load_system_config_yaml  # noqa: E402
from rlt_online_rl.inference import (  # noqa: E402
    ActorClient,
    ChunkFeatures,
    EnvDriver,
    MachineAFeatureClient,
    PolicyPlan,
)
from rlt_online_rl.replay import NullReplayClient, ReplayClient  # noqa: E402
from rlt_online_rl.runtime_logging import metrics_path_for, setup_process_logging  # noqa: E402

# ── 从 pika_sync_ros 复用通用 RLT 组件（避免重复实现） ───────────────────────
from pika_sync_ros import (  # noqa: E402
    HumanInterventionState,
    PikaChunkEnvAdapter,
    PhaseAwareActorClient,
    RolloutPhaseController,
    RolloutRuntimeContext,
    StaticOnlinePhaseController,
    TaskState,
    TeleopTriggerNode,
    _bind_runtime_hook,
    _build_obs_subscription_qos,
    _coerce_reward_output,
    _default_done_fn,
    _default_reward_fn,
    _default_success_fn,
    _image_msg_to_rgb_u8_hwc,
    _load_callable,
    _make_learner_status_reader,
    _missing_observation_fields,
    _override_system_urls,
    _resolve_min_online_actor_version,
    _ros_stamp_to_sec,
)
from manual_signal_bridge import (  # noqa: E402
    ENTER_CRITICAL_PHASE_SERVICE,
    RECORD_DONE_SERVICE,
    RECORD_FAILURE_SERVICE,
    RECORD_SUCCESS_SERVICE,
    REQUEST_NEXT_EPISODE_SERVICE,
    SET_CRITICAL_POLICY_ACTOR_SERVICE,
    SET_CRITICAL_POLICY_BASE_SERVICE,
    TOGGLE_CRITICAL_PHASE_SERVICE,
    ManualSignalBridge,
)

logger = logging.getLogger("dobot_umi_ros")

DEFAULT_CONFIG = REPO_ROOT / "configs" / "tasks" / "dobot_umi" / "online_rl.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# 默认参数
# ─────────────────────────────────────────────────────────────────────────────
# 越疆 Dobot SDK（TCP/IP）
DEFAULT_DOBOT_IP             = "192.168.5.1"
DEFAULT_DOBOT_DASHBOARD_PORT = 29999
DEFAULT_DOBOT_FEEDBACK_PORT  = 30004

# 知行夹爪 SDK（RS-485 串口）
DEFAULT_GRIPPER_PORT      = "/dev/ttyUSB0"
DEFAULT_GRIPPER_SLAVE_ID  = 1
DEFAULT_GRIPPER_BAUDRATE  = 115200
DEFAULT_GRIPPER_SPEED_PCT = 30
DEFAULT_GRIPPER_FORCE_PCT = 50
DEFAULT_GRIPPER_POS_CLOSE = 12000   # 编码器完全闭合位置

# RealSense 相机 ROS 话题
DEFAULT_CAM_FRONT_TOPIC    = "/cam_front/color/image_raw"
DEFAULT_CAM_WRIST_TOPIC    = "/cam_wrist/color/image_raw"

# UMI 设备（ROS 话题）
DEFAULT_UMI_VIO_POSE_TOPIC     = "/umi1/vio/pose"           # VIO 末端位姿（PoseStamped）
DEFAULT_UMI_ACTION_TOPIC       = "/umi/human_action"         # 旧版 7D 关节动作（向后兼容）
DEFAULT_TELEOP_TRIGGER_SVC     = "/umi/teleop_trigger"
DEFAULT_UMI_ENCODER_TOPIC      = "/umi1/vitai/encoder_state" # UMI 手柄夹爪编码器话题

# 夹爪物理行程（m）
GRIPPER_MAX_M = 0.085


def _gripper_travel_to_m(value: float, *, max_gripper_m: float = GRIPPER_MAX_M) -> float:
    """Convert model/replay gripper travel units to physical opening in meters."""
    travel = float(np.clip(value, 0.0, DEFAULT_GRIPPER_POS_CLOSE))
    ratio_closed = travel / max(float(DEFAULT_GRIPPER_POS_CLOSE), 1.0)
    return float(np.clip((1.0 - ratio_closed) * max_gripper_m, 0.0, max_gripper_m))


def _gripper_m_to_travel(value: float, *, max_gripper_m: float = GRIPPER_MAX_M) -> float:
    """Convert physical opening in meters to model/replay gripper travel units."""
    opening_m = float(np.clip(value, 0.0, max_gripper_m))
    ratio_closed = 1.0 - opening_m / max(float(max_gripper_m), 1e-6)
    return float(np.clip(ratio_closed * DEFAULT_GRIPPER_POS_CLOSE, 0.0, DEFAULT_GRIPPER_POS_CLOSE))


# ─────────────────────────────────────────────────────────────────────────────
# 两路相机观测缓冲节点（ROS 话题；关节角由 SDK 直接读取）
# ─────────────────────────────────────────────────────────────────────────────

class DobotUMIObsBuffer(Node):
    """订阅两路 RealSense 图像，缓存最新帧。

    关节角不再通过 ROS 话题订阅，而是通过 DobotSDKArm.get_joint_angles_rad() 读取。
    """

    def __init__(
        self,
        *,
        cam_front_topic: str,
        cam_wrist_topic: str,
        sync_queue_size: int = 200,
        sub_qos: QoSProfile | None = None,
    ):
        super().__init__("dobot_umi_obs_buffer")
        self._lock = threading.Lock()
        self._front_msg: ROSImage | None = None
        self._wrist_msg: ROSImage | None = None
        qs = max(int(sync_queue_size), 2)
        self._front_queue: deque[ROSImage] = deque(maxlen=qs)
        self._wrist_queue: deque[ROSImage] = deque(maxlen=qs)
        self._last_wait_log_ts = 0.0
        qos = sub_qos or qos_profile_sensor_data

        self.create_subscription(ROSImage, cam_front_topic, self._on_front, qos)
        self.create_subscription(ROSImage, cam_wrist_topic, self._on_wrist, qos)

    def _on_front(self, msg: ROSImage) -> None:
        with self._lock:
            self._front_msg = msg
            self._front_queue.append(msg)

    def _on_wrist(self, msg: ROSImage) -> None:
        with self._lock:
            self._wrist_msg = msg
            self._wrist_queue.append(msg)

    def snapshot(self) -> tuple[ROSImage | None, ROSImage | None]:
        with self._lock:
            return self._front_msg, self._wrist_msg

    @staticmethod
    def _stamp(msg) -> float:
        return _ros_stamp_to_sec(getattr(msg.header, "stamp", None))

    def aligned_snapshot(self) -> tuple[ROSImage, ROSImage] | None:
        with self._lock:
            if not (self._front_queue and self._wrist_queue):
                return None
            t_ref = min(
                self._stamp(self._front_queue[-1]),
                self._stamp(self._wrist_queue[-1]),
            )
            aligned = []
            for q in (self._front_queue, self._wrist_queue):
                while len(q) > 1 and self._stamp(q[0]) < t_ref:
                    q.popleft()
                if self._stamp(q[0]) < t_ref:
                    return None
                aligned.append(q[0])
            return aligned[0], aligned[1]

    def _missing(self) -> list[str]:
        f, w = self.snapshot()
        missing = []
        if f is None: missing.append("cam_front")
        if w is None: missing.append("cam_wrist")
        return missing

    def wait_ready(self, timeout_s: float | None = None) -> None:
        start = time.time()
        while rclpy.ok():
            missing = self._missing()
            if not missing:
                return
            now = time.time()
            if now - self._last_wait_log_ts >= 2.0:
                self._last_wait_log_ts = now
                self.get_logger().warning(f"等待相机话题，缺少: {missing}")
            if timeout_s is not None and (now - start) > timeout_s:
                raise RuntimeError(f"超时：等待相机话题，缺少: {missing}")
            time.sleep(0.02)


# ─────────────────────────────────────────────────────────────────────────────
# 越疆 Dobot 机械臂 SDK 直驱控制器
# ─────────────────────────────────────────────────────────────────────────────

class DobotSDKArm:
    """越疆 Dobot 机械臂 SDK 直驱（TCP/IP）。"""

    def __init__(self, ip: str, dashboard_port: int, feedback_port: int):
        self._ip = ip
        self._dashboard_port = dashboard_port
        self._feedback_port = feedback_port
        self._dashboard = None
        self._feedback = None
        self._connected = False
        self._lock = threading.Lock()
        self._robot_mode = -1
        self._q_actual: list[float] = [0.0] * 6
        self._feed_running = False
        self._feed_thread = None

    @staticmethod
    def _chk(api_obj) -> bool:
        sock = getattr(api_obj, "socket_dobot", None)
        if sock is None or isinstance(sock, int):
            return False
        try:
            sock.getpeername()
            return True
        except Exception:
            return False

    @staticmethod
    def _close(api_obj) -> None:
        try:
            sock = getattr(api_obj, "socket_dobot", None)
            if sock and not isinstance(sock, int):
                sock.close()
                api_obj.socket_dobot = 0
        except Exception:
            pass

    def connect(self) -> None:
        if not _HAS_DOBOT_SDK:
            raise ImportError(
                "[DobotSDKArm] Dobot SDK 未能导入（DobotApiDashboard / DobotApiFeedBack）。\n"
                "请检查 third_party/dobot_umi_sdk/dobot_sdk/dobot_api.py 是否存在，\n"
                "以及 sys.path 中是否包含该目录。当前搜索路径前几项：\n"
                + "\n".join(f"  {p}" for p in sys.path[:8])
            )
        logger.info(f"[DobotSDKArm] 连接 {self._ip}:{self._dashboard_port} ...")
        while True:
            try:
                d = DobotApiDashboard(self._ip, self._dashboard_port)
                if self._chk(d):
                    self._dashboard = d
                    break
                self._close(d)
            except Exception as e:
                logger.warning(f"[DobotSDKArm] 连接异常: {e}")
            time.sleep(3)
        logger.info("[DobotSDKArm] Dashboard 已连接")
        for _ in range(10):
            try:
                fb = DobotApiFeedBack(self._ip, self._feedback_port)
                if self._chk(fb):
                    self._feedback = fb
                    self._feed_running = True
                    self._feed_thread = threading.Thread(target=self._fb_loop, daemon=True)
                    self._feed_thread.start()
                    time.sleep(0.5)
                    logger.info("[DobotSDKArm] FeedBack 已连接")
                    break
                self._close(fb)
            except Exception as e:
                logger.warning(f"[DobotSDKArm] FeedBack 异常: {e}")
            time.sleep(3)
        self._connected = True

    def _fb_loop(self) -> None:
        while self._feed_running and self._feedback is not None:
            try:
                data = self._feedback.feedBackData()
                if data is None:
                    continue
                if hex(data["TestValue"][0]) != "0x123456789abcdef":
                    continue
                with self._lock:
                    self._robot_mode = int(data["RobotMode"][0])
                    # QActual 单位为度（degrees），转换为弧度存储
                    self._q_actual = list(np.deg2rad(data["QActual"][0]).astype(np.float32))
            except Exception:
                time.sleep(0.05)

    def disconnect(self) -> None:
        self._feed_running = False
        if self._feed_thread:
            self._feed_thread.join(timeout=1.0)
        if self._feedback:
            try:
                self._close(self._feedback)
            except Exception:
                pass
        if self._dashboard:
            try:
                self._dashboard.close()
            except Exception:
                pass
        self._connected = False

    def enable(self) -> bool:
        if not self._connected:
            return False
        resp = self._dashboard.EnableRobot()
        ok = _parse_dobot_resp(resp)[0] == 0
        if ok:
            logger.info("[DobotSDKArm] 已使能")
        return ok

    def disable(self) -> bool:
        if not self._connected:
            return False
        return _parse_dobot_resp(self._dashboard.DisableRobot())[0] == 0

    def get_joint_angles_rad(self) -> np.ndarray:
        with self._lock:
            if self._q_actual and abs(sum(self._q_actual)) > 1e-9:
                return np.array(self._q_actual, dtype=np.float32)
        resp = self._dashboard.GetAngle()
        _, vals = _parse_dobot_resp(resp)
        if vals and len(vals) >= 6:
            return np.deg2rad(np.array(vals[:6], dtype=np.float32))
        return np.zeros(6, dtype=np.float32)

    def get_robot_mode(self) -> int:
        with self._lock:
            if self._robot_mode >= 1:
                return self._robot_mode
        code, _ = _parse_dobot_resp(self._dashboard.RobotMode())
        return code if code is not None else -1

    def servo_j(self, joints_rad: np.ndarray, t: float = 0.1) -> bool:
        if not self._connected:
            return False
        jd = np.rad2deg(np.asarray(joints_rad, dtype=np.float64).reshape(-1)[:6])
        # Dobot SDK 文档给出的典型范围是 aheadtime in [20, 100]。
        # 这里随控制周期自适应，并限制在保守范围内。
        aheadtime = float(np.clip(t * 1000.0 * 0.4, 20.0, 40.0))
        resp = self._dashboard.ServoJ(jd[0], jd[1], jd[2], jd[3], jd[4], jd[5], t, aheadtime, 500.0)
        return _parse_dobot_resp(resp)[0] == 0

    def servo_p(
        self,
        x_m: float, y_m: float, z_m: float,
        rx_rad: float, ry_rad: float, rz_rad: float,
        t: float = 0.1,
    ) -> bool:
        """笛卡尔末端位姿伺服（ServoP）。

        Args:
            x_m, y_m, z_m:      末端位置，单位：米（m），内部转换为 mm 发送给 SDK。
            rx_rad, ry_rad, rz_rad: 末端姿态 XYZ Euler 角，单位：弧度（rad），内部转换为度。
            t:                  伺服插补时间（s），建议与控制周期一致。
        Returns:
            True 表示 SDK 返回成功（code==0）。
        """
        if not self._connected:
            return False
        resp = self._dashboard.ServoP(
            x_m * 1000.0, y_m * 1000.0, z_m * 1000.0,
            float(np.rad2deg(rx_rad)),
            float(np.rad2deg(ry_rad)),
            float(np.rad2deg(rz_rad)),
            t, 50.0, 500.0,
        )
        return _parse_dobot_resp(resp)[0] == 0

    def move_j(self, joints_rad: np.ndarray, *, wait: bool = True, timeout: float = 30.0,
               speed_pct: int = -1) -> bool:
        """关节运动（MovJ）。speed_pct: 速度比例 1~100，-1 表示不覆盖（使用控制器默认值）。"""
        if not self._connected:
            return False
        jd = np.rad2deg(np.asarray(joints_rad, dtype=np.float64).reshape(-1)[:6])
        v = int(np.clip(speed_pct, 1, 100)) if speed_pct > 0 else -1
        a = v  # 加速度比例与速度比例保持一致
        resp = self._dashboard.MovJ(jd[0], jd[1], jd[2], jd[3], jd[4], jd[5], 1, v=v, a=a)
        if _parse_dobot_resp(resp)[0] != 0:
            return False
        if not wait:
            return True
        start = time.time()
        while time.time() - start < timeout:
            if self.get_robot_mode() == 5:
                return True
            time.sleep(0.05)
        return False

    def move_l(
        self,
        x_m: float, y_m: float, z_m: float,
        rx_rad: float, ry_rad: float, rz_rad: float,
        *,
        wait: bool = True,
        timeout: float = 30.0,
        speed_pct: int = -1,
    ) -> bool:
        """笛卡尔直线运动（MovL），位置单位 m，姿态单位 rad（XYZ Euler）。
        speed_pct: 速度比例 1~100，-1 表示不覆盖（使用控制器默认值）。
        """
        if not self._connected:
            return False
        v = int(np.clip(speed_pct, 1, 100)) if speed_pct > 0 else -1
        a = v
        resp = self._dashboard.MovL(
            x_m * 1000.0, y_m * 1000.0, z_m * 1000.0,
            float(np.rad2deg(rx_rad)),
            float(np.rad2deg(ry_rad)),
            float(np.rad2deg(rz_rad)),
            0,  # coordinateMode=0: 基座坐标系
            v=v, a=a,
        )
        if _parse_dobot_resp(resp)[0] != 0:
            return False
        if not wait:
            return True
        start = time.time()
        while time.time() - start < timeout:
            if self.get_robot_mode() == 5:
                return True
            time.sleep(0.05)
        return False

    def set_speed_factor(self, speed_pct: int) -> bool:
        """设置全局速度比例（1~100%）。仅影响 MovJ/MovL，不影响 ServoJ。"""
        if not self._connected:
            return False
        speed_pct = int(np.clip(speed_pct, 1, 100))
        resp = self._dashboard.SpeedFactor(speed_pct)
        return _parse_dobot_resp(resp)[0] == 0

    def get_end_effector_pose_matrix(self) -> np.ndarray:
        """通过 SDK GetPose() 获取末端执行器在基座坐标系下的 4×4 齐次变换矩阵。

        返回格式：位置单位为米（m）。
        旋转使用 Dobot 惯例的 XYZ Euler 角（rx/ry/rz，单位 deg）。
        若获取失败则返回单位矩阵。
        """
        if not self._connected or _Rotation is None:
            return np.eye(4, dtype=np.float64)
        try:
            resp = self._dashboard.GetPose()
            _, vals = _parse_dobot_resp(resp)
            if vals and len(vals) >= 6:
                x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = vals[:6]
                R = _Rotation.from_euler("XYZ", [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = [x_mm * 1e-3, y_mm * 1e-3, z_mm * 1e-3]
                return T
        except Exception as e:
            logger.warning(f"[DobotSDKArm] GetPose 失败: {e}")
        return np.eye(4, dtype=np.float64)

    def inverse_kinematics_from_matrix(
        self, T_target: np.ndarray, q_seed: np.ndarray | None = None
    ) -> np.ndarray | None:
        """逆运动学：给定 4×4 目标位姿矩阵，返回 6D 关节角（弧度）。

        调用 Dobot SDK 的 InverseKin 命令。
        传入 q_seed（就近选解）可避免多解跳变，失败时回退到 q_seed 或当前关节角。

        Args:
            T_target:  4×4 目标齐次变换矩阵，位置单位为米（m）。
            q_seed:    IK 初始猜测（6D 弧度），用于就近选解及失败回退。
        Returns:
            6D 关节角（弧度）；失败时返回 q_seed（若有）或当前关节角。
        """
        if not self._connected or _Rotation is None:
            return q_seed.astype(np.float32, copy=True) if q_seed is not None else None
        try:
            x_mm   = float(T_target[0, 3] * 1000.0)
            y_mm   = float(T_target[1, 3] * 1000.0)
            z_mm   = float(T_target[2, 3] * 1000.0)
            euler  = _Rotation.from_matrix(T_target[:3, :3]).as_euler("XYZ", degrees=True)
            rx_deg, ry_deg, rz_deg = float(euler[0]), float(euler[1]), float(euler[2])
            # 注意：Dobot SDK InverseKin 不支持 useJointNear/JointNear 参数，直接调用基础版
            resp = self._dashboard.InverseKin(x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg)
            code, vals = _parse_dobot_resp(resp)
            logger.warning("[DobotSDKArm] InverseKin raw resp=%r  code=%s  vals=%s",
                           resp, code, vals)
            if vals and len(vals) >= 6:
                result = np.deg2rad(np.array(vals[:6], dtype=np.float32))
                logger.warning("[DobotSDKArm] InverseKin 成功: joints_deg=%s",
                               np.round(np.rad2deg(result), 2).tolist())
                return result
            logger.warning("[DobotSDKArm] InverseKin 解析失败（vals=%s），回退到 seed", vals)
        except Exception as e:
            logger.warning(f"[DobotSDKArm] InverseKin 失败（将保持位置）: {e}")
        # 回退：保持种子关节角
        if q_seed is not None:
            return q_seed.astype(np.float32, copy=True)
        return self.get_joint_angles_rad()

    def stop(self) -> bool:
        if not self._connected:
            return False
        return _parse_dobot_resp(self._dashboard.Stop())[0] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 知行夹爪 SDK 直驱控制器（RS-485）
# ─────────────────────────────────────────────────────────────────────────────

class ZhixingSDKGripper:
    """知行夹爪 SDK 直驱（RS-485 串口）。"""

    def __init__(self, port: str, slave_id: int, baudrate: int, speed_pct: int, force_pct: int):
        self._port = port
        self._slave_id = slave_id
        self._baudrate = baudrate
        self._speed_pct = speed_pct
        self._force_pct = force_pct
        self._motor = None
        self._lock = threading.Lock()
        # 上次实际发送的编码器目标位置
        self._last_sent_pos: int | None = None
        # 死区：20 个编码器单位 ≈ 0.14mm，过滤真正的噪声
        self._pos_deadband: int = 20
        # 异步发送：_pending_pos 由回调更新，后台线程以固定 20Hz 发送
        self._pending_pos: int | None = None
        # EMA 平滑目标位置（浮点，发送时取整）；alpha 越小越平滑，0.4 ≈ ~75ms 时间常数@20Hz
        self._ema_pos: float | None = None
        self._ema_alpha: float = 0.4
        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._send_thread.start()

    def init(self) -> bool:
        try:
            self._motor = MotorController(self._port, self._slave_id, self._baudrate, 0.5)
            self._motor.set_target_speed(self._speed_pct)
            self._motor.set_target_force(self._force_pct)
            self._motor.set_target_acceleration(2000)
            self._motor.set_target_deceleration(2000)
            logger.info(f"[ZhixingSDKGripper] 已初始化 {self._port}")
            return True
        except Exception as e:
            logger.error(f"[ZhixingSDKGripper] 初始化失败: {e}")
            return False

    def _send_loop(self) -> None:
        """后台线程：以 20Hz 将 EMA 平滑后的目标位置发送给夹爪。

        关键设计：锁内只做轻量内存计算，Modbus I/O 在锁外执行，
        避免长时间持锁阻塞主线程的 set_opening_m / get_position_m。
        """
        while True:
            time.sleep(0.05)  # 20Hz
            # ── 锁内：只读/写内存变量 ──────────────────────────────────────
            with self._lock:
                if self._motor is None or self._pending_pos is None:
                    continue
                # EMA 平滑
                if self._ema_pos is None:
                    self._ema_pos = float(self._pending_pos)
                else:
                    self._ema_pos = (self._ema_alpha * self._pending_pos
                                     + (1.0 - self._ema_alpha) * self._ema_pos)
                pos = int(round(self._ema_pos))
                # 死区过滤
                if (self._last_sent_pos is not None
                        and abs(pos - self._last_sent_pos) < self._pos_deadband):
                    continue
                self._last_sent_pos = pos
                motor = self._motor  # 拿到引用，锁外使用
            # ── 锁外：做 Modbus I/O（MotorController 内部已有线程锁）───────
            try:
                motor.set_target_position(pos)
                motor.trigger_motion()
            except Exception as e:
                logger.warning("[ZhixingSDKGripper] _send_loop Modbus 异常: %s", e)

    def set_opening_m(self, distance_m: float) -> None:
        d = float(np.clip(distance_m, 0.0, GRIPPER_MAX_M))
        ratio = 1.0 - d / GRIPPER_MAX_M
        pos = int(ratio * DEFAULT_GRIPPER_POS_CLOSE)
        with self._lock:
            # 只更新 pending，由后台线程实际发送（避免阻塞调用方）
            self._pending_pos = pos

    def get_position_m(self) -> float:
        """返回夹爪当前位置估计（米）。

        直接使用缓存的 EMA 目标值（不触发 Modbus 读取），避免阻塞主控制线程。
        _ema_pos 是我们发给夹爪的平滑目标，是夹爪实际位置的最佳估计。
        """
        with self._lock:
            if self._ema_pos is not None:
                ratio = self._ema_pos / max(DEFAULT_GRIPPER_POS_CLOSE, 1)
                return float(np.clip((1.0 - ratio) * GRIPPER_MAX_M, 0.0, GRIPPER_MAX_M))
            if self._last_sent_pos is not None:
                ratio = self._last_sent_pos / max(DEFAULT_GRIPPER_POS_CLOSE, 1)
                return float(np.clip((1.0 - ratio) * GRIPPER_MAX_M, 0.0, GRIPPER_MAX_M))
            return 0.0

    def open(self) -> None:
        self.set_opening_m(GRIPPER_MAX_M)

    def release(self) -> None:
        with self._lock:
            self._motor = None


# ─────────────────────────────────────────────────────────────────────────────
# UMI VIO 位姿录制节点 —— 增量映射到机械臂关节角
# ─────────────────────────────────────────────────────────────────────────────

class UMIPoseRecorder(Node):
    """订阅 UMI 设备的 VIO 位姿话题，通过增量映射输出机械臂目标关节角。

    工作原理（按 't' 对齐时）：

      1. 用户手动将 UMI 手柄调整到与机械臂末端位姿大致一致的方向/位置；
      2. 按 't' 键触发 set_alignment_reference()：
           - 记录此刻 UMI 位姿为 T_umi_ref（VIO 世界坐标系）
           - 记录此刻机械臂末端位姿为 T_ee_ref（通过 SDK GetPose）
           - 记录此刻关节角为 q_ref（作为 IK 种子及失败回退）
      3. 后续每步计算：
           delta_T      = T_umi_ref^{-1} @ T_umi_curr   （UMI 手柄相对运动）
           T_ee_target  = T_ee_ref @ delta_T              （相同增量映射到机械臂末端）
           q_target     = IK(T_ee_target, seed=q_ref)    （逆运动学求解关节角）

    输出格式与 PikaChunkEnvAdapter / machine_A 侧一致：
        snapshot_latest() → (np.ndarray shape (7,), seq_id)
        其中 7D = [j1..j6 rad, gripper_travel]（关节角 + 夹爪行程值）

    控制链路：
        snapshot_latest() 的关节角结果通过 DobotUMIEnvAdapter._sample_latest_human_action()
        送入 send_action(_source="teleop")，再由 DobotSDKArm.servo_j() 驱动机械臂。

    夹爪处理：
        夹爪当前位置由 gripper.get_position_m() 实时读取后转换成行程值，
        在人工接管阶段按模型/回放使用的 gripper_travel 表示输出。
    """

    def __init__(
        self,
        pose_topic: str,
        arm: DobotSDKArm,
        gripper: ZhixingSDKGripper,
        max_gripper_m: float = GRIPPER_MAX_M,
        R_align: np.ndarray | None = None,
        R_align_rot: np.ndarray | None = None,
        motion_scale: float = 1.0,
        smooth_alpha: float = 1.0,
        encoder_topic: str | None = DEFAULT_UMI_ENCODER_TOPIC,
    ):
        super().__init__("umi_pose_recorder")
        if not _HAS_SCIPY:
            raise ImportError(
                "UMIPoseRecorder 需要 scipy。请 pip install scipy。"
            )
        self._arm = arm
        self._gripper = gripper
        self._max_gripper_m = max_gripper_m
        # R_align: 3×3 旋转矩阵，将 UMI 平移增量从 UMI 体坐标系变换到机械臂末端坐标系
        # 默认 None 等价于单位矩阵（不做对齐）
        if R_align is not None:
            self._R_align = np.asarray(R_align, dtype=np.float64).reshape(3, 3)
            logger.info("[UMIPoseRecorder] R_align(translation)=\n%s", self._R_align)
        else:
            self._R_align = None
        # R_align_rot: 3×3 旋转矩阵，单独用于旋转增量对齐
        # None = 与 R_align 相同（大多数情况下两者一致）
        # 仅当 UMI VIO 平移坐标系和旋转坐标系定义不一致时才单独配置
        if R_align_rot is not None:
            self._R_align_rot = np.asarray(R_align_rot, dtype=np.float64).reshape(3, 3)
            logger.info("[UMIPoseRecorder] R_align(rotation)=\n%s", self._R_align_rot)
        else:
            self._R_align_rot = self._R_align  # 默认与平移对齐矩阵相同
        # motion_scale: UMI 增量缩放系数（平移和旋转同时缩放）
        self._motion_scale = float(motion_scale)
        logger.info("[UMIPoseRecorder] motion_scale=%.3f", self._motion_scale)
        # smooth_alpha: EMA 平滑系数（0<α≤1）。1.0=不平滑，越小越平滑
        self._smooth_alpha = float(np.clip(smooth_alpha, 0.0, 1.0))
        logger.info("[UMIPoseRecorder] smooth_alpha=%.3f", self._smooth_alpha)

        # UMI 手柄夹爪编码器：使用 cal_encoder_position 几何映射（正弦公式）
        self._latest_umi_gripper_angle: float | None = None  # 最新编码器角度（度），None=未收到
        self._policy_enabled_getter: Callable[[], bool] | None = None

        self._lock = threading.Lock()

        # 最新 UMI 末端位姿（4×4，VIO 世界坐标系）
        self._latest_T: np.ndarray | None = None
        # EMA 平滑后的 UMI 末端位姿（4×4）
        self._smooth_T: np.ndarray | None = None
        self._latest_seq: int = -1

        # 对齐参考帧（按 't' 时记录）
        self._T_umi_ref: np.ndarray | None = None   # UMI 参考位姿
        self._T_ee_ref: np.ndarray | None = None    # 机械臂末端参考位姿
        self._q_ref: np.ndarray | None = None       # 机械臂参考关节角（IK 种子及失败回退）

        # 上次成功输出的动作（IK 失败时保持）
        self._last_output: np.ndarray | None = None

        self.create_subscription(ROSPoseStamped, pose_topic, self._on_pose, 50)
        self.get_logger().info(f"UMI VIO 位姿话题: {pose_topic}")

        # UMI 夹爪编码器订阅
        if encoder_topic and _HAS_ENCODER_STATE:
            self.create_subscription(
                UMIEncoderState, encoder_topic, self._on_encoder_state,
                qos_profile_sensor_data,
            )
            self.get_logger().info(
                f"UMI 夹爪编码器话题: {encoder_topic}  "
                f"(cal_encoder_position 正弦映射 → [0, {self._max_gripper_m * 1000:.1f}mm])"
            )
        elif encoder_topic and not _HAS_ENCODER_STATE:
            self.get_logger().warn(
                "common_msgs 不可用，UMI 夹爪编码器话题已禁用。"
                "请 source handheld-umi_ws/install/setup.bash 后重启。"
            )

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _umi_gripper_angle_to_m(self, angle_deg: float) -> float:
        """将 UMI 手柄夹爪编码器角度（度）转换为机械臂夹爪开合量（米）。

        实测标定：
            angle = 0°  → UMI 完全张开 → 机械臂夹爪完全张开（max_gripper_m）
            angle = 50° → UMI 完全闭合 → 机械臂夹爪完全闭合（0 m）

        方向与原始 cal_encoder_position 相反（角度越大 = 越闭合），
        因此先用 cal_encoder_position 计算几何位移，再归一化后取反：

            pos_at_angle   = cal_encoder_position(angle)
            pos_at_closed  = cal_encoder_position(UMI_GRIPPER_CLOSED_DEG)  # ~50°
            ratio_closed   = clip(pos_at_angle / pos_at_closed, 0, 1)
            gripper_m      = (1 - ratio_closed) * max_gripper_m

        即：angle=0 → ratio=0 → gripper_m=max（张开）
            angle=50 → ratio=1 → gripper_m=0（闭合）
        """
        import math as _math
        _BASE_RAD        = _math.radians(34.02)
        _ORIGIN_HYPO_MM  = 42.0
        # UMI 手柄完全闭合时对应的编码器角度（度），由实测标定
        _CLOSED_DEG      = 50.0

        def _cal(deg: float) -> float:
            rad = _math.radians(max(0.0, deg))
            return 2.0 * (_math.sin(_BASE_RAD) - _math.sin(_BASE_RAD - rad)) * _ORIGIN_HYPO_MM

        pos_at_angle  = _cal(angle_deg)
        pos_at_closed = _cal(_CLOSED_DEG)   # ≈ 70.17 mm

        ratio_closed = float(np.clip(pos_at_angle / max(pos_at_closed, 1e-6), 0.0, 1.0))
        return float(np.clip((1.0 - ratio_closed) * self._max_gripper_m, 0.0, self._max_gripper_m))

    @staticmethod
    def _pose_msg_to_matrix(msg) -> np.ndarray:
        """geometry_msgs/Pose → 4×4 齐次变换矩阵（位置单位：m）。"""
        p = msg.position
        q = msg.orientation  # x, y, z, w
        R = _Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [p.x, p.y, p.z]
        return T

    def _on_pose(self, msg: ROSPoseStamped) -> None:
        T = self._pose_msg_to_matrix(msg.pose)
        with self._lock:
            self._latest_T = T
            alpha = self._smooth_alpha
            if self._smooth_T is None or alpha >= 1.0:
                # 初始化或不平滑：直接赋值
                self._smooth_T = T.copy()
            else:
                # 平移：线性 EMA
                p_smooth = alpha * T[:3, 3] + (1.0 - alpha) * self._smooth_T[:3, 3]
                # 旋转：四元数 NLERP（线性插值后归一化，近似 SLERP，无额外依赖）
                q_curr = _Rotation.from_matrix(T[:3, :3]).as_quat()
                q_prev = _Rotation.from_matrix(self._smooth_T[:3, :3]).as_quat()
                # 保证插值走短弧（避免符号翻转导致绕远路）
                if np.dot(q_curr, q_prev) < 0.0:
                    q_curr = -q_curr
                q_new = alpha * q_curr + (1.0 - alpha) * q_prev
                q_new /= np.linalg.norm(q_new)
                R_new = _Rotation.from_quat(q_new).as_matrix()
                smooth = np.eye(4, dtype=np.float64)
                smooth[:3, :3] = R_new
                smooth[:3, 3] = p_smooth
                self._smooth_T = smooth
            self._latest_seq += 1

    def _on_encoder_state(self, msg) -> None:
        """处理 /umi1/vitai/encoder_state 消息（common_msgs/EncoderState）。

        直接在回调中调用 set_opening_m() 驱动机械臂夹爪，
        不经过 snapshot_latest → send_action 流水线，保证夹爪响应及时。
        内部死区过滤避免高频 Modbus 指令导致的抖动。
        仅在编码器状态正常（ENCODER_OK=0）时执行。
        """
        if msg.encoder_status != 0:
            return
        if self._policy_enabled_getter is not None:
            try:
                if bool(self._policy_enabled_getter()):
                    return
            except Exception:
                pass

        angle = float(msg.gripper_angle)
        with self._lock:
            self._latest_umi_gripper_angle = angle

        gripper_m = self._umi_gripper_angle_to_m(angle)
        try:
            self._gripper.set_opening_m(gripper_m)
        except Exception as e:
            self.get_logger().warn(f"[_on_encoder_state] 夹爪控制失败: {e}")

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        with self._lock:
            return self._latest_T is not None

    def bind_policy_enabled_getter(self, getter: Callable[[], bool]) -> None:
        self._policy_enabled_getter = getter

    def set_alignment_reference(self) -> None:
        """对齐参考帧：在用户按 't' 并手动对齐后自动调用。

        记录此刻 UMI 位姿（T_umi_ref）、机械臂末端位姿（T_ee_ref）和
        关节角（q_ref），作为后续增量映射的基准。
        """
        logger.warning("[set_alignment_reference] 开始执行（守护线程）")
        try:
            with self._lock:
                T_umi = self._latest_T
            if T_umi is None:
                logger.warning("[set_alignment_reference] 尚未收到 UMI 位姿，跳过")
                return
            logger.warning("[set_alignment_reference] 调用 get_end_effector_pose_matrix() ...")
            T_ee = self._arm.get_end_effector_pose_matrix()
            logger.warning("[set_alignment_reference] T_ee 获取完成: pos_mm=[%.1f,%.1f,%.1f]",
                           T_ee[0,3]*1000, T_ee[1,3]*1000, T_ee[2,3]*1000)
            logger.warning("[set_alignment_reference] 调用 get_joint_angles_rad() ...")
            q_now = self._arm.get_joint_angles_rad()
            logger.warning("[set_alignment_reference] q_now_deg=%s",
                           np.round(np.rad2deg(q_now), 2).tolist())
            with self._lock:
                self._T_umi_ref = T_umi.copy()
                self._T_ee_ref  = T_ee.copy()
                self._q_ref     = q_now.copy()
                # 对齐时同步重置平滑状态，防止旧滞后影响新基准
                self._smooth_T  = T_umi.copy()
            logger.warning("[set_alignment_reference] ✅ 对齐参考帧写入完成 "
                           "T_umi_ref=%s T_ee_ref=%s q_ref=%s",
                           self._T_umi_ref is not None,
                           self._T_ee_ref is not None,
                           self._q_ref is not None)
        except Exception as e:
            logger.warning("[set_alignment_reference] ❌ 异常: %s", e, exc_info=True)

    def invalidate_alignment_reference(self) -> None:
        """立刻废弃旧对齐参考，直到新的参考帧写入前都只允许原地保持。

        按 `t` 从 policy 切到 teleop 时，新的 set_alignment_reference() 是异步线程。
        若这段空窗里仍沿用上一次 teleop 的参考帧，首个 teleop step 可能按旧参考
        计算增量，造成危险突跳。这里先清空参考和上次输出，强制 snapshot_latest()
        在新参考就绪前走 hold-current 路径。
        """
        with self._lock:
            self._T_umi_ref = None
            self._T_ee_ref = None
            self._q_ref = None
            self._last_output = None
        logger.warning("[invalidate_alignment_reference] Cleared stale teleop alignment reference.")

    def snapshot_latest(self) -> tuple[np.ndarray | None, int]:
        """返回 (7D 动作, seq_id)。

        7D = [j1..j6 rad, gripper_travel]（关节角 + 夹爪行程值，与 machine_A 侧格式一致）。

        增量映射得到目标末端位姿 T_ee_target，调用 Dobot SDK InverseKin 得到关节角；
        IK 失败时保持上一次输出；尚未对齐（'t' 未按）时原地保持当前关节角。
        """
        with self._lock:
            T_umi_curr = self._smooth_T      # ← 使用 EMA 平滑后的位姿
            seq        = self._latest_seq
            T_umi_ref  = self._T_umi_ref
            T_ee_ref   = self._T_ee_ref
            q_ref      = self._q_ref

        # 夹爪开合量：_on_encoder_state 已直接驱动物理夹爪。
        # 此处读取物理夹爪位置并转换为专家数据/模型使用的行程值，作为 action 的记录值。
        gripper_m = float(np.clip(self._gripper.get_position_m(), 0.0, self._max_gripper_m))
        gripper_travel = _gripper_m_to_travel(gripper_m, max_gripper_m=self._max_gripper_m)

        def _hold_current() -> np.ndarray:
            q = self._arm.get_joint_angles_rad()
            return np.concatenate([q, [gripper_travel]], dtype=np.float32)

        if T_umi_curr is None:
            logger.warning("[snapshot_latest] smooth_T is None → hold current (UMI 话题未收到数据?)")
            return _hold_current(), seq

        if T_umi_ref is None or T_ee_ref is None or q_ref is None:
            # 尚未按 't' 对齐 → 原地保持
            logger.warning("[snapshot_latest] 对齐参考帧未就绪(T_umi_ref=%s, T_ee_ref=%s, q_ref=%s) → hold current",
                           T_umi_ref is not None, T_ee_ref is not None, q_ref is not None)
            return _hold_current(), seq

        # ── 增量映射 → IK → 关节角 ───────────────────────────────────────────
        # delta_T：UMI 手柄相对参考帧的增量，在 UMI 体坐标系中表达（右乘）
        # T_ee_target = T_ee_ref @ delta_T_aligned：把对齐后的增量施加到机械臂末端体坐标系
        delta_T = np.linalg.inv(T_umi_ref) @ T_umi_curr

        # ── R_align：UMI 手柄坐标系 → 机械臂末端坐标系 的固定旋转 ──────────────
        # 平移和旋转分量分别用独立的对齐矩阵（大多数情况下相同，但 VIO 系统有时不一致）
        #   delta_T_aligned.t = R_align     @ delta_T.t            (平移分量)
        #   delta_T_aligned.R = R_align_rot @ delta_T.R @ R_align_rot.T (旋转分量)
        delta_T_aligned = np.eye(4, dtype=np.float64)
        if self._R_align is not None:
            delta_T_aligned[:3, 3] = self._R_align @ delta_T[:3, 3]
        else:
            delta_T_aligned[:3, 3] = delta_T[:3, 3]
        if self._R_align_rot is not None:
            Rr = self._R_align_rot
            delta_T_aligned[:3, :3] = Rr @ delta_T[:3, :3] @ Rr.T
        else:
            delta_T_aligned[:3, :3] = delta_T[:3, :3]

        # ── motion_scale：缩放增量幅度，抑制抖动放大 ──────────────────────────
        # 平移分量：直接线性缩放
        # 旋转分量：将旋转向量（轴角）缩放后重建旋转矩阵
        if self._motion_scale != 1.0:
            delta_T_aligned[:3, 3] *= self._motion_scale
            rvec = _Rotation.from_matrix(delta_T_aligned[:3, :3]).as_rotvec()
            delta_T_aligned[:3, :3] = _Rotation.from_rotvec(rvec * self._motion_scale).as_matrix()

        T_ee_target = T_ee_ref @ delta_T_aligned
        # 就近选解：优先用上一次成功输出的关节角，初始时用对齐参考帧的关节角
        with self._lock:
            last_out = self._last_output
        q_seed = last_out[:6].copy() if last_out is not None else q_ref
        logger.warning("[snapshot_latest] 调用 InverseKin, target_pos_mm=[%.1f,%.1f,%.1f], q_seed_deg=%s",
                    T_ee_target[0,3]*1000, T_ee_target[1,3]*1000, T_ee_target[2,3]*1000,
                    np.round(np.rad2deg(q_seed), 2).tolist())
        q_target    = self._arm.inverse_kinematics_from_matrix(T_ee_target, q_seed=q_seed)

        if q_target is None:
            logger.warning("[snapshot_latest] IK 返回 None → hold last_output or current")
            if self._last_output is not None:
                return self._last_output.copy(), seq
            return _hold_current(), seq

        logger.warning("[snapshot_latest] IK 成功: q_target_deg=%s", np.round(np.rad2deg(q_target), 2).tolist())
        action = np.concatenate([q_target, [gripper_travel]], dtype=np.float32)
        with self._lock:
            self._last_output = action.copy()
        return action, seq


# ─────────────────────────────────────────────────────────────────────────────
# 支持自动对齐的 Teleop 触发节点
# ─────────────────────────────────────────────────────────────────────────────

class DobotUMITeleopTriggerNode(TeleopTriggerNode):
    """在 TeleopTriggerNode 基础上增加：进入人工接管时自动记录 UMI 对齐参考帧。

    当 't' 键触发且策略被关闭（切换到人工接管模式）时，
    自动调用 umi_pose_recorder.set_alignment_reference() 和
    robot_bridge.reset_teleop_state()，
    使增量映射从当前机械臂姿态开始，无需额外标定；
    同时清除上一次 teleop 发送记录，防止首步跳变。
    """

    def __init__(
        self,
        *args,
        umi_pose_recorder: UMIPoseRecorder,
        robot_bridge: "DobotUMIRobotBridge",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._umi_pose_recorder = umi_pose_recorder
        self._robot_bridge = robot_bridge

    def _on_trigger(self, _request, response):
        try:
            response = super()._on_trigger(_request, response)
            # 若刚切换到人工接管模式，自动对齐参考帧并重置限幅基准
            # set_alignment_reference() 内含 SDK TCP 调用（GetPose），放后台线程异步执行，
            # 避免阻塞 ROS 服务回调导致客户端超时。
            logger.warning("[_on_trigger] response.success=%s message=%r",
                           response.success, response.message)
            if response.success and "mode=teleop" in response.message:
                logger.warning("[_on_trigger] 切换到人工模式 → 异步记录 UMI 对齐参考帧 + 重置限幅基准")
                self._umi_pose_recorder.invalidate_alignment_reference()
                self._robot_bridge.reset_teleop_state()
                threading.Thread(
                    target=self._umi_pose_recorder.set_alignment_reference,
                    daemon=True,
                ).start()
        except Exception as e:
            logger.warning("[_on_trigger] ❌ 未捕获异常: %s", e, exc_info=True)
            response.success = False
            response.message = f"_on_trigger 异常: {e}"
        return response


# ─────────────────────────────────────────────────────────────────────────────
# 机器人桥接（SDK 直驱；相机仍走 ROS 话题）
# ─────────────────────────────────────────────────────────────────────────────

class DobotUMIRobotBridge:
    """将 DobotUMIObsBuffer（相机 ROS） + DobotSDKArm + ZhixingSDKGripper
    封装为 PikaChunkEnvAdapter 所需的 robot 接口。
    """

    # ── 安全限幅默认值 ──────────────────────────────────────────────────────
    # teleop 每步各关节最大允许变化量（弧度）。
    # 约 0.05 rad ≈ 2.9°/step，20 Hz 下对应约 57°/s，流畅且安全。
    # 通过 args.teleop_max_delta_rad 覆盖。
    DEFAULT_TELEOP_MAX_DELTA_RAD: float = 0.05
    # teleop 夹爪每步最大变化量（米）
    DEFAULT_TELEOP_MAX_DELTA_GRIPPER_M: float = 0.085  # 不限幅，允许全行程一步到位
    # policy 每步各关节最大允许变化量（弧度）。
    # 比 teleop 更保守，用于抑制 ref_chunk / actor 推理异常时的单步突跳。
    DEFAULT_POLICY_MAX_DELTA_RAD: float = 0.03
    # policy ServoJ EMA 默认更强，抑制 VLA/ref_chunk 的高频小抖。
    DEFAULT_POLICY_EMA_ALPHA: float = 0.25
    # policy 关节死区；小于该阈值的命令变化视为推理噪声并保持上一平滑目标。
    DEFAULT_POLICY_DEADBAND_RAD: float = 0.0015
    # policy 夹爪每步最大变化量（米）
    DEFAULT_POLICY_MAX_DELTA_GRIPPER_M: float = 0.01
    # policy 夹爪目标 EMA；越小越平滑，抑制接触前后的微小开合抖动。
    DEFAULT_POLICY_GRIPPER_EMA_ALPHA: float = 1.0
    # policy 夹爪死区；小于该开合变化时保持上一目标，避免在抓取附近反复开合。
    DEFAULT_POLICY_GRIPPER_DEADBAND_M: float = 0.0
    # rollout 层统一动作增量限幅默认值（6 关节 + 夹爪）。
    # 作为 robot bridge 关节增量限幅之前的第一道保险。
    DEFAULT_ACTION_DELTA_LIMITS: tuple[float, ...] = (0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.005)

    def __init__(
        self,
        args: argparse.Namespace,
        cam_recorder: "RealSenseImageRecorder",
        arm: DobotSDKArm,
        gripper: ZhixingSDKGripper,
    ):
        self._args = args
        self._cam_recorder = cam_recorder
        self._arm = arm
        self._gripper = gripper
        self._control_state_lock = threading.Lock()
        # 上一次实际发送的关节角（用于逐步限幅，None 表示尚未发送过）
        self._last_sent_q6: np.ndarray | None = None
        self._last_sent_gripper_m: float | None = None
        # EMA 平滑缓冲（用于抑制 ServoJ 微抖）
        self._ema_q6: np.ndarray | None = None
        self._ema_gripper_m: float | None = None
        self._last_ema_source: str | None = None

    def shutdown(self) -> None:
        # 注意：不调用 disable()，不发 DisableRobot 指令。
        # 只断开 TCP socket，机械臂保持使能状态，不会抱闸失力。
        self._arm.disconnect()
        # 夹爪 modbus 连接直接关闭，不做额外动作
        try:
            self._gripper.release()
        except Exception:
            pass

    def set_policy_control_active(self, enabled: bool) -> None:
        pass  # SDK 直驱无需暂停流；夹爪每次 send_action 时直接设置

    def wait_for_observation_ready(self, timeout_s: float | None = None) -> None:
        self._cam_recorder.wait_ready(timeout_s=timeout_s or 10.0)

    def get_observation(self, resize_hw: tuple[int, int], task: str) -> dict[str, Any]:
        retries = max(int(self._args.capture_retries), 1)
        images = None
        for attempt in range(retries):
            if self._cam_recorder.is_ready():
                images = self._cam_recorder.get_images(resize_hw=resize_hw)
                if images is not None:
                    break
            time.sleep(self._args.capture_retry_sleep_s)
        if images is None:
            raise RuntimeError("采集观测失败：RealSense 相机未就绪，超过重试次数")

        # 关节角通过 SDK 反馈线程直接读取；夹爪转换为专家数据/模型使用的行程值。
        q6 = self._arm.get_joint_angles_rad()
        gripper_m = float(np.clip(self._gripper.get_position_m(), 0.0, self._args.max_gripper_m))
        gripper_travel = _gripper_m_to_travel(gripper_m, max_gripper_m=self._args.max_gripper_m)
        state7 = np.concatenate([q6, [gripper_travel]], dtype=np.float32)

        return {
            "state": state7,
            "images": {
                "cam_front": images["cam_front"],
                "cam_wrist": images["cam_wrist"],
            },
            "prompt": task,
        }

    def send_action(self, action7: np.ndarray, *, _source: str = "policy") -> np.ndarray:
        """执行 7D 动作。

        action7 格式：[j1..j6 rad, gripper_travel]（关节角 + 夹爪行程值，与 replay buffer 记录格式一致）。
        teleop / policy 均通过 servo_j 下发关节角。

        安全限幅：
            - policy 路径：每步关节/夹爪增量受限，抑制 ref_chunk 或 actor 推理异常时的大步跳变。
            - teleop 路径：同样逐步限幅，避免 UMI 对齐或操作者快速抖动造成的突跳。
            - 首步统一以当前实测关节角为基准，避免刚接管时的瞬间大幅移动。
        """
        action7 = np.asarray(action7, dtype=np.float32).reshape(-1)
        if action7.shape[0] < 7:
            raise ValueError(f"动作维度不足 7，当前 {action7.shape}")

        with self._control_state_lock:
            q6_desired = action7[:6]
            gripper_travel_desired = float(np.clip(action7[6], 0.0, DEFAULT_GRIPPER_POS_CLOSE))
            gripper_m_desired = _gripper_travel_to_m(
                gripper_travel_desired,
                max_gripper_m=self._args.max_gripper_m,
            )
            if _source == "policy":
                max_dq = float(
                    getattr(
                        self._args,
                        "policy_max_delta_rad",
                        DobotUMIRobotBridge.DEFAULT_POLICY_MAX_DELTA_RAD,
                    )
                )
                max_dg = float(
                    getattr(
                        self._args,
                        "policy_max_delta_gripper_m",
                        DobotUMIRobotBridge.DEFAULT_POLICY_MAX_DELTA_GRIPPER_M,
                    )
                )
            else:
                max_dq = float(
                    getattr(
                        self._args,
                        "teleop_max_delta_rad",
                        DobotUMIRobotBridge.DEFAULT_TELEOP_MAX_DELTA_RAD,
                    )
                )
                max_dg = float(
                    getattr(
                        self._args,
                        "teleop_max_delta_gripper_m",
                        DobotUMIRobotBridge.DEFAULT_TELEOP_MAX_DELTA_GRIPPER_M,
                    )
                )

            if _source == "policy":
                ema_alpha = float(getattr(self._args, "policy_ema_alpha", 0.35))
            else:
                ema_alpha = float(getattr(self._args, "teleop_ema_alpha", 0.5))

            q6, gripper_m, clipped = self._limit_action_step(
                q6_desired=q6_desired,
                gripper_m_desired=gripper_m_desired,
                max_dq=max_dq,
                max_dg=max_dg,
            )
            deadband_rad = 0.0
            if _source == "policy":
                deadband_rad = float(
                    getattr(
                        self._args,
                        "policy_deadband_rad",
                        DobotUMIRobotBridge.DEFAULT_POLICY_DEADBAND_RAD,
                    )
                )
            gripper_deadband_m = 0.0
            gripper_ema_alpha = 1.0
            if _source == "policy":
                gripper_deadband_m = float(
                    getattr(
                        self._args,
                        "policy_gripper_deadband_m",
                        DobotUMIRobotBridge.DEFAULT_POLICY_GRIPPER_DEADBAND_M,
                    )
                )
                gripper_ema_alpha = float(
                    getattr(
                        self._args,
                        "policy_gripper_ema_alpha",
                        DobotUMIRobotBridge.DEFAULT_POLICY_GRIPPER_EMA_ALPHA,
                    )
                )

            q6_sent, gripper_m_sent = self._apply_ema_and_send(
                q6,
                gripper_m,
                alpha=ema_alpha,
                source=_source,
                deadband_rad=deadband_rad,
                gripper_alpha=gripper_ema_alpha,
                gripper_deadband_m=gripper_deadband_m,
            )
        gripper_travel_sent = _gripper_m_to_travel(gripper_m_sent, max_gripper_m=self._args.max_gripper_m)
        if clipped:
            logger.warning(
                "[EXEC|%s CLIPPED] desired_deg=%s limited_deg=%s sent_deg=%s desired_gripper=%.1f limited_gripper=%.1f gripper_mm=%.1f",
                _source,
                np.round(np.rad2deg(q6_desired), 2).tolist(),
                np.round(np.rad2deg(q6), 2).tolist(),
                np.round(np.rad2deg(q6_sent), 2).tolist(),
                gripper_travel_desired,
                gripper_travel_sent,
                gripper_m_sent * 1000,
            )
        return np.concatenate(
            [q6_sent.astype(np.float32, copy=False), np.asarray([gripper_travel_sent], dtype=np.float32)]
        )

    def _limit_action_step(
        self,
        *,
        q6_desired: np.ndarray,
        gripper_m_desired: float,
        max_dq: float,
        max_dg: float,
    ) -> tuple[np.ndarray, float, bool]:
        if self._last_sent_q6 is None:
            self._last_sent_q6 = self._arm.get_joint_angles_rad().copy()
        if self._last_sent_gripper_m is None:
            self._last_sent_gripper_m = float(
                np.clip(self._gripper.get_position_m(), 0.0, self._args.max_gripper_m)
            )

        delta_q = q6_desired - self._last_sent_q6
        delta_q_clipped = np.clip(delta_q, -max_dq, max_dq)
        q6 = self._last_sent_q6 + delta_q_clipped

        delta_g = gripper_m_desired - self._last_sent_gripper_m
        delta_g_clipped = float(np.clip(delta_g, -max_dg, max_dg))
        gripper_m = float(
            np.clip(self._last_sent_gripper_m + delta_g_clipped, 0.0, self._args.max_gripper_m)
        )

        self._last_sent_q6 = q6.copy()
        self._last_sent_gripper_m = gripper_m
        clipped = bool(
            np.any(np.abs(delta_q - delta_q_clipped) > 1e-6)
            or abs(delta_g - delta_g_clipped) > 1e-6
        )
        return q6, gripper_m, clipped

    def _apply_ema_and_send(
        self,
        q6: np.ndarray,
        gripper_m: float,
        *,
        alpha: float,
        source: str,
        deadband_rad: float = 0.0,
        gripper_alpha: float = 1.0,
        gripper_deadband_m: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        gripper_alpha = float(np.clip(gripper_alpha, 0.0, 1.0))
        if self._ema_q6 is None or self._last_ema_source != source:
            self._ema_q6 = q6.copy()
            self._ema_gripper_m = float(gripper_m)
        else:
            if deadband_rad > 0.0:
                delta = q6 - self._ema_q6
                q6 = np.where(np.abs(delta) < deadband_rad, self._ema_q6, q6)
            self._ema_q6 = alpha * q6 + (1.0 - alpha) * self._ema_q6
            if self._ema_gripper_m is None:
                self._ema_gripper_m = float(gripper_m)
            if gripper_deadband_m > 0.0 and abs(float(gripper_m) - float(self._ema_gripper_m)) < gripper_deadband_m:
                gripper_m = float(self._ema_gripper_m)
            self._ema_gripper_m = float(gripper_alpha * gripper_m + (1.0 - gripper_alpha) * self._ema_gripper_m)
        if self._ema_gripper_m is None:
            self._ema_gripper_m = float(gripper_m)
        self._last_ema_source = source
        control_hz = float(getattr(self._args, "control_frequency_hz", 20.0))
        # ServoJ 的 t 应与控制周期一致；SDK 文档下限约为 0.02s。
        servo_t = max(0.02, 1.0 / control_hz)
        self._arm.servo_j(self._ema_q6, t=servo_t)
        self._gripper.set_opening_m(float(self._ema_gripper_m))
        return self._ema_q6.copy(), float(self._ema_gripper_m)

    def reset_teleop_state(self) -> None:
        """清除控制缓存，强制后续首步以当前实测状态为基准。"""
        with self._control_state_lock:
            self._last_sent_q6 = None
            self._last_sent_gripper_m = None
            self._ema_q6 = None  # 清空 EMA 缓冲，避免上次遥操作状态污染新一轮
            self._ema_gripper_m = None
            self._last_ema_source = None


# ─────────────────────────────────────────────────────────────────────────────
# Dobot-UMI 专用 EnvAdapter：修复 teleop 分支需主动发指令的问题
# ─────────────────────────────────────────────────────────────────────────────

class DobotUMIEnvAdapter(PikaChunkEnvAdapter):
    """PikaChunkEnvAdapter 的 Dobot-UMI 定制子类。

    原版 pika_sync_ros.PikaChunkEnvAdapter 在人工接管（else）分支中，
    不调用 robot.send_action()，因为 Pika 机器人是人直接握着物理控制的。
    但 Dobot 只能靠 SDK ServoJ 驱动，必须主动发指令。

    本子类仅覆盖 _sample_latest_human_action：在返回动作之前，
    同时将该动作以 _source="teleop" 发给机械臂，触发实际运动。

    当前行为：
        _sample_latest_human_action 正常执行 → 机械臂跟随 UMI 运动。
        execute_chunk 中策略分支的 robot.send_action(bounded) 调用 DobotUMIRobotBridge
        的 send_action(_source="policy")，该路径会经过逐步限幅后真实发送。
    """

    def _apply_action_limits(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)[: self._system.rl.action_dim]
        if self._action_delta_limits is None or self._last_sent_action is None:
            self._last_sent_action = action.copy()
            return action

        prev = self._last_sent_action.copy()
        current_for_limit = action.copy()
        prev_for_limit = prev.copy()
        if action.shape[0] >= 7:
            current_for_limit[6] = _gripper_travel_to_m(
                float(action[6]),
                max_gripper_m=self._robot._args.max_gripper_m,
            )
            prev_for_limit[6] = _gripper_travel_to_m(
                float(prev[6]),
                max_gripper_m=self._robot._args.max_gripper_m,
            )

        delta = np.clip(
            current_for_limit - prev_for_limit,
            -self._action_delta_limits,
            self._action_delta_limits,
        )
        bounded_for_limit = prev_for_limit + delta
        bounded = action.copy()
        bounded[: bounded_for_limit.shape[0]] = bounded_for_limit
        if bounded.shape[0] >= 7:
            bounded[6] = _gripper_m_to_travel(
                float(bounded_for_limit[6]),
                max_gripper_m=self._robot._args.max_gripper_m,
            )
        self._last_sent_action = bounded.copy()
        return bounded

    def _sample_latest_human_action(self, observation) -> np.ndarray:
        logger.warning("[DobotUMIEnvAdapter] _sample_latest_human_action 被调用")
        action = super()._sample_latest_human_action(observation)
        logger.warning("[DobotUMIEnvAdapter] super() 返回 action_deg=%s，调用 send_action(teleop)",
                    np.round(np.rad2deg(action[:6]), 2).tolist())
        # 主动将 IK 计算出的关节角目标发送给机械臂（teleop 路径，ServoJ 执行）
        return self._robot.send_action(action, _source="teleop")

    def _reset_robot_to_mode_start(self) -> None:
        """覆盖父类方法：用 MovJ 代替 ServoJ 插值，实现安全的大幅度归位运动。

        父类逻辑（pika_sync_ros）通过 send_action(waypoint) 逐步发送插值点，
        底层走 ServoJ(t=0.1s)——ServoJ 没有速度/加速度规划，距离稍大就会全速冲向目标。

        Dobot 的正确大幅运动接口是 MovL/MovJ：内置梯形速度规划，无论距离多远都平滑安全。
        此处直接一步到位，忽略父类的 ServoJ 插值逻辑。

        reset_action 格式（由 yaml 配置）：
          - 7D [x_m, y_m, z_m, rx_rad, ry_rad, rz_rad, gripper_m]：用 MovL（笛卡尔）
          - 7D [j1..j6 rad, gripper_m]（值全部 < π 时判断为关节角）：用 MovJ
          is_cartesian 字段为 True 时强制走笛卡尔路径。
        """
        target_raw = (
            self._system.env_driver.critical_phase_reset_action
            if self._task_mode == "critical_phase"
            else self._system.env_driver.full_task_reset_action
        )
        if target_raw is None:
            logger.info("No reset action configured for task_mode=%s; skipping reset.", self._task_mode)
            return

        target = np.asarray(target_raw, dtype=np.float64).reshape(-1)
        gripper_m_target = float(np.clip(
            target[6] if target.shape[0] > 6 else 0.08,
            0.0, self._robot._args.max_gripper_m,
        ))

        # ── 降速归位，避免过快运动 ─────────────────────────────────────────────
        reset_speed_pct = int(getattr(self._robot._args, "reset_speed_pct", 20))
        self._robot._arm.set_speed_factor(reset_speed_pct)
        logger.warning("[RESET] 速度比例设为 %d%%", reset_speed_pct)

        # ── 判断是笛卡尔目标还是关节角目标 ──────────────────────────────────────
        # 判断依据：配置里 is_cartesian=true，或者前3个值明显是位置（量纲~0.1~1.0 m）
        is_cartesian = bool(getattr(self._system.env_driver, "reset_action_is_cartesian", False))
        if not is_cartesian:
            # 自动判断：位置值通常在 0.1~2.0m 范围，关节角弧度通常在 -π~π
            # 若 x,y,z 均小于 3.0（m），姿态均小于 4.0（rad ≈ 229°），视为笛卡尔
            xyz = target[:3]
            is_cartesian = bool(np.all(np.abs(xyz) < 3.0) and np.max(np.abs(xyz)) > 0.05)

        if is_cartesian:
            x, y, z, rx, ry, rz = target[:6]
            logger.warning(
                "[RESET|MovL] 笛卡尔归位: pos_m=[%.4f, %.4f, %.4f]  euler_rad=[%.4f, %.4f, %.4f]  gripper_mm=%.1f  speed=%d%%",
                x, y, z, rx, ry, rz, gripper_m_target * 1000, reset_speed_pct,
            )
            ok = self._robot._arm.move_l(x, y, z, rx, ry, rz, wait=True, timeout=60.0,
                                          speed_pct=reset_speed_pct)
            if not ok:
                logger.warning("[RESET|MovL] move_l 超时或失败，继续执行后续步骤")
        else:
            q6_target = target[:6]
            logger.warning(
                "[RESET|MovJ] 关节角归位: joints_deg=%s  gripper_mm=%.1f  speed=%d%%",
                np.round(np.rad2deg(q6_target), 2).tolist(),
                gripper_m_target * 1000, reset_speed_pct,
            )
            ok = self._robot._arm.move_j(q6_target, wait=True, timeout=60.0,
                                          speed_pct=reset_speed_pct)
            if not ok:
                logger.warning("[RESET|MovJ] move_j 超时或失败，继续执行后续步骤")

        # ── 恢复正常速度（SpeedFactor 也同步恢复，双保险）────────────────────────
        self._robot._arm.set_speed_factor(100)

        # ── 夹爪归位 ────────────────────────────────────────────────────────────
        self._robot._gripper.set_opening_m(gripper_m_target)
        # 清掉上一回合残留的 policy/teleop 控制缓存，避免下一回合首步沿着旧末状态续跑。
        self._robot.reset_teleop_state()
        self._last_sent_action = None

        time.sleep(0.3)


# ─────────────────────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="越疆 Dobot + 知行夹爪 + 双 RealSense + UMI 介入的 RLT 在线 RL 适配器"
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--task", type=str, default="pick and place the object")
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--max_chunk_steps_per_episode", type=int, default=200)
    parser.add_argument("--idle_sleep_sec", type=float, default=0.02)
    parser.add_argument("--machine_a_ws_url", type=str, default=None)
    parser.add_argument("--actor_service_url", type=str, default=None)
    parser.add_argument("--replay_service_url", type=str, default=None)
    parser.add_argument("--reward_factory", type=str, default=None)
    parser.add_argument("--success_factory", type=str, default=None)
    parser.add_argument("--done_factory", type=str, default=None)
    parser.add_argument("--safe_action_filter_factory", type=str, default=None)
    parser.add_argument(
        "--action_delta_limits",
        type=float,
        nargs=7,
        default=DobotUMIRobotBridge.DEFAULT_ACTION_DELTA_LIMITS,
        help="7D 各维度动作增量上限裁切（6 关节 + 夹爪）。默认启用保守真机安全值。",
    )

    # ── 图像 ──────────────────────────────────────────────────────────────────
    parser.add_argument("--image_h", type=int, default=224)
    parser.add_argument("--image_w", type=int, default=224)
    parser.add_argument("--capture_retries", type=int, default=30)
    parser.add_argument("--capture_retry_sleep_s", type=float, default=0.01)
    parser.add_argument("--disable_obs_stamp_align", action="store_true",
                        help="禁用多话题时间戳对齐，直接取最新帧")
    parser.add_argument("--obs_align_queue_size", type=int, default=200)
    parser.add_argument("--obs_sub_queue_depth", type=int, default=2000)

    # ── 越疆 Dobot SDK（TCP/IP） ───────────────────────────────────────────────
    parser.add_argument("--dobot_ip", type=str, default=DEFAULT_DOBOT_IP)
    parser.add_argument("--dobot_dashboard_port", type=int, default=DEFAULT_DOBOT_DASHBOARD_PORT)
    parser.add_argument("--dobot_feedback_port", type=int, default=DEFAULT_DOBOT_FEEDBACK_PORT)

    # ── 双 RealSense 相机（pyrealsense2 SDK 直驱） ───────────────────────────
    parser.add_argument("--realsense_front_serial", type=str, default="341522300463",
                        help="前置 RealSense 序列号")
    parser.add_argument("--realsense_wrist_serial", type=str, default="427622270458",
                        help="腕部 RealSense 序列号")
    parser.add_argument("--realsense_width", type=int, default=640)
    parser.add_argument("--realsense_height", type=int, default=480)
    parser.add_argument("--realsense_fps", type=int, default=30)

    # ── 知行夹爪 SDK（RS-485） ────────────────────────────────────────────────
    parser.add_argument("--gripper_port", type=str, default=DEFAULT_GRIPPER_PORT)
    parser.add_argument("--gripper_slave_id", type=int, default=DEFAULT_GRIPPER_SLAVE_ID)
    parser.add_argument("--gripper_baudrate", type=int, default=DEFAULT_GRIPPER_BAUDRATE)
    parser.add_argument("--gripper_speed_pct", type=int, default=DEFAULT_GRIPPER_SPEED_PCT)
    parser.add_argument("--gripper_force_pct", type=int, default=DEFAULT_GRIPPER_FORCE_PCT)
    parser.add_argument("--max_gripper_m", type=float, default=GRIPPER_MAX_M)
    parser.add_argument("--reset_speed_pct", type=int, default=20,
                        help="reset 归位时的全局速度比例（1~100%%），默认 20%% 慢速安全归位")

    # ── UMI 人为介入（ROS 话题） ─────────────────────────────────────────────
    parser.add_argument("--umi_vio_pose_topic", type=str, default=DEFAULT_UMI_VIO_POSE_TOPIC,
                        help="UMI VIO 末端位姿话题（geometry_msgs/PoseStamped）")
    parser.add_argument("--umi_encoder_topic", type=str, default=DEFAULT_UMI_ENCODER_TOPIC,
                        help="UMI 手柄夹爪编码器话题（common_msgs/EncoderState）。"
                             "设为空字符串可禁用，改为读取机械臂侧物理夹爪位置。")
    parser.add_argument("--teleop_trigger_service", type=str, default=DEFAULT_TELEOP_TRIGGER_SVC)
    parser.add_argument("--policy_resume_delay_s", type=float, default=1.0)
    parser.add_argument("--start_in_human_mode", action="store_true")
    parser.add_argument(
        "--require_online_approval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warmup 达标后，不自动切换到 RL；必须手动批准后，下一回合才允许 online actor 控制。",
    )
    parser.add_argument("--obs_ready_timeout_s", type=float, default=None)
    parser.add_argument("--ros_domain_id", type=int, default=9,
                        help="ROS_DOMAIN_ID（UMI 默认发布在 domain 9）")

    # ── Teleop 安全限幅 ───────────────────────────────────────────────────────
    parser.add_argument(
        "--teleop_max_delta_rad",
        type=float,
        default=DobotUMIRobotBridge.DEFAULT_TELEOP_MAX_DELTA_RAD,
        help="teleop 每步每关节最大变化量（弧度）。默认 0.03 rad ≈ 1.7°/step。"
             "调大可响应更快；调小更安全。设为 999 可禁用限幅。",
    )
    parser.add_argument(
        "--teleop_max_delta_gripper_m",
        type=float,
        default=DobotUMIRobotBridge.DEFAULT_TELEOP_MAX_DELTA_GRIPPER_M,
        help="teleop 每步夹爪最大变化量（米）。默认 0.005 m/step。",
    )
    parser.add_argument(
        "--teleop_ema_alpha",
        type=float,
        default=0.5,
        help="teleop 路径 ServoJ EMA 平滑系数。默认 0.5，越小越平滑，越大响应越快。",
    )
    parser.add_argument(
        "--policy_max_delta_rad",
        type=float,
        default=DobotUMIRobotBridge.DEFAULT_POLICY_MAX_DELTA_RAD,
        help="policy 每步每关节最大变化量（弧度）。默认 0.03 rad ≈ 1.7°/step，用于限制 ref_chunk/actor 异常突跳。",
    )
    parser.add_argument(
        "--policy_max_delta_gripper_m",
        type=float,
        default=DobotUMIRobotBridge.DEFAULT_POLICY_MAX_DELTA_GRIPPER_M,
        help="policy 每步夹爪最大变化量（米）。默认 0.01 m/step。",
    )
    parser.add_argument(
        "--policy_ema_alpha",
        type=float,
        default=DobotUMIRobotBridge.DEFAULT_POLICY_EMA_ALPHA,
        help="policy 路径 ServoJ EMA 平滑系数。默认 0.25，越小越平滑但延迟越大。",
    )
    parser.add_argument(
        "--policy_deadband_rad",
        type=float,
        default=DobotUMIRobotBridge.DEFAULT_POLICY_DEADBAND_RAD,
        help="policy 路径关节死区。小于该弧度阈值的变化视为推理噪声并保持上一平滑目标；设 0 禁用。",
    )
    parser.add_argument(
        "--policy_gripper_ema_alpha",
        type=float,
        default=DobotUMIRobotBridge.DEFAULT_POLICY_GRIPPER_EMA_ALPHA,
        help="policy 路径夹爪 EMA 平滑系数。默认 0.2，越小越平滑但闭合响应更慢。",
    )
    parser.add_argument(
        "--policy_gripper_deadband_m",
        type=float,
        default=DobotUMIRobotBridge.DEFAULT_POLICY_GRIPPER_DEADBAND_M,
        help="policy 路径夹爪死区。小于该开合变化时保持上一目标，默认 0.003 m；设 0 禁用。",
    )

    # ── 训练控制 ─────────────────────────────────────────────────────────────
    parser.add_argument("--step_trace_stride", type=int, default=None)
    parser.add_argument("--eval_actor_only", action="store_true")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    system = _override_system_urls(load_system_config_yaml(args.config), args)

    # ── 步长 / 确定性覆盖 ─────────────────────────────────────────────────────
    configured_stride = int(system.env_driver.step_trace_stride)
    override_stride = configured_stride if args.step_trace_stride is None else max(int(args.step_trace_stride), 0)
    effective_step_trace_stride = 0 if args.eval_actor_only else override_stride
    effective_actor_deterministic = True if args.eval_actor_only else system.env_driver.actor_deterministic
    system = dataclasses.replace(
        system,
        env_driver=dataclasses.replace(
            system.env_driver,
            actor_deterministic=effective_actor_deterministic,
            step_trace_stride=effective_step_trace_stride,
        ),
    )

    log_path = setup_process_logging("dobot_umi_ros", system, console_level=logging.INFO)

    reward_fn     = _load_callable(args.reward_factory)     or _default_reward_fn
    success_fn    = _load_callable(args.success_factory)    or _default_success_fn
    done_fn       = _load_callable(args.done_factory)       or _default_done_fn
    safe_action_filter = _load_callable(args.safe_action_filter_factory)

    task_state         = TaskState(args.task)
    intervention_state = HumanInterventionState(policy_enabled=not args.start_in_human_mode)

    # ── ROS 初始化 ────────────────────────────────────────────────────────────
    # UMI 使用 ROS_DOMAIN_ID=9 发布 VIO 位姿话题，必须在同一 domain 内才能订阅
    rclpy.init(domain_id=args.ros_domain_id)

    # 两路 RealSense 相机（pyrealsense2 SDK 直驱）
    if not _HAS_REALSENSE_RECORDER:
        raise ImportError("RealSenseImageRecorder 不可用，请确认 examples/dobot_umi/robot_utils.py 可导入")
    cam_recorder = RealSenseImageRecorder(
        serial_front=args.realsense_front_serial,
        serial_wrist=args.realsense_wrist_serial,
        width=args.realsense_width,
        height=args.realsense_height,
        fps=args.realsense_fps,
    )
    cam_recorder.start()

    # SDK 直驱：越疆机械臂 + 知行夹爪
    arm = DobotSDKArm(
        ip=args.dobot_ip,
        dashboard_port=args.dobot_dashboard_port,
        feedback_port=args.dobot_feedback_port,
    )
    gripper = ZhixingSDKGripper(
        port=args.gripper_port,
        slave_id=args.gripper_slave_id,
        baudrate=args.gripper_baudrate,
        speed_pct=args.gripper_speed_pct,
        force_pct=args.gripper_force_pct,
    )
    arm.connect()
    arm.enable()
    gripper.init()

    # UMI VIO 位姿录制（订阅 /umi1/vio/pose，ROS 话题）
    # 负责增量映射：UMI 手柄运动 → 机械臂关节角目标
    _r_align_flat = getattr(system.env_driver, "umi_to_ee_R_align", None)
    _R_align = (np.asarray(_r_align_flat, dtype=np.float64).reshape(3, 3)
                if _r_align_flat is not None else None)
    _r_align_rot_flat = getattr(system.env_driver, "umi_to_ee_R_align_rot", None)
    _R_align_rot = (np.asarray(_r_align_rot_flat, dtype=np.float64).reshape(3, 3)
                    if _r_align_rot_flat is not None else None)
    _motion_scale = float(getattr(system.env_driver, "umi_motion_scale", 1.0))
    _smooth_alpha = float(getattr(system.env_driver, "umi_smooth_alpha", 1.0))
    umi_pose_recorder = UMIPoseRecorder(
        pose_topic=args.umi_vio_pose_topic,
        arm=arm,
        gripper=gripper,
        max_gripper_m=args.max_gripper_m,
        R_align=_R_align,
        R_align_rot=_R_align_rot,
        motion_scale=_motion_scale,
        smooth_alpha=_smooth_alpha,
        encoder_topic=args.umi_encoder_topic or None,
    )
    umi_pose_recorder.bind_policy_enabled_getter(intervention_state.is_policy_enabled)

    # UMI 触发服务（切换策略/人工模式，切换时自动对齐参考帧）
    robot = DobotUMIRobotBridge(args, cam_recorder, arm, gripper)
    teleop_node = DobotUMITeleopTriggerNode(
        intervention_state=intervention_state,
        service_name=args.teleop_trigger_service,
        resume_delay_s=args.policy_resume_delay_s,
        gripper_streamer=None,  # SDK 直驱无需 streamer
        umi_pose_recorder=umi_pose_recorder,
        robot_bridge=robot,
    )

    nodes: list[Node] = [umi_pose_recorder, teleop_node]

    runtime_context = RolloutRuntimeContext(
        system=system,
        obs_node=None,  # type: ignore[arg-type]  # Dobot 版本使用 RealSense SDK 直驱，无 ROS obs buffer
        task_state=task_state,
        intervention_state=intervention_state,
        robot=robot,
    )

    manual_signal_bridge = ManualSignalBridge()
    nodes.extend(_bind_runtime_hook(manual_signal_bridge, runtime_context))
    nodes.extend(_bind_runtime_hook(reward_fn, runtime_context))
    nodes.extend(_bind_runtime_hook(success_fn, runtime_context))
    nodes.extend(_bind_runtime_hook(done_fn, runtime_context))
    nodes.extend(_bind_runtime_hook(safe_action_filter, runtime_context))

    executor = MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    feature_provider = MachineAFeatureClient(
        system.env_driver.machine_a_ws_url,
        connect_timeout_sec=system.env_driver.machine_a_connect_timeout_sec,
        recv_timeout_sec=system.env_driver.machine_a_recv_timeout_sec,
        retry_interval_sec=system.env_driver.machine_a_retry_interval_sec,
    )
    replay_client = (
        NullReplayClient()
        if args.eval_actor_only
        else ReplayClient(
            system.env_driver.replay_service_url,
            timeout_sec=system.env_driver.replay_request_timeout_sec,
        )
    )
    min_online_actor_version = 0 if args.eval_actor_only else _resolve_min_online_actor_version(system)
    learner_status_path = metrics_path_for(system, "learner_status.json")

    phase_controller = (
        StaticOnlinePhaseController()
        if args.eval_actor_only
        else RolloutPhaseController(
            replay_client,
            system.rl.warmup_min_size,
            min_online_actor_version=min_online_actor_version,
            require_online_approval=args.require_online_approval,
            logger_=logger,
        )
    )
    base_actor_client = ActorClient(
        system.env_driver.actor_service_url,
        timeout_sec=system.env_driver.actor_request_timeout_sec,
    )
    phase_controller.bind_actor_version_getter(base_actor_client.get_actor_param_version)
    phase_controller.bind_learner_status_getter(_make_learner_status_reader(learner_status_path))
    phase_controller.bind_online_approval_getter(runtime_context.has_online_approval)
    phase_controller.bind_online_approval_consumer(runtime_context.consume_online_approval)

    actor_client = PhaseAwareActorClient(base_actor_client, phase_controller, runtime_context)

    # PikaChunkEnvAdapter 复用：human_action_recorder 传入 UMI 位姿录制节点
    env = DobotUMIEnvAdapter(
        system=system,
        robot=robot,
        task_state=task_state,
        intervention_state=intervention_state,
        human_action_recorder=umi_pose_recorder,   # ← 使用 UMIPoseRecorder（接口兼容）
        phase_controller=phase_controller,
        runtime_context=runtime_context,
        reward_fn=reward_fn,
        success_fn=success_fn,
        done_fn=done_fn,
        safe_action_filter=safe_action_filter,
        max_chunk_steps_per_episode=args.max_chunk_steps_per_episode,
        idle_sleep_sec=args.idle_sleep_sec,
        action_delta_limits=args.action_delta_limits,
        resize_hw=(args.image_h, args.image_w),
        obs_ready_timeout_s=args.obs_ready_timeout_s,
    )

    driver = EnvDriver(
        env=env,
        feature_provider=feature_provider,
        actor_client=actor_client,
        replay_client=replay_client,
        rl_config=system.rl,
        env_config=system.env_driver,
        eval_actor_only=args.eval_actor_only,
        metrics_path=str(metrics_path_for(system, "robot_rollout_metrics.jsonl")),
    )

    logger.info("启动 Dobot+UMI robot rollout, log=%s config=%s", log_path, args.config)
    logger.info("Machine A ws: %s", system.env_driver.machine_a_ws_url)
    logger.info("Actor service: %s", system.env_driver.actor_service_url)
    logger.info("Replay service: %s", system.env_driver.replay_service_url)
    logger.info("控制频率: %.2f Hz", system.env_driver.control_frequency_hz)
    logger.info("Chunk 执行长度: %d", system.env_driver.chunk_exec_horizon)
    logger.info("任务模式: %s", system.env_driver.task_mode)
    logger.info("仅评估模式: %s", args.eval_actor_only)
    logger.info("Actor 确定性: %s", system.env_driver.actor_deterministic)
    logger.info("Step trace stride: %s", system.env_driver.step_trace_stride)
    logger.info(
        "手动服务 next=%s success=%s failure=%s done=%s critical=%s toggle=%s actor=%s base=%s",
        REQUEST_NEXT_EPISODE_SERVICE, RECORD_SUCCESS_SERVICE, RECORD_FAILURE_SERVICE,
        RECORD_DONE_SERVICE, ENTER_CRITICAL_PHASE_SERVICE, TOGGLE_CRITICAL_PHASE_SERVICE,
        SET_CRITICAL_POLICY_ACTOR_SERVICE, SET_CRITICAL_POLICY_BASE_SERVICE,
    )

    try:
        driver.run_forever(num_episodes=args.num_episodes)
    except KeyboardInterrupt:
        logger.info("接收到 KeyboardInterrupt，正在关闭...")
    finally:
        robot.shutdown()
        try:
            cam_recorder.stop()
        except Exception:
            pass
        executor.shutdown()
        for node in nodes:
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
