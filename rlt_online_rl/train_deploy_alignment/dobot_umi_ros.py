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
  6. get_observation() 返回 cam_front / cam_wrist 图像键。
  7. 默认配置文件指向 configs/tasks/dobot_umi/online_rl.yaml。

所有通用的 RLT 在线训练逻辑（EnvDriver、ReplayClient、ActorClient、
PikaChunkEnvAdapter 等）直接从 pika_sync_ros.py 导入复用，不重复实现。
"""
from __future__ import annotations

import argparse
from collections import deque
import dataclasses
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

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
for _p in (str(_SDK_ROOT), str(_SDK_ROOT / "dobot_sdk"), str(_SDK_ROOT / "adaptive_sdk")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from dobot_sdk.dobot_api import DobotApiDashboard, DobotApiFeedBack
    _HAS_DOBOT_SDK = True
except ImportError:
    _HAS_DOBOT_SDK = False

try:
    from adaptive_sdk.changingtek_p_rtu_Servo import MotorController
    _HAS_MOTOR_SDK = True
except ImportError:
    _HAS_MOTOR_SDK = False

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
DEFAULT_GRIPPER_SPEED_PCT = 5
DEFAULT_GRIPPER_FORCE_PCT = 50
DEFAULT_GRIPPER_POS_CLOSE = 12000   # 编码器完全闭合位置

# RealSense 相机 ROS 话题
DEFAULT_CAM_FRONT_TOPIC    = "/cam_front/color/image_raw"
DEFAULT_CAM_WRIST_TOPIC    = "/cam_wrist/color/image_raw"

# UMI 设备（ROS 话题）
DEFAULT_UMI_ACTION_TOPIC   = "/umi/human_action"
DEFAULT_TELEOP_TRIGGER_SVC = "/umi/teleop_trigger"

# 夹爪物理行程（m）
GRIPPER_MAX_M = 0.085


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
            if not self._missing():
                return
            now = time.time()
            if now - self._last_wait_log_ts >= 2.0:
                self._last_wait_log_ts = now
                self.get_logger().warning(f"等待相机话题，缺少: {self._missing()}")
            if timeout_s is not None and (now - start) > timeout_s:
                raise RuntimeError(f"超时：等待相机话题，缺少: {self._missing()}")
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
                    self._q_actual = list(data["QActual"][0])
            except Exception:
                time.sleep(0.05)

    def disconnect(self) -> None:
        self._feed_running = False
        if self._feed_thread:
            self._feed_thread.join(timeout=1.0)
        if self._feedback:
            try:
                del self._feedback
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
        resp = self._dashboard.ServoJ(jd[0], jd[1], jd[2], jd[3], jd[4], jd[5], t, 50.0, 500.0)
        return _parse_dobot_resp(resp)[0] == 0

    def move_j(self, joints_rad: np.ndarray, *, wait: bool = True, timeout: float = 30.0) -> bool:
        if not self._connected:
            return False
        jd = np.rad2deg(np.asarray(joints_rad, dtype=np.float64).reshape(-1)[:6])
        resp = self._dashboard.MovJ(jd[0], jd[1], jd[2], jd[3], jd[4], jd[5], 1)
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

    def set_opening_m(self, distance_m: float) -> None:
        d = float(np.clip(distance_m, 0.0, GRIPPER_MAX_M))
        ratio = 1.0 - d / GRIPPER_MAX_M
        pos = int(ratio * DEFAULT_GRIPPER_POS_CLOSE)
        with self._lock:
            if self._motor is None:
                return
            self._motor.set_target_position(pos)
            self._motor.trigger_motion()

    def get_position_m(self) -> float:
        with self._lock:
            if self._motor is None:
                return 0.0
            try:
                pos = self._motor.read_real_position()
                ratio = pos / max(DEFAULT_GRIPPER_POS_CLOSE, 1)
                return (1.0 - ratio) * GRIPPER_MAX_M
            except Exception:
                return 0.0

    def open(self) -> None:
        self.set_opening_m(GRIPPER_MAX_M)

    def release(self) -> None:
        with self._lock:
            self._motor = None
        self._front_queue: deque[ROSImage]   = deque(maxlen=qs)
        self._wrist_queue: deque[ROSImage]   = deque(maxlen=qs)
        self._last_wait_log_ts = 0.0
        qos = sub_qos or qos_profile_sensor_data

        self.create_subscription(JointState, joint_topic,    self._on_joint, qos)
        self.create_subscription(ROSImage,   cam_front_topic, self._on_front, qos)
        self.create_subscription(ROSImage,   cam_wrist_topic, self._on_wrist, qos)

    def _on_joint(self, msg: JointState) -> None:
        with self._lock:
            self._joint_msg = msg
            self._joint_queue.append(msg)

    def _on_front(self, msg: ROSImage) -> None:
        with self._lock:
            self._front_msg = msg
            self._front_queue.append(msg)

    def _on_wrist(self, msg: ROSImage) -> None:
        with self._lock:
            self._wrist_msg = msg
            self._wrist_queue.append(msg)

    def snapshot(self) -> tuple[JointState | None, ROSImage | None, ROSImage | None]:
        with self._lock:
            return self._joint_msg, self._front_msg, self._wrist_msg

    @staticmethod
    def _stamp(msg) -> float:
        return _ros_stamp_to_sec(getattr(msg.header, "stamp", None))

    def aligned_snapshot(self) -> tuple[JointState, ROSImage, ROSImage] | None:
        """返回时间戳最近对齐的三元组；若数据不足则返回 None。"""
        with self._lock:
            if not (self._joint_queue and self._front_queue and self._wrist_queue):
                return None
            t_ref = min(
                self._stamp(self._joint_queue[-1]),
                self._stamp(self._front_queue[-1]),
                self._stamp(self._wrist_queue[-1]),
            )
            aligned = []
            for q in (self._joint_queue, self._front_queue, self._wrist_queue):
                while len(q) > 1 and self._stamp(q[0]) < t_ref:
                    q.popleft()
                if self._stamp(q[0]) < t_ref:
                    return None
                aligned.append(q[0])
            return aligned[0], aligned[1], aligned[2]

    def _missing(self) -> list[str]:
        j, f, w = self.snapshot()
        missing = []
        if j is None: missing.append("joint_states")
        if f is None: missing.append("cam_front")
        if w is None: missing.append("cam_wrist")
        return missing

    def wait_ready(self, timeout_s: float | None = None) -> None:
        start = time.time()
        while rclpy.ok():
            if not self._missing():
                return
            now = time.time()
            if now - self._last_wait_log_ts >= 2.0:
                self._last_wait_log_ts = now
                self.get_logger().warning(f"等待 ROS 观测，缺少: {self._missing()}")
            if timeout_s is not None and (now - start) > timeout_s:
                raise RuntimeError(f"超时：等待 ROS 话题，缺少: {self._missing()}")
            time.sleep(0.02)


# ─────────────────────────────────────────────────────────────────────────────
# UMI 人为介入动作录制节点（ROS 话题）
# ─────────────────────────────────────────────────────────────────────────────

class UMIHumanActionRecorder(Node):
    """订阅 UMI 设备发布的 7D 示教动作（6 关节 rad + 夹爪 m）。"""

    def __init__(self, action_topic: str):
        super().__init__("umi_human_action_recorder")
        self._lock = threading.Lock()
        self._latest_action: np.ndarray | None = None
        self._latest_seq: int = -1
        self.create_subscription(JointState, action_topic, self._on_action, 50)
        self.get_logger().info(f"UMI 示教动作话题: {action_topic}")

    def _on_action(self, msg: JointState) -> None:
        action = np.asarray(msg.position, dtype=np.float32).reshape(-1)
        if action.shape[0] < 7:
            self.get_logger().warning(f"UMI 动作维度不足 7，已忽略（当前={action.shape[0]}）")
            return
        with self._lock:
            self._latest_action = action[:7].copy()
            self._latest_seq += 1

    def snapshot_latest(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            if self._latest_action is None:
                return None, self._latest_seq
            return self._latest_action.copy(), self._latest_seq


# ─────────────────────────────────────────────────────────────────────────────
# 机器人桥接（SDK 直驱；相机仍走 ROS 话题）
# ─────────────────────────────────────────────────────────────────────────────

class DobotUMIRobotBridge:
    """将 DobotUMIObsBuffer（相机 ROS） + DobotSDKArm + ZhixingSDKGripper
    封装为 PikaChunkEnvAdapter 所需的 robot 接口。
    """

    def __init__(
        self,
        args: argparse.Namespace,
        obs_node: DobotUMIObsBuffer,
        arm: DobotSDKArm,
        gripper: ZhixingSDKGripper,
    ):
        self._args = args
        self._obs_node = obs_node
        self._arm = arm
        self._gripper = gripper

    def shutdown(self) -> None:
        try:
            self._arm.disable()
        except Exception:
            pass
        self._arm.disconnect()
        self._gripper.release()

    def set_policy_control_active(self, enabled: bool) -> None:
        pass  # SDK 直驱无需暂停流；夹爪每次 send_action 时直接设置

    def wait_for_observation_ready(self, timeout_s: float | None = None) -> None:
        self._obs_node.wait_ready(timeout_s=timeout_s)

    def get_observation(self, resize_hw: tuple[int, int], task: str) -> dict[str, Any]:
        retries = max(int(self._args.capture_retries), 1)
        snap = (
            self._obs_node.snapshot()
            if self._args.disable_obs_stamp_align
            else self._obs_node.aligned_snapshot()
        )
        for _ in range(retries):
            if snap is not None and not self._obs_node._missing():
                break
            time.sleep(self._args.capture_retry_sleep_s)
            snap = (
                self._obs_node.snapshot()
                if self._args.disable_obs_stamp_align
                else self._obs_node.aligned_snapshot()
            )

        if snap is None or self._obs_node._missing():
            snap = self._obs_node.snapshot()
            if self._obs_node._missing():
                raise RuntimeError(f"采集观测失败：{self._obs_node._missing()}")

        front_msg, wrist_msg = snap

        # 关节角通过 SDK 反馈线程直接读取
        q6 = self._arm.get_joint_angles_rad()
        gripper_m = self._gripper.get_position_m()
        state7 = np.concatenate([q6, [gripper_m]], dtype=np.float32)

        return {
            "state": state7,
            "images": {
                "cam_front": _image_msg_to_rgb_u8_hwc(front_msg, resize_hw),
                "cam_wrist": _image_msg_to_rgb_u8_hwc(wrist_msg, resize_hw),
            },
            "prompt": task,
        }

    def send_action(self, action7: np.ndarray) -> None:
        action7 = np.asarray(action7, dtype=np.float32).reshape(-1)
        if action7.shape[0] < 7:
            raise ValueError(f"动作维度不足 7，当前 {action7.shape}")
        q6 = action7[:6]
        gripper_m = float(np.clip(action7[6], 0.0, self._args.max_gripper_m))
        # ServoJ 伺服关节控制（非阻塞）
        self._arm.servo_j(q6, t=0.1)
        # 直接设置夹爪位置
        self._gripper.set_opening_m(gripper_m)


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
        default=None,
        help="7D 各维度动作增量上限裁切（6 关节 + 夹爪）",
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

    # ── 双 RealSense 话题（ROS） ─────────────────────────────────────────────
    parser.add_argument("--cam_front_topic", type=str, default=DEFAULT_CAM_FRONT_TOPIC)
    parser.add_argument("--cam_wrist_topic", type=str, default=DEFAULT_CAM_WRIST_TOPIC)

    # ── 知行夹爪 SDK（RS-485） ────────────────────────────────────────────────
    parser.add_argument("--gripper_port", type=str, default=DEFAULT_GRIPPER_PORT)
    parser.add_argument("--gripper_slave_id", type=int, default=DEFAULT_GRIPPER_SLAVE_ID)
    parser.add_argument("--gripper_baudrate", type=int, default=DEFAULT_GRIPPER_BAUDRATE)
    parser.add_argument("--gripper_speed_pct", type=int, default=DEFAULT_GRIPPER_SPEED_PCT)
    parser.add_argument("--gripper_force_pct", type=int, default=DEFAULT_GRIPPER_FORCE_PCT)
    parser.add_argument("--max_gripper_m", type=float, default=GRIPPER_MAX_M)

    # ── UMI 人为介入（ROS 话题） ─────────────────────────────────────────────
    parser.add_argument("--umi_action_topic", type=str, default=DEFAULT_UMI_ACTION_TOPIC)
    parser.add_argument("--teleop_trigger_service", type=str, default=DEFAULT_TELEOP_TRIGGER_SVC)
    parser.add_argument("--policy_resume_delay_s", type=float, default=1.0)
    parser.add_argument("--start_in_human_mode", action="store_true")
    parser.add_argument("--obs_ready_timeout_s", type=float, default=None)

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
    rclpy.init()

    obs_sub_qos = _build_obs_subscription_qos(
        capture_like=False,
        depth=args.obs_sub_queue_depth,
    )

    # 两路相机缓冲（ROS 话题）
    obs_node = DobotUMIObsBuffer(
        cam_front_topic=args.cam_front_topic,
        cam_wrist_topic=args.cam_wrist_topic,
        sync_queue_size=args.obs_align_queue_size,
        sub_qos=obs_sub_qos,
    )

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

    # UMI 示教动作录制（订阅 /umi/human_action，ROS 话题）
    umi_action_recorder = UMIHumanActionRecorder(action_topic=args.umi_action_topic)

    # UMI 触发服务（切换策略/人工模式）
    teleop_node = TeleopTriggerNode(
        intervention_state=intervention_state,
        service_name=args.teleop_trigger_service,
        resume_delay_s=args.policy_resume_delay_s,
        gripper_streamer=None,  # SDK 直驱无需 streamer
    )

    nodes: list[Node] = [obs_node, umi_action_recorder, teleop_node]

    robot = DobotUMIRobotBridge(args, obs_node, arm, gripper)
    runtime_context = RolloutRuntimeContext(
        system=system,
        obs_node=obs_node,
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
            logger_=logger,
        )
    )
    base_actor_client = ActorClient(
        system.env_driver.actor_service_url,
        timeout_sec=system.env_driver.actor_request_timeout_sec,
    )
    phase_controller.bind_actor_version_getter(base_actor_client.get_actor_param_version)
    phase_controller.bind_learner_status_getter(_make_learner_status_reader(learner_status_path))

    actor_client = PhaseAwareActorClient(base_actor_client, phase_controller, runtime_context)

    # PikaChunkEnvAdapter 复用：human_action_recorder 传入 UMI 录制节点
    env = PikaChunkEnvAdapter(
        system=system,
        robot=robot,
        task_state=task_state,
        intervention_state=intervention_state,
        human_action_recorder=umi_action_recorder,   # ← 关键：使用 UMI 录制节点
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
