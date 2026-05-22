"""robot_utils.py — 越疆 Dobot 机械臂 + 知行夹爪 + 双 RealSense 相机工具类。

硬件接入方式：
  - Dobot 机械臂：SDK 直驱（TCP/IP，DobotApiDashboard + DobotApiFeedBack）
  - 知行夹爪：SDK 直驱（RS-485 串口，MotorController）
  - RealSense 相机：ROS 话题（/cam_front, /cam_wrist）
  - UMI 示教设备：ROS 话题（/umi/human_action）

依赖：
  - rclpy（ROS 2）— 仅用于相机和 UMI 设备
  - sensor_msgs/Image, sensor_msgs/JointState
  - cv_bridge
  - third_party/dobot_umi_sdk/dobot_sdk/dobot_api.py
  - third_party/dobot_umi_sdk/adaptive_sdk/changingtek_p_rtu_Servo.py
"""
# ruff: noqa
from __future__ import annotations

import re
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# ── SDK 路径注入 ──────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SDK_ROOT  = _REPO_ROOT / "third_party" / "dobot_umi_sdk"
for _p in (str(_SDK_ROOT), str(_SDK_ROOT / "dobot_sdk"), str(_SDK_ROOT / "adaptive_sdk")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from dobot_sdk.dobot_api import DobotApiDashboard, DobotApiFeedBack
    _HAS_DOBOT_SDK = True
except ImportError:
    _HAS_DOBOT_SDK = False
    DobotApiDashboard = None  # type: ignore
    DobotApiFeedBack = None   # type: ignore

try:
    from adaptive_sdk.changingtek_p_rtu_Servo import MotorController
    _HAS_MOTOR_SDK = True
except ImportError:
    _HAS_MOTOR_SDK = False
    MotorController = None  # type: ignore

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image as ROSImage
    from sensor_msgs.msg import JointState
    from cv_bridge import CvBridge
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False
    Node = object  # type: ignore[assignment,misc]

from examples.dobot_umi import constants


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

_RESPONSE_PATTERN = re.compile(r"(-?\d+),?\{?([\d.,\-\s]*)\}?")


def _parse_dobot_response(resp: str):
    """解析越疆响应字符串 → (error_code, [values...])"""
    if not resp:
        return None, None
    try:
        resp = resp.strip().rstrip(";")
        m = _RESPONSE_PATTERN.match(resp)
        if m:
            code = int(m.group(1))
            vals_s = m.group(2)
            vals = [float(v.strip()) for v in vals_s.split(",") if v.strip()] if vals_s else None
            return code, vals
        if resp.lstrip("-").isdigit():
            return int(resp), None
    except Exception:
        pass
    return None, None


def _ros_stamp_to_sec(stamp) -> float:
    if stamp is None:
        return 0.0
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _ros_image_to_rgb_u8(msg, resize_hw=None) -> np.ndarray:
    """将 ROS Image 消息转为 HWC uint8 RGB numpy 数组。"""
    bridge = CvBridge()
    encoding = msg.encoding if msg.encoding else "bgr8"
    if "rgb" in encoding.lower():
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
    else:
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize_hw is not None:
        h, w = resize_hw
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 越疆 Dobot 机械臂 SDK 直驱控制器
# ─────────────────────────────────────────────────────────────────────────────

class DobotSDKArm:
    """越疆 Dobot 机械臂 SDK 直驱控制器（TCP/IP）。

    使用 DobotApiDashboard（端口 29999）发送指令，
    使用 DobotApiFeedBack（端口 30004）实时读取关节角。

    Args:
        ip:             机械臂 IP（默认 192.168.5.1）
        dashboard_port: 控制指令端口
        feedback_port:  反馈数据端口
        use_feedback:   是否启用反馈线程（推荐 True）
    """

    def __init__(
        self,
        ip: str = constants.DOBOT_IP,
        dashboard_port: int = constants.DOBOT_DASHBOARD_PORT,
        feedback_port: int = constants.DOBOT_FEEDBACK_PORT,
        use_feedback: bool = constants.DOBOT_USE_FEEDBACK,
    ):
        if not _HAS_DOBOT_SDK:
            raise ImportError(
                "Dobot SDK 不可用。请确认 third_party/dobot_umi_sdk/dobot_sdk/dobot_api.py 存在。"
            )
        self._ip = ip
        self._dashboard_port = dashboard_port
        self._feedback_port = feedback_port
        self._use_feedback = use_feedback

        self._dashboard: Any = None
        self._feedback: Any = None
        self._connected = False

        self._state_lock = threading.Lock()
        self._robot_mode: int = -1
        self._q_actual: list[float] = [0.0] * 6   # 关节角（弧度）

        self._feed_thread: Optional[threading.Thread] = None
        self._feed_running = False

    @staticmethod
    def _is_connected(api_obj) -> bool:
        sock = getattr(api_obj, "socket_dobot", None)
        if sock is None or isinstance(sock, int):
            return False
        try:
            sock.getpeername()
            return True
        except Exception:
            return False

    @staticmethod
    def _close_socket(api_obj) -> None:
        try:
            sock = getattr(api_obj, "socket_dobot", None)
            if sock and not isinstance(sock, int):
                sock.close()
                api_obj.socket_dobot = 0
        except Exception:
            pass

    def connect(self) -> None:
        """连接机械臂；若机械臂未就绪将持续重试。"""
        print(f"[DobotSDKArm] 正在连接 {self._ip}:{self._dashboard_port} ...")
        while True:
            try:
                d = DobotApiDashboard(self._ip, self._dashboard_port)
                if self._is_connected(d):
                    self._dashboard = d
                    break
                self._close_socket(d)
                print("[DobotSDKArm] Dashboard 连接未就绪，3s 后重试...")
            except Exception as e:
                print(f"[DobotSDKArm] Dashboard 连接异常: {e}")
            time.sleep(3)
        print("[DobotSDKArm] Dashboard 已连接")

        if self._use_feedback and DobotApiFeedBack is not None:
            for _ in range(10):
                try:
                    fb = DobotApiFeedBack(self._ip, self._feedback_port)
                    if self._is_connected(fb):
                        self._feedback = fb
                        self._start_feedback()
                        time.sleep(0.5)
                        print("[DobotSDKArm] FeedBack 已连接")
                        break
                    self._close_socket(fb)
                    print("[DobotSDKArm] FeedBack 连接未就绪，3s 后重试...")
                except Exception as e:
                    print(f"[DobotSDKArm] FeedBack 连接异常: {e}")
                time.sleep(3)
            else:
                print("[DobotSDKArm] FeedBack 连接失败（将回退到 GetAngle 查询）")

        self._connected = True

    def disconnect(self) -> None:
        self._feed_running = False
        if self._feed_thread:
            self._feed_thread.join(timeout=1.0)
            self._feed_thread = None
        if self._feedback:
            try:
                del self._feedback
            except Exception:
                pass
            self._feedback = None
        if self._dashboard:
            try:
                self._dashboard.close()
            except Exception:
                pass
            self._dashboard = None
        self._connected = False
        print("[DobotSDKArm] 已断开")

    def _start_feedback(self) -> None:
        self._feed_running = True
        self._feed_thread = threading.Thread(target=self._feedback_loop, daemon=True)
        self._feed_thread.start()

    def _feedback_loop(self) -> None:
        while self._feed_running and self._feedback is not None:
            try:
                data = self._feedback.feedBackData()
                if data is None:
                    continue
                if hex(data["TestValue"][0]) != "0x123456789abcdef":
                    continue
                with self._state_lock:
                    self._robot_mode = int(data["RobotMode"][0])
                    # QActual 为弧度（SDK 已转换）
                    self._q_actual = list(data["QActual"][0])
            except Exception:
                time.sleep(0.05)

    def enable(self) -> bool:
        if not self._connected:
            return False
        resp = self._dashboard.EnableRobot()
        ok = _parse_dobot_response(resp)[0] == 0
        if ok:
            print("[DobotSDKArm] 已使能")
        return ok

    def disable(self) -> bool:
        if not self._connected:
            return False
        resp = self._dashboard.DisableRobot()
        return _parse_dobot_response(resp)[0] == 0

    def clear_error(self) -> bool:
        if not self._connected:
            return False
        resp = self._dashboard.ClearError()
        return _parse_dobot_response(resp)[0] == 0

    def get_joint_angles_rad(self) -> np.ndarray:
        """返回 6 关节角（弧度）。优先使用反馈线程；否则调用 GetAngle。"""
        with self._state_lock:
            if self._q_actual and abs(sum(self._q_actual)) > 1e-9:
                return np.array(self._q_actual, dtype=np.float32)
        # 回退：Dashboard GetAngle（返回角度°）
        resp = self._dashboard.GetAngle()
        _, vals = _parse_dobot_response(resp)
        if vals and len(vals) >= 6:
            return np.deg2rad(np.array(vals[:6], dtype=np.float32))
        return np.zeros(6, dtype=np.float32)

    def get_robot_mode(self) -> int:
        with self._state_lock:
            if self._robot_mode >= 1:
                return self._robot_mode
        resp = self._dashboard.RobotMode()
        code, _ = _parse_dobot_response(resp)
        return code if code is not None else -1

    def is_idle(self) -> bool:
        """RobotMode == 5 表示空闲（运动完成）。"""
        return self.get_robot_mode() == 5

    def servo_j(
        self,
        joints_rad: np.ndarray,
        t: float = 0.1,
        aheadtime: float = 50.0,
        gain: float = 500.0,
    ) -> bool:
        """ServoJ 伺服关节运动（非阻塞，适用于实时控制）。

        Args:
            joints_rad: 6 元素关节角数组（弧度）
        """
        if not self._connected:
            return False
        j = np.asarray(joints_rad, dtype=np.float64).reshape(-1)[:6]
        jd = np.rad2deg(j)  # ServoJ 接受角度（°）
        resp = self._dashboard.ServoJ(jd[0], jd[1], jd[2], jd[3], jd[4], jd[5], t, aheadtime, gain)
        return _parse_dobot_response(resp)[0] == 0

    def move_j(
        self,
        joints_rad: np.ndarray,
        *,
        wait: bool = True,
        timeout: float = 30.0,
    ) -> bool:
        """MovJ 关节运动（默认阻塞等待完成）。

        Args:
            joints_rad: 6 元素关节角数组（弧度）
        """
        if not self._connected:
            return False
        j = np.asarray(joints_rad, dtype=np.float64).reshape(-1)[:6]
        jd = np.rad2deg(j)
        resp = self._dashboard.MovJ(
            jd[0], jd[1], jd[2], jd[3], jd[4], jd[5],
            1,  # coordinateMode=1（关节角模式）
        )
        if _parse_dobot_response(resp)[0] != 0:
            return False
        if wait:
            return self._wait_idle(timeout)
        return True

    def stop(self) -> bool:
        if not self._connected:
            return False
        resp = self._dashboard.Stop()
        return _parse_dobot_response(resp)[0] == 0

    def _wait_idle(self, timeout: float = 30.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            mode = self.get_robot_mode()
            if mode == 5:
                return True
            if mode == 9:
                print("[DobotSDKArm] 运动中止：机械臂报警")
                return False
            time.sleep(0.05)
        print("[DobotSDKArm] 等待运动超时")
        return False

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 知行夹爪 SDK 直驱控制器（RS-485 串口）
# ─────────────────────────────────────────────────────────────────────────────

class ZhixingSDKGripper:
    """知行夹爪 SDK 直驱控制器（RS-485 串口，MotorController）。

    Args:
        port:       串口设备节点（默认 /dev/ttyUSB0）
        slave_id:   Modbus 从机 ID（默认 1）
        baudrate:   波特率（默认 115200）
        speed_pct:  运动速度百分比（1–100）
        force_pct:  夹持力百分比（1–100）
    """

    def __init__(
        self,
        port: str = constants.GRIPPER_SERIAL_PORT,
        slave_id: int = constants.GRIPPER_SLAVE_ID,
        baudrate: int = constants.GRIPPER_BAUDRATE,
        speed_pct: int = constants.GRIPPER_SPEED_PCT,
        force_pct: int = constants.GRIPPER_FORCE_PCT,
    ):
        if not _HAS_MOTOR_SDK:
            raise ImportError(
                "MotorController SDK 不可用。请确认 "
                "third_party/dobot_umi_sdk/adaptive_sdk/changingtek_p_rtu_Servo.py 存在。"
            )
        self._port = port
        self._slave_id = slave_id
        self._baudrate = baudrate
        self._speed_pct = speed_pct
        self._force_pct = force_pct

        self._motor: Any = None
        self._initialized = False
        self._lock = threading.Lock()

    def init(self) -> bool:
        """初始化夹爪，设置速度/力参数。"""
        try:
            self._motor = MotorController(
                self._port, self._slave_id, self._baudrate, 0.5
            )
            self._motor.set_target_speed(self._speed_pct)
            self._motor.set_target_force(self._force_pct)
            self._motor.set_target_acceleration(2000)
            self._motor.set_target_deceleration(2000)
            self._initialized = True
            print(
                f"[ZhixingSDKGripper] 已初始化 {self._port} "
                f"(speed={self._speed_pct}%, force={self._force_pct}%)"
            )
            return True
        except Exception as e:
            print(f"[ZhixingSDKGripper] 初始化失败: {e}")
            return False

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._motor is not None

    def set_opening_m(self, distance_m: float) -> None:
        """设置夹爪开合距离（m，0=关闭，0.085=完全打开）并立即执行。

        距离映射到内部编码器位置 [GRIPPER_POS_OPEN, GRIPPER_POS_CLOSE]。
        """
        d = float(np.clip(distance_m, constants.GRIPPER_CLOSE_M, constants.GRIPPER_OPEN_M))
        ratio = 1.0 - d / constants.GRIPPER_OPEN_M   # 0=张开, 1=闭合
        pos = int(ratio * constants.GRIPPER_POS_CLOSE)
        with self._lock:
            if not self.is_ready:
                return
            self._motor.set_target_position(pos)
            self._motor.trigger_motion()

    def open(self) -> None:
        """完全打开夹爪（非阻塞）。"""
        self.set_opening_m(constants.GRIPPER_OPEN_M)

    def close(self) -> None:
        """完全关闭夹爪（非阻塞）。"""
        self.set_opening_m(constants.GRIPPER_CLOSE_M)

    def stop(self) -> None:
        """就地停止夹爪（以当前位置为目标）。"""
        with self._lock:
            if not self.is_ready:
                return
            try:
                cur = self._motor.read_real_position()
                self._motor.set_target_position(cur)
                self._motor.trigger_motion()
            except Exception as e:
                print(f"[ZhixingSDKGripper] stop 异常: {e}")

    def get_position_m(self) -> float:
        """读取夹爪当前位置（m）。"""
        with self._lock:
            if not self.is_ready:
                return 0.0
            try:
                pos = self._motor.read_real_position()
                ratio = pos / max(constants.GRIPPER_POS_CLOSE, 1)
                return (1.0 - ratio) * constants.GRIPPER_OPEN_M
            except Exception:
                return 0.0

    def set_force(self, force_pct: int) -> None:
        """动态调整夹持力百分比（1–100）。"""
        force_pct = int(np.clip(force_pct, 1, 100))
        with self._lock:
            if not self.is_ready or force_pct == self._force_pct:
                return
            self._force_pct = force_pct
            try:
                self._motor.set_target_force(force_pct)
                self._motor.trigger_motion()
            except Exception as e:
                print(f"[ZhixingSDKGripper] set_force 异常: {e}")

    def release(self) -> None:
        with self._lock:
            if self._motor is not None:
                try:
                    self.stop()
                except Exception:
                    pass
                self._motor = None
                self._initialized = False
        print("[ZhixingSDKGripper] 已释放")

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 图像缓冲节点（两路 RealSense，ROS 话题）
# ─────────────────────────────────────────────────────────────────────────────

class DobotImageRecorder(Node):
    """订阅两路 RealSense 相机话题，缓存最新帧。"""

    def __init__(
        self,
        *,
        cam_front_topic: str = constants.CAM_FRONT_TOPIC,
        cam_wrist_topic: str = constants.CAM_WRIST_TOPIC,
        queue_size: int = 200,
    ):
        super().__init__("dobot_image_recorder")
        self._lock = threading.Lock()
        self._bridge = CvBridge()

        self._front_msg = None
        self._wrist_msg = None
        self._front_queue: deque = deque(maxlen=queue_size)
        self._wrist_queue: deque = deque(maxlen=queue_size)
        self._last_wait_log_ts = 0.0

        qos = qos_profile_sensor_data
        self.create_subscription(ROSImage, cam_front_topic, self._on_front, qos)
        self.create_subscription(ROSImage, cam_wrist_topic, self._on_wrist, qos)

    def _on_front(self, msg) -> None:
        with self._lock:
            self._front_msg = msg
            self._front_queue.append(msg)

    def _on_wrist(self, msg) -> None:
        with self._lock:
            self._wrist_msg = msg
            self._wrist_queue.append(msg)

    def snapshot(self):
        with self._lock:
            return self._front_msg, self._wrist_msg

    @staticmethod
    def _stamp_sec(msg) -> float:
        return _ros_stamp_to_sec(getattr(msg.header, "stamp", None))

    def aligned_snapshot(self):
        with self._lock:
            if not (self._front_queue and self._wrist_queue):
                return None
            frame_time = min(
                self._stamp_sec(self._front_queue[-1]),
                self._stamp_sec(self._wrist_queue[-1]),
            )
            for q in (self._front_queue, self._wrist_queue):
                while len(q) > 1 and self._stamp_sec(q[0]) < frame_time:
                    q.popleft()
                if self._stamp_sec(q[0]) < frame_time:
                    return None
            return self._front_queue[0], self._wrist_queue[0]

    def is_ready(self) -> bool:
        front, wrist = self.snapshot()
        return front is not None and wrist is not None

    def wait_ready(self, timeout_s=None) -> None:
        start = time.time()
        while rclpy.ok():
            if self.is_ready():
                return
            now = time.time()
            if now - self._last_wait_log_ts >= 2.0:
                self._last_wait_log_ts = now
                self.get_logger().warning("等待 RealSense 图像帧 (cam_front / cam_wrist) ...")
            if timeout_s is not None and (now - start) > timeout_s:
                raise RuntimeError("超时：未收到 RealSense 图像")
            time.sleep(0.02)

    def get_images(self, resize_hw=None, *, align_timestamps: bool = True) -> dict[str, np.ndarray]:
        if align_timestamps:
            snap = self.aligned_snapshot()
            if snap is None:
                snap = self.snapshot()
        else:
            snap = self.snapshot()
        front_msg, wrist_msg = snap
        if front_msg is None or wrist_msg is None:
            raise RuntimeError("图像帧尚未就绪，请先调用 wait_ready()")
        return {
            constants.IMAGE_KEY_FRONT: _ros_image_to_rgb_u8(front_msg, resize_hw),
            constants.IMAGE_KEY_WRIST: _ros_image_to_rgb_u8(wrist_msg, resize_hw),
        }


# ─────────────────────────────────────────────────────────────────────────────
# UMI 人为介入动作录制节点（ROS 话题）
# ─────────────────────────────────────────────────────────────────────────────

class UMIHumanActionRecorder(Node):
    """订阅 UMI 设备发布的人类示教动作（7D：6 关节弧度 + 夹爪距离 m）。

    UMI 设备以 JointState 格式发布示教动作到 /umi/human_action，
    本节点缓存最新一帧供 EnvDriver 在人工介入阶段使用。
    """

    def __init__(self, action_topic: str = constants.UMI_HUMAN_ACTION_TOPIC):
        super().__init__("umi_human_action_recorder")
        self._lock = threading.Lock()
        self._latest_action: np.ndarray | None = None
        self._latest_seq: int = -1
        self.create_subscription(JointState, action_topic, self._on_action, 50)
        self.get_logger().info(f"UMI 人类示教动作订阅话题: {action_topic}")

    def _on_action(self, msg) -> None:
        action = np.asarray(msg.position, dtype=np.float32).reshape(-1)
        if action.shape[0] < 7:
            self.get_logger().warning(
                f"UMI 动作维度不足 7（当前 {action.shape[0]}），已忽略"
            )
            return
        with self._lock:
            self._latest_action = action[:7].copy()
            self._latest_seq += 1

    def snapshot_latest(self):
        with self._lock:
            if self._latest_action is None:
                return None, self._latest_seq
            return self._latest_action.copy(), self._latest_seq

    def is_ready(self) -> bool:
        with self._lock:
            return self._latest_action is not None
