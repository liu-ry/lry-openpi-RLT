"""robot_utils.py — 越疆 Dobot 机械臂 + 知行夹爪 + 双 RealSense 相机工具类。

硬件接入方式：
  - Dobot 机械臂：SDK 直驱（TCP/IP，DobotApiDashboard + DobotApiFeedBack）
  - 知行夹爪：SDK 直驱（RS-485 串口，MotorController）
  - RealSense 相机：pyrealsense2 SDK 直驱（不依赖 ROS）
  - UMI 示教设备：ROS 话题（/umi/human_action）

依赖：
  - pyrealsense2 — 相机直驱
  - rclpy（ROS 2）— 仅 UMI 设备
  - sensor_msgs/JointState, geometry_msgs/PoseStamped
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
    from dobot_api import DobotApiDashboard, DobotApiFeedBack
    _HAS_DOBOT_SDK = True
except ImportError:
    _HAS_DOBOT_SDK = False
    DobotApiDashboard = None  # type: ignore
    DobotApiFeedBack = None   # type: ignore

try:
    from changingtek_p_rtu_Servo import MotorController
    _HAS_MOTOR_SDK = True
except ImportError:
    _HAS_MOTOR_SDK = False
    MotorController = None  # type: ignore

try:
    import pyrealsense2 as rs
    _HAS_REALSENSE = True
except ImportError:
    _HAS_REALSENSE = False
    rs = None  # type: ignore

# ROS — 仅 UMI 示教设备使用
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped as ROSPoseStamped
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False
    Node = object  # type: ignore[assignment,misc]

try:
    from scipy.spatial.transform import Rotation as _Rotation
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    _Rotation = None  # type: ignore

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
    """将 ROS Image 消息转为 HWC uint8 RGB numpy 数组。（保留，供需要 ROS 桥接的场景使用）"""
    from cv_bridge import CvBridge
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

    def get_end_effector_pose_matrix(self) -> np.ndarray:
        """通过 SDK GetPose() 获取末端执行器在基座坐标系下的 4×4 齐次变换矩阵。

        返回格式：位置单位为米（m）。
        旋转使用 Dobot 惯例的 XYZ Euler 角（rx/ry/rz，单位 deg）。
        若获取失败或 scipy 不可用则返回单位矩阵。
        """
        if not self._connected or _Rotation is None:
            return np.eye(4, dtype=np.float64)
        try:
            resp = self._dashboard.GetPose()
            _, vals = _parse_dobot_response(resp)
            if vals and len(vals) >= 6:
                x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = vals[:6]
                R = _Rotation.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = [x_mm * 1e-3, y_mm * 1e-3, z_mm * 1e-3]
                return T
        except Exception as e:
            print(f"[DobotSDKArm] GetPose 失败: {e}")
        return np.eye(4, dtype=np.float64)

    def inverse_kinematics_from_matrix(
        self, T_target: np.ndarray, q_seed: np.ndarray | None = None
    ) -> np.ndarray | None:
        """逆运动学：给定 4×4 目标位姿矩阵，返回 6D 关节角（弧度）。

        优先使用 Dobot SDK 的 GetInverseKin 命令（CR 系列固件通常支持）。
        若不支持或失败，回退到种子关节角（保持当前位置）。

        Args:
            T_target:  4×4 目标齐次变换矩阵，位置单位为米（m）。
            q_seed:    IK 初始猜测（6D 弧度），用于选解和失败回退。
        Returns:
            6D 关节角（弧度）；失败时返回 q_seed（若有）或当前关节角。
        """
        if not self._connected or _Rotation is None:
            return q_seed.astype(np.float32, copy=True) if q_seed is not None else None
        try:
            x_mm   = float(T_target[0, 3] * 1000.0)
            y_mm   = float(T_target[1, 3] * 1000.0)
            z_mm   = float(T_target[2, 3] * 1000.0)
            euler  = _Rotation.from_matrix(T_target[:3, :3]).as_euler("xyz", degrees=True)
            rx_deg, ry_deg, rz_deg = float(euler[0]), float(euler[1]), float(euler[2])
            resp = self._dashboard.GetInverseKin(x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg)
            _, vals = _parse_dobot_response(resp)
            if vals and len(vals) >= 6:
                return np.deg2rad(np.array(vals[:6], dtype=np.float32))
        except Exception as e:
            print(f"[DobotSDKArm] GetInverseKin 不可用或失败（将保持位置）: {e}")
        if q_seed is not None:
            return q_seed.astype(np.float32, copy=True)
        return self.get_joint_angles_rad()

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
# 图像采集器（两路 RealSense，pyrealsense2 SDK 直驱）
# ─────────────────────────────────────────────────────────────────────────────

class RealSenseImageRecorder:
    """使用 pyrealsense2 SDK 直接采集两路 RealSense 彩色图像。

    不依赖 ROS，无需提前启动相机 launch 文件。

    Args:
        serial_front:  正面相机序列号；为空时使用枚举到的第 1 台设备。
        serial_wrist:  腕部相机序列号；为空时使用枚举到的第 2 台设备。
        width / height / fps:  分辨率和帧率（D405 支持 640×480@30）。
        align_depth:   是否对齐深度到彩色（默认 False，仅用彩色流）。

    使用示例::

        rec = RealSenseImageRecorder(serial_front="...", serial_wrist="...")
        rec.start()
        images = rec.get_images()   # {"cam_front": ndarray, "cam_wrist": ndarray}
        rec.stop()
    """

    def __init__(
        self,
        *,
        serial_front: str = constants.REALSENSE_FRONT_SERIAL,
        serial_wrist: str = constants.REALSENSE_WRIST_SERIAL,
        width: int = constants.REALSENSE_WIDTH,
        height: int = constants.REALSENSE_HEIGHT,
        fps: int = constants.REALSENSE_FPS,
    ):
        if not _HAS_REALSENSE:
            raise ImportError(
                "pyrealsense2 不可用。请安装：pip install pyrealsense2"
            )
        self._serial_front = serial_front
        self._serial_wrist = serial_wrist
        self._width = width
        self._height = height
        self._fps = fps

        self._pipeline_front: Any = None
        self._pipeline_wrist: Any = None
        self._started = False
        self._lock = threading.Lock()

        self._latest_front: Optional[np.ndarray] = None
        self._latest_wrist: Optional[np.ndarray] = None
        self._latest_ts_front: float = 0.0
        self._latest_ts_wrist: float = 0.0

        self._grab_thread: Optional[threading.Thread] = None
        self._grab_running = False

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _list_connected_serials() -> list[str]:
        ctx = rs.context()
        return [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]

    def _make_pipeline(self, serial: str) -> Any:
        pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, self._width, self._height, rs.format.rgb8, self._fps)
        pipeline.start(cfg)
        return pipeline

    @staticmethod
    def _frame_to_rgb_u8(frame, resize_hw=None) -> np.ndarray:
        img = np.asanyarray(frame.get_data())  # HWC uint8 RGB
        if resize_hw is not None:
            h, w = resize_hw
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return img

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """启动两路相机流并开始后台采集线程。"""
        if self._started:
            return
        serials = self._list_connected_serials()
        print(f"[RealSenseImageRecorder] 检测到 {len(serials)} 台设备: {serials}")

        # 解析正面相机序列号
        front_sn = self._serial_front
        if not front_sn:
            if serials:
                front_sn = serials[0]
                print(f"[RealSenseImageRecorder] 未指定 front 序列号，自动选择: {front_sn}")
            else:
                raise RuntimeError("未检测到任何 RealSense 设备")

        # 解析腕部相机序列号
        wrist_sn = self._serial_wrist
        if not wrist_sn:
            remaining = [s for s in serials if s != front_sn]
            if remaining:
                wrist_sn = remaining[0]
                print(f"[RealSenseImageRecorder] 未指定 wrist 序列号，自动选择: {wrist_sn}")
            else:
                print("[RealSenseImageRecorder] ⚠ 仅检测到 1 台设备，front 和 wrist 将使用同一相机")
                wrist_sn = front_sn

        self._pipeline_front = self._make_pipeline(front_sn)
        self._pipeline_wrist = self._make_pipeline(wrist_sn) if wrist_sn != front_sn else None
        self._started = True

        # 后台采集线程
        self._grab_running = True
        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._grab_thread.start()
        print("[RealSenseImageRecorder] 已启动（后台采集线程运行中）")

    def stop(self) -> None:
        """停止采集线程并关闭相机流。"""
        self._grab_running = False
        if self._grab_thread:
            self._grab_thread.join(timeout=2.0)
            self._grab_thread = None
        if self._pipeline_front:
            try:
                self._pipeline_front.stop()
            except Exception:
                pass
            self._pipeline_front = None
        if self._pipeline_wrist:
            try:
                self._pipeline_wrist.stop()
            except Exception:
                pass
            self._pipeline_wrist = None
        self._started = False
        print("[RealSenseImageRecorder] 已停止")

    def _grab_loop(self) -> None:
        """后台线程：持续从相机流抓取最新帧。"""
        while self._grab_running:
            try:
                if self._pipeline_front:
                    frames = self._pipeline_front.wait_for_frames(timeout_ms=200)
                    color = frames.get_color_frame()
                    if color:
                        img = self._frame_to_rgb_u8(color)
                        with self._lock:
                            self._latest_front = img
                            self._latest_ts_front = frames.get_timestamp()
            except Exception:
                pass

            try:
                if self._pipeline_wrist:
                    frames = self._pipeline_wrist.wait_for_frames(timeout_ms=200)
                    color = frames.get_color_frame()
                    if color:
                        img = self._frame_to_rgb_u8(color)
                        with self._lock:
                            self._latest_wrist = img
                            self._latest_ts_wrist = frames.get_timestamp()
                elif self._pipeline_front is None:
                    pass
                else:
                    # 单相机模式：wrist 复用 front
                    with self._lock:
                        self._latest_wrist = self._latest_front
                        self._latest_ts_wrist = self._latest_ts_front
            except Exception:
                pass

    # ── 就绪检查 ──────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        with self._lock:
            return self._latest_front is not None and self._latest_wrist is not None

    def wait_ready(self, timeout_s: float = 10.0) -> None:
        """阻塞等待两路相机均有帧可用。"""
        start = time.time()
        while True:
            if self.is_ready():
                return
            if time.time() - start > timeout_s:
                raise RuntimeError("超时：RealSense 相机未就绪（等待首帧超时）")
            time.sleep(0.05)

    # ── 图像读取 ──────────────────────────────────────────────────────────────

    def get_images(
        self,
        resize_hw: Optional[tuple[int, int]] = None,
        *,
        align_timestamps: bool = True,
    ) -> dict[str, np.ndarray]:
        """返回最新一对彩色图像（HWC uint8 RGB）。

        Args:
            resize_hw:         目标尺寸 (H, W)；None 保持原始分辨率。
            align_timestamps:  当前 SDK 模式下时间戳已由后台线程近实时同步，该参数保留供接口兼容。
        """
        with self._lock:
            front = self._latest_front
            wrist = self._latest_wrist

        if front is None or wrist is None:
            raise RuntimeError("图像帧尚未就绪，请先调用 wait_ready() 或 start()")

        if resize_hw is not None:
            h, w = resize_hw
            front = cv2.resize(front, (w, h), interpolation=cv2.INTER_LINEAR)
            wrist = cv2.resize(wrist, (w, h), interpolation=cv2.INTER_LINEAR)

        return {
            constants.IMAGE_KEY_FRONT: front.copy(),
            constants.IMAGE_KEY_WRIST: wrist.copy(),
        }

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


# 向后兼容别名（旧代码若直接用 DobotImageRecorder 仍可工作，但已不推荐）
DobotImageRecorder = RealSenseImageRecorder


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
           T_ee_target  = T_ee_ref @ delta_T              （相同增量映射到机械臂）
           q_target     = IK(T_ee_target, seed=q_ref)

    输出格式：
        snapshot_latest() → (np.ndarray shape (7,), seq_id)
        其中 7D = [j1..j6 rad, gripper_m]
    """

    def __init__(
        self,
        pose_topic: str,
        arm: "DobotSDKArm",
        gripper: "ZhixingSDKGripper",
        max_gripper_m: float = constants.GRIPPER_OPEN_M,
    ):
        super().__init__("umi_pose_recorder")
        if not _HAS_SCIPY:
            raise ImportError("UMIPoseRecorder 需要 scipy。请 pip install scipy。")
        self._arm = arm
        self._gripper = gripper
        self._max_gripper_m = max_gripper_m
        self._lock = threading.Lock()

        self._latest_T: np.ndarray | None = None
        self._latest_seq: int = -1

        self._T_umi_ref: np.ndarray | None = None
        self._T_ee_ref: np.ndarray | None = None
        self._q_ref: np.ndarray | None = None
        self._last_output: np.ndarray | None = None

        self.create_subscription(ROSPoseStamped, pose_topic, self._on_pose, 50)
        self.get_logger().info(f"UMI VIO 位姿话题订阅: {pose_topic}")

    @staticmethod
    def _pose_msg_to_matrix(msg) -> np.ndarray:
        p = msg.position
        q = msg.orientation
        R = _Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [p.x, p.y, p.z]
        return T

    def _on_pose(self, msg: ROSPoseStamped) -> None:
        T = self._pose_msg_to_matrix(msg.pose)
        with self._lock:
            self._latest_T = T
            self._latest_seq += 1

    def is_ready(self) -> bool:
        with self._lock:
            return self._latest_T is not None

    def set_alignment_reference(self) -> None:
        """对齐参考帧：在用户按 't' 并手动对齐后调用。"""
        with self._lock:
            T_umi = self._latest_T
        if T_umi is None:
            self.get_logger().warning("set_alignment_reference: 尚未收到 UMI 位姿，跳过")
            return
        T_ee = self._arm.get_end_effector_pose_matrix()
        q_now = self._arm.get_joint_angles_rad()
        with self._lock:
            self._T_umi_ref = T_umi.copy()
            self._T_ee_ref  = T_ee.copy()
            self._q_ref     = q_now.copy()
        self.get_logger().info(
            "对齐参考帧已记录 | EE 位置(m): [%.4f, %.4f, %.4f]",
            T_ee[0, 3], T_ee[1, 3], T_ee[2, 3],
        )

    def snapshot_latest(self) -> tuple[np.ndarray | None, int]:
        """返回 (7D [关节角×6 + 夹爪], seq_id)。"""
        with self._lock:
            T_umi_curr = self._latest_T
            seq        = self._latest_seq
            T_umi_ref  = self._T_umi_ref
            T_ee_ref   = self._T_ee_ref
            q_ref      = self._q_ref

        gripper_m = float(np.clip(self._gripper.get_position_m(), 0.0, self._max_gripper_m))

        def _hold() -> np.ndarray:
            q = self._arm.get_joint_angles_rad()
            return np.concatenate([q, [gripper_m]], dtype=np.float32)

        if T_umi_curr is None or T_umi_ref is None or T_ee_ref is None or q_ref is None:
            return _hold(), seq

        delta_T     = np.linalg.inv(T_umi_ref) @ T_umi_curr
        T_ee_target = T_ee_ref @ delta_T
        q_target    = self._arm.inverse_kinematics_from_matrix(T_ee_target, q_seed=q_ref)

        if q_target is None:
            return (self._last_output.copy() if self._last_output is not None else _hold()), seq

        action = np.concatenate([q_target, [gripper_m]], dtype=np.float32)
        with self._lock:
            self._last_output = action.copy()
        return action, seq


# 向后兼容别名
UMIHumanActionRecorder = UMIPoseRecorder
