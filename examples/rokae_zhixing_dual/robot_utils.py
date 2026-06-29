"""Hardware helpers for dual Rokae AR arms and Zhixing grippers."""
# ruff: noqa
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from examples.rokae_zhixing_dual import constants

# Reuse proven camera and Zhixing gripper implementations from Dobot+UMI.
from examples.dobot_umi.robot_utils import (  # noqa: E402
    VitAITactileCamera,
    ZhixingSDKGripper,
)

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None  # type: ignore


def _load_pyrokae(sdk_python_dir: str | None = None) -> Any:
    sdk_dir = Path(sdk_python_dir or constants.ROKAE_SDK_PYTHON_DIR)
    if str(sdk_dir) not in sys.path:
        sys.path.insert(0, str(sdk_dir))
    try:
        import pyrokae  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyrokae 不可用。请确认珞石 SDK Python 目录正确，并使用与 "
            "pyrokae 扩展匹配的 Python 版本。当前默认路径: "
            f"{sdk_dir}"
        ) from exc
    return pyrokae


class RokaeSDKArm:
    """Thin wrapper around pyrokae.RokaeAR for one 7-DoF arm."""

    def __init__(
        self,
        *,
        remote_ip: str,
        local_ip: str,
        name: str,
        sdk_python_dir: str | None = None,
        servo_dt: float = constants.ROKAE_SERVO_DT,
        servo_lookahead: float = constants.ROKAE_SERVO_LOOKAHEAD,
        servo_kp: float = constants.ROKAE_SERVO_KP,
        movej_speed: float = constants.ROKAE_MOVEJ_SPEED,
        movej_zone: float = constants.ROKAE_MOVEJ_ZONE,
    ) -> None:
        self._remote_ip = remote_ip
        self._local_ip = local_ip
        self._name = name
        self._sdk_python_dir = sdk_python_dir
        self._servo_dt = float(servo_dt)
        self._servo_lookahead = float(servo_lookahead)
        self._servo_kp = float(servo_kp)
        self._movej_speed = float(movej_speed)
        self._movej_zone = float(movej_zone)
        self._robot: Any | None = None
        self._pyrokae: Any | None = None
        self._connected = False
        self._realtime_enabled = False
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._connected and self._robot is not None

    def connect(self) -> None:
        self._pyrokae = _load_pyrokae(self._sdk_python_dir)
        print(f"[RokaeSDKArm:{self._name}] connecting remote={self._remote_ip} local={self._local_ip}")
        self._robot = self._pyrokae.RokaeAR(self._remote_ip, self._local_ip)
        self._connected = True

    def enable(self) -> None:
        if not self.is_connected:
            raise RuntimeError(f"{self._name} arm is not connected")
        with self._lock:
            self._robot.recoverState()
            self._robot.setOperationMode(self._pyrokae.OperateMode.automatic)
            self._robot.setPower(True)
            self._robot.enableRealtimeMotion(
                self._servo_dt,
                self._servo_lookahead,
                self._servo_kp,
            )
            self._realtime_enabled = True
        print(f"[RokaeSDKArm:{self._name}] enabled realtime motion")

    def disable(self) -> None:
        if not self.is_connected:
            return
        with self._lock:
            try:
                if self._realtime_enabled:
                    self._robot.disableRealtimeMotion()
                    self._realtime_enabled = False
            finally:
                try:
                    self._robot.setMotionControlMode(self._pyrokae.MotionControlMode.Idle)
                except Exception:
                    pass

    def disconnect(self) -> None:
        self.disable()
        self._robot = None
        self._connected = False
        print(f"[RokaeSDKArm:{self._name}] disconnected")

    def get_joint_angles_rad(self) -> np.ndarray:
        if not self.is_connected:
            return np.zeros(constants.ARM_DOF, dtype=np.float32)
        with self._lock:
            joints = self._robot.getJointPos()
        return np.asarray(joints[: constants.ARM_DOF], dtype=np.float32)

    def servo_j(self, joints_rad: np.ndarray) -> None:
        if not self.is_connected:
            return
        target = np.asarray(joints_rad, dtype=np.float64).reshape(-1)[: constants.ARM_DOF]
        if target.size != constants.ARM_DOF:
            raise ValueError(f"Rokae servo_j expects {constants.ARM_DOF} joints, got {target.size}")
        with self._lock:
            if not self._realtime_enabled:
                self._robot.enableRealtimeMotion(
                    self._servo_dt,
                    self._servo_lookahead,
                    self._servo_kp,
                )
                self._realtime_enabled = True
            self._robot.servoJ(target.tolist(), self._servo_dt, self._servo_lookahead, self._servo_kp)

    def move_j(
        self,
        joints_rad: np.ndarray,
        *,
        wait: bool = True,
        timeout: float = 30.0,
        speed: float | None = None,
        restore_realtime: bool = True,
    ) -> None:
        if not self.is_connected:
            return
        target = np.asarray(joints_rad, dtype=np.float64).reshape(-1)[: constants.ARM_DOF]
        if target.size != constants.ARM_DOF:
            raise ValueError(f"Rokae move_j expects {constants.ARM_DOF} joints, got {target.size}")
        with self._lock:
            if self._realtime_enabled:
                self._robot.disableRealtimeMotion()
                self._realtime_enabled = False
            self._robot.recoverState()
            self._robot.setOperationMode(self._pyrokae.OperateMode.automatic)
            self._robot.setMotionControlMode(self._pyrokae.MotionControlMode.NrtCommand)
            self._robot.setPower(True)
            self._robot.moveJ_joint(target.tolist(), self._movej_speed if speed is None else speed, self._movej_zone)
        if wait:
            self.move_wait(timeout)
            if restore_realtime:
                self.enable()

    def inverse_kinematics(self, end_pose: np.ndarray) -> np.ndarray:
        if not self.is_connected:
            raise RuntimeError(f"{self._name} arm is not connected")
        pose = np.asarray(end_pose, dtype=np.float64).reshape(-1)
        if pose.size != 6:
            raise ValueError(f"Rokae inverse_kinematics expects 6D pose, got {pose.size}")
        with self._lock:
            joints = self._robot.inverseKinematics(pose.tolist())
        joints_arr = np.asarray(joints[: constants.ARM_DOF], dtype=np.float32)
        if joints_arr.size != constants.ARM_DOF or not np.all(np.isfinite(joints_arr)):
            raise RuntimeError(f"{self._name} inverseKinematics returned invalid joints: {joints}")
        return joints_arr

    def move_j_pose(
        self,
        end_pose: np.ndarray,
        *,
        wait: bool = True,
        timeout: float = 30.0,
        speed: float | None = None,
        restore_realtime: bool = True,
    ) -> None:
        joints = self.inverse_kinematics(end_pose)
        self.move_j(joints, wait=wait, timeout=timeout, speed=speed, restore_realtime=restore_realtime)

    def move_wait(self, timeout: float = 30.0) -> None:
        if not self.is_connected:
            return
        with self._lock:
            self._robot.moveWait(timeout)

    def stop(self) -> None:
        if not self.is_connected:
            return
        with self._lock:
            try:
                self._robot.moveStop()
            except Exception:
                pass

    def get_end_pose(self) -> np.ndarray:
        if not self.is_connected:
            return np.zeros(6, dtype=np.float32)
        with self._lock:
            pose = self._robot.getEndPose()
        return np.asarray(pose[:6], dtype=np.float32)

    def __del__(self) -> None:
        try:
            self.disconnect()
        except Exception:
            pass


class DualRokaeArm:
    """Synchronous command helper for left and right Rokae arms."""

    def __init__(self, left: RokaeSDKArm, right: RokaeSDKArm) -> None:
        self.left = left
        self.right = right

    def connect(self) -> None:
        try:
            self.left.connect()
            self.right.connect()
        except Exception:
            self.disconnect()
            raise

    def enable(self) -> None:
        try:
            self.left.enable()
            self.right.enable()
        except Exception:
            self.disable()
            raise

    def disable(self) -> None:
        errors: list[BaseException] = []
        for arm in (self.left, self.right):
            try:
                arm.disable()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"dual-arm disable failed: {errors[0]}") from errors[0]

    def disconnect(self) -> None:
        for arm in (self.left, self.right):
            try:
                arm.disconnect()
            except Exception:
                pass

    def get_joint_angles_rad(self) -> np.ndarray:
        return np.concatenate(
            [self.left.get_joint_angles_rad(), self.right.get_joint_angles_rad()],
            dtype=np.float32,
        )

    def servo_j(self, joints_rad: np.ndarray) -> None:
        q = np.asarray(joints_rad, dtype=np.float32).reshape(-1)
        if q.size != constants.ARM_DOF * 2:
            raise ValueError(f"DualRokaeArm expects {constants.ARM_DOF * 2} joints, got {q.size}")
        errors: list[BaseException] = []

        def _send(arm: RokaeSDKArm, target: np.ndarray) -> None:
            try:
                arm.servo_j(target)
            except BaseException as exc:  # propagate after both threads finish
                errors.append(exc)

        threads = [
            threading.Thread(target=_send, args=(self.left, q[: constants.ARM_DOF]), daemon=True),
            threading.Thread(target=_send, args=(self.right, q[constants.ARM_DOF :]), daemon=True),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise RuntimeError(f"dual-arm servo_j failed: {errors[0]}") from errors[0]

    def move_j(
        self,
        joints_rad: np.ndarray,
        *,
        wait: bool = True,
        timeout: float = 30.0,
        restore_realtime: bool = True,
    ) -> None:
        q = np.asarray(joints_rad, dtype=np.float32).reshape(-1)
        if q.size != constants.ARM_DOF * 2:
            raise ValueError(f"DualRokaeArm expects {constants.ARM_DOF * 2} joints, got {q.size}")
        try:
            self.left.move_j(q[: constants.ARM_DOF], wait=False, restore_realtime=False)
            self.right.move_j(q[constants.ARM_DOF :], wait=False, restore_realtime=False)
            if wait:
                deadline = time.monotonic() + timeout
                self.left.move_wait(max(0.1, deadline - time.monotonic()))
                self.right.move_wait(max(0.1, deadline - time.monotonic()))
                if restore_realtime:
                    self.left.enable()
                    self.right.enable()
        except Exception:
            self.stop()
            raise

    def move_j_pose(
        self,
        left_end_pose: np.ndarray,
        right_end_pose: np.ndarray,
        *,
        wait: bool = True,
        timeout: float = 30.0,
        restore_realtime: bool = True,
    ) -> None:
        try:
            left_q = self.left.inverse_kinematics(left_end_pose)
            right_q = self.right.inverse_kinematics(right_end_pose)
            self.move_j(
                np.concatenate([left_q, right_q], dtype=np.float32),
                wait=wait,
                timeout=timeout,
                restore_realtime=restore_realtime,
            )
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        self.left.stop()
        self.right.stop()


class MultiRealSenseImageRecorder:
    """RealSense recorder for front, left wrist, and optional right wrist cameras."""

    def __init__(
        self,
        *,
        serial_front: str = constants.REALSENSE_FRONT_SERIAL,
        serial_left_wrist: str = constants.REALSENSE_LEFT_WRIST_SERIAL,
        serial_right_wrist: str = constants.REALSENSE_RIGHT_WRIST_SERIAL,
        width: int = constants.REALSENSE_WIDTH,
        height: int = constants.REALSENSE_HEIGHT,
        fps: int = constants.REALSENSE_FPS,
    ) -> None:
        if rs is None:
            raise ImportError("pyrealsense2 不可用。请安装 pyrealsense2 后再连接相机。")
        self._requested = {
            constants.IMAGE_KEY_FRONT: serial_front,
            constants.IMAGE_KEY_LEFT_WRIST: serial_left_wrist,
            constants.IMAGE_KEY_RIGHT_WRIST: serial_right_wrist,
        }
        self._width = width
        self._height = height
        self._fps = fps
        self._pipelines: dict[str, Any] = {}
        self._latest: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._grab_thread: threading.Thread | None = None
        self._grab_running = False

    @staticmethod
    def _list_connected_serials() -> list[str]:
        ctx = rs.context()
        return [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]

    def _resolve_serials(self) -> dict[str, str]:
        connected = self._list_connected_serials()
        print(f"[MultiRealSenseImageRecorder] detected {len(connected)} devices: {connected}")
        if not connected:
            raise RuntimeError("未检测到任何 RealSense 设备")

        resolved: dict[str, str] = {}
        used: set[str] = set()
        for key, requested in self._requested.items():
            if requested:
                if requested not in connected:
                    raise RuntimeError(
                        f"RealSense serial for {key} not found: {requested}. "
                        f"Connected devices: {connected}"
                    )
                resolved[key] = requested
                used.add(requested)
                continue
            remaining = [sn for sn in connected if sn not in used]
            if remaining:
                resolved[key] = remaining[0]
                used.add(remaining[0])

        if constants.IMAGE_KEY_FRONT not in resolved:
            resolved[constants.IMAGE_KEY_FRONT] = connected[0]
        if constants.IMAGE_KEY_LEFT_WRIST not in resolved:
            resolved[constants.IMAGE_KEY_LEFT_WRIST] = resolved[constants.IMAGE_KEY_FRONT]
        if constants.IMAGE_KEY_RIGHT_WRIST not in resolved:
            resolved[constants.IMAGE_KEY_RIGHT_WRIST] = resolved[constants.IMAGE_KEY_LEFT_WRIST]
        return resolved

    def _make_pipeline(self, serial: str) -> Any:
        pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, self._width, self._height, rs.format.rgb8, self._fps)
        pipeline.start(cfg)
        return pipeline

    def start(self) -> None:
        if self._pipelines:
            return
        resolved = self._resolve_serials()
        serial_to_pipeline: dict[str, Any] = {}
        for key, serial in resolved.items():
            if serial not in serial_to_pipeline:
                serial_to_pipeline[serial] = self._make_pipeline(serial)
            self._pipelines[key] = serial_to_pipeline[serial]
            print(f"[MultiRealSenseImageRecorder] {key} -> {serial}")
        self._grab_running = True
        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._grab_thread.start()

    def stop(self) -> None:
        self._grab_running = False
        if self._grab_thread:
            self._grab_thread.join(timeout=2.0)
            self._grab_thread = None
        stopped: set[int] = set()
        for pipeline in self._pipelines.values():
            ident = id(pipeline)
            if ident in stopped:
                continue
            stopped.add(ident)
            try:
                pipeline.stop()
            except Exception:
                pass
        self._pipelines.clear()
        print("[MultiRealSenseImageRecorder] stopped")

    def _grab_loop(self) -> None:
        while self._grab_running:
            for key, pipeline in list(self._pipelines.items()):
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=200)
                    color = frames.get_color_frame()
                    if color:
                        img = np.asanyarray(color.get_data())
                        with self._lock:
                            self._latest[key] = img
                except Exception:
                    pass

    def is_ready(self) -> bool:
        with self._lock:
            return all(key in self._latest for key in self._pipelines)

    def wait_ready(self, timeout_s: float = 10.0) -> None:
        start = time.time()
        while True:
            if self.is_ready():
                return
            if time.time() - start > timeout_s:
                raise RuntimeError("超时：RealSense 相机未就绪（等待首帧超时）")
            time.sleep(0.05)

    def get_images(self, resize_hw: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
        with self._lock:
            images = {key: value.copy() for key, value in self._latest.items()}
        missing = set(self._pipelines) - set(images)
        if missing:
            raise RuntimeError(f"图像帧尚未就绪: {sorted(missing)}")
        if resize_hw is None:
            return images
        h, w = resize_hw
        return {
            key: cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            for key, img in images.items()
        }

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
