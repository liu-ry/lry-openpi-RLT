"""Hardware helpers for dual TianjiMarvin AR arms and Zhixing grippers."""
# ruff: noqa
from __future__ import annotations

import threading
import time
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

from examples.tianji_marvin_dual import constants

# Reuse proven camera and Zhixing gripper implementations from Dobot+UMI.
from examples.dobot_umi.robot_utils import (  # noqa: E402
    VitAITactileCamera,
    ZhixingSDKGripper,
)

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None  # type: ignore


def _load_marvin_sdk(sdk_python_dir: str | None = None) -> type[Any]:
    sdk_dir = Path(sdk_python_dir or constants.TIANJI_SDK_PYTHON_DIR).expanduser().resolve()
    sdk_parent = sdk_dir.parent if sdk_dir.name == "tianji_marvin_sdk" else sdk_dir
    for path in (sdk_parent, sdk_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        from tianji_marvin_sdk.concise_marvin_api import MarvinSDK  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "MarvinSDK 不可用。请确认天机 SDK 目录正确，并使用与 "
            "libMarvinSDK/libKine 匹配的 Python 版本。当前默认路径: "
            f"{sdk_dir}"
        ) from exc
    return MarvinSDK


class TianjiMarvinSDKArm:
    """One-arm view over a shared Tianji Marvin controller.

    The MarvinSDK connection owns both physical arms. This wrapper keeps the
    repository-facing interface in radians while the SDK boundary uses degrees.
    """

    def __init__(
        self,
        *,
        arm: str,
        name: str,
        movej_vel_ratio: float = constants.TIANJI_MOVEJ_VEL_RATIO,
        movej_acc_ratio: float = constants.TIANJI_MOVEJ_ACC_RATIO,
    ) -> None:
        if arm not in ("A", "B"):
            raise ValueError(f"Tianji arm must be 'A' or 'B', got {arm!r}")
        self._arm = arm
        self._name = name
        self._movej_vel_ratio = float(movej_vel_ratio)
        self._movej_acc_ratio = float(movej_acc_ratio)
        self._sdk: Any | None = None
        self._lock: threading.RLock = threading.RLock()

    def bind(self, sdk: Any | None, lock: threading.RLock | None = None) -> None:
        self._sdk = sdk
        if lock is not None:
            self._lock = lock

    @property
    def is_connected(self) -> bool:
        return self._sdk is not None

    def connect(self) -> None:
        if not self.is_connected:
            raise RuntimeError("Tianji arm views are connected through DualTianjiMarvinArm.connect()")

    def enable(self) -> None:
        if not self.is_connected:
            raise RuntimeError(f"{self._name} arm is not connected")
        with self._lock:
            self._sdk.set_position_state(self._arm, constants.TIANJI_POSITION_VEL_RATIO, constants.TIANJI_POSITION_ACC_RATIO)
        print(f"[TianjiMarvinSDKArm:{self._name}] enabled position control")

    def disable(self) -> None:
        if not self.is_connected:
            return
        with self._lock:
            try:
                self._sdk.disable(self._arm)
            except Exception:
                pass

    def disconnect(self) -> None:
        self._sdk = None
        print(f"[TianjiMarvinSDKArm:{self._name}] disconnected")

    def get_joint_angles_rad(self) -> np.ndarray:
        if not self.is_connected:
            return np.zeros(constants.ARM_DOF, dtype=np.float32)
        with self._lock:
            joints_deg = self._sdk.get_current_joints(self._arm)
        return np.deg2rad(np.asarray(joints_deg[: constants.ARM_DOF], dtype=np.float32))

    def servo_j(self, joints_rad: np.ndarray) -> None:
        if not self.is_connected:
            return
        target = np.asarray(joints_rad, dtype=np.float64).reshape(-1)[: constants.ARM_DOF]
        if target.size != constants.ARM_DOF:
            raise ValueError(f"TianjiMarvin servo_j expects {constants.ARM_DOF} joints, got {target.size}")
        target_deg = np.rad2deg(target).tolist()
        with self._lock:
            self._sdk.set_joint_position_cmd(self._arm, target_deg)

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
            raise ValueError(f"TianjiMarvin move_j expects {constants.ARM_DOF} joints, got {target.size}")
        target_deg = np.rad2deg(target).tolist()
        with self._lock:
            self._sdk.movej(
                self._arm,
                target_deg,
                vel_ratio=self._movej_vel_ratio if speed is None else float(speed),
                acc_ratio=self._movej_acc_ratio,
                blocking=wait,
            )
        if restore_realtime:
            self.enable()

    def inverse_kinematics(self, end_pose: np.ndarray) -> np.ndarray:
        if not self.is_connected:
            raise RuntimeError(f"{self._name} arm is not connected")
        pose = np.asarray(end_pose, dtype=np.float64).reshape(-1)
        if pose.size != 6:
            raise ValueError(f"TianjiMarvin inverse_kinematics expects 6D pose, got {pose.size}")
        pose_mm_deg = pose.copy()
        pose_mm_deg[:3] *= 1000.0
        pose_mm_deg[3:] = np.rad2deg(pose_mm_deg[3:])
        with self._lock:
            joints = self._sdk.ik(self._arm, pose_mm_deg.tolist())
        joints_arr = np.deg2rad(np.asarray(joints[: constants.ARM_DOF], dtype=np.float32))
        if joints_arr.size != constants.ARM_DOF or not np.all(np.isfinite(joints_arr)):
            raise RuntimeError(f"{self._name} ik returned invalid joints: {joints}")
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
        return

    def stop(self) -> None:
        if not self.is_connected:
            return
        with self._lock:
            try:
                self._sdk.stop_pln(self._arm)
                self._sdk.soft_stop(self._arm)
            except Exception:
                pass

    def get_end_pose(self) -> np.ndarray:
        if not self.is_connected:
            return np.zeros(6, dtype=np.float32)
        with self._lock:
            pose = self._sdk.get_current_tcppose(self._arm)
        pose_arr = np.asarray(pose[:6], dtype=np.float32)
        pose_arr[:3] *= 0.001
        pose_arr[3:] = np.deg2rad(pose_arr[3:])
        return pose_arr

    def __del__(self) -> None:
        try:
            self.disconnect()
        except Exception:
            pass


class DualTianjiMarvinArm:
    """Synchronous command helper for left and right TianjiMarvin arms."""

    def __init__(
        self,
        left: TianjiMarvinSDKArm,
        right: TianjiMarvinSDKArm,
        *,
        robot_ip: str = constants.ROBOT_IP,
        sdk_python_dir: str | None = None,
        config_path: str | None = None,
    ) -> None:
        self.left = left
        self.right = right
        self._robot_ip = robot_ip
        self._sdk_python_dir = sdk_python_dir
        self._config_path = config_path
        self._sdk: Any | None = None
        self._sdk_lock = threading.RLock()

    def connect(self) -> None:
        try:
            sdk_cls = _load_marvin_sdk(self._sdk_python_dir)
            self._sdk = sdk_cls(ip=self._robot_ip, config_path=self._config_path)
            with self._sdk_lock:
                self._sdk.connect()
            self.left.bind(self._sdk, self._sdk_lock)
            self.right.bind(self._sdk, self._sdk_lock)
            print(f"[DualTianjiMarvinArm] connected robot_ip={self._robot_ip}")
        except Exception:
            self.disconnect()
            raise

    def enable(self) -> None:
        try:
            if self._sdk is None:
                raise RuntimeError("Tianji Marvin SDK is not connected")
            with self._sdk_lock:
                self._sdk.set_position_state(None, constants.TIANJI_POSITION_VEL_RATIO, constants.TIANJI_POSITION_ACC_RATIO)
        except Exception:
            self.disable()
            raise

    def disable(self) -> None:
        if self._sdk is None:
            return
        errors: list[BaseException] = []
        with self._sdk_lock:
            for arm_id in ("A", "B"):
                try:
                    self._sdk.disable(arm_id)
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            raise RuntimeError(f"dual-arm disable failed: {errors[0]}") from errors[0]

    def disconnect(self) -> None:
        self.left.bind(None)
        self.right.bind(None)
        if self._sdk is not None:
            try:
                with self._sdk_lock:
                    self._sdk.release()
            except Exception:
                pass
        self._sdk = None

    def get_joint_angles_rad(self) -> np.ndarray:
        return np.concatenate(
            [self.left.get_joint_angles_rad(), self.right.get_joint_angles_rad()],
            dtype=np.float32,
        )

    def servo_j(self, joints_rad: np.ndarray) -> None:
        q = np.asarray(joints_rad, dtype=np.float32).reshape(-1)
        if q.size != constants.ARM_DOF * 2:
            raise ValueError(f"DualTianjiMarvinArm expects {constants.ARM_DOF * 2} joints, got {q.size}")
        self.left.servo_j(q[: constants.ARM_DOF])
        self.right.servo_j(q[constants.ARM_DOF :])

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
            raise ValueError(f"DualTianjiMarvinArm expects {constants.ARM_DOF * 2} joints, got {q.size}")
        try:
            self.left.move_j(q[: constants.ARM_DOF], wait=False, restore_realtime=False)
            self.right.move_j(q[constants.ARM_DOF :], wait=False, restore_realtime=False)
            if wait:
                if self._sdk is None:
                    raise RuntimeError("Tianji Marvin SDK is not connected")
                deadline = time.monotonic() + float(timeout)
                wait_fn = getattr(self._sdk, "_wait_for_motion_complete", None)
                if callable(wait_fn):
                    wait_fn(constants.LEFT_ARM_ID, timeout=max(0.1, deadline - time.monotonic()))
                    wait_fn(constants.RIGHT_ARM_ID, timeout=max(0.1, deadline - time.monotonic()))
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
