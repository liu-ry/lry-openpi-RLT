"""Hardware helpers for dual Rokae AR arms and Zhixing grippers."""
# ruff: noqa
from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from pathlib import Path
from typing import Any

import numpy as np

from examples.rokae_zhixing_dual import constants

# Reuse proven camera and Zhixing gripper implementations from Dobot+UMI.
# Keep OpenCV/RealSense imports lazy; importing them before RokaeAR can make
# the vendor SDK segfault during construction on the hardware runtime.
rs = None  # type: ignore


class ZhixingSDKGripper:
    """Lazy proxy for the Dobot+UMI Zhixing gripper helper.

    Importing that module pulls in scipy/ROS native extensions. Delay it until
    after RokaeAR construction so the Rokae SDK initializes in a clean process,
    matching the working umi-teleop startup order.
    """

    def __init__(self, *args, **kwargs) -> None:
        from examples.dobot_umi.robot_utils import ZhixingSDKGripper as _Impl

        self._impl = _Impl(*args, **kwargs)

    def set_raw_position(self, raw_position: float) -> None:
        """Set Zhixing SDK raw target position directly."""
        pos = int(np.clip(raw_position, constants.GRIPPER_POS_OPEN, constants.GRIPPER_POS_CLOSE))
        impl = self._impl
        with impl._lock:
            if not impl.is_ready:
                return
            impl._motor.set_target_position(pos)
            impl._motor.trigger_motion()

    def get_raw_position(self) -> float:
        """Read Zhixing SDK raw position directly."""
        impl = self._impl
        with impl._lock:
            if not impl.is_ready:
                return 0.0
            try:
                return float(impl._motor.read_real_position())
            except Exception:
                return 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)


class VitAITactileCamera:
    """Lazy proxy for the Dobot+UMI tactile camera helper."""

    def __init__(self, *args, **kwargs) -> None:
        from examples.dobot_umi.robot_utils import VitAITactileCamera as _Impl

        self._impl = _Impl(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)


def _load_pyrokae(sdk_python_dir: str | None = None) -> Any:
    sdk_dir = Path(sdk_python_dir or constants.ROKAE_SDK_PYTHON_DIR)
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "Rokae pyrokae SDK is vendored as pyrokae.cpython-310-x86_64-linux-gnu.so. "
            f"Current Python is {sys.version.split()[0]}. Run real hardware with Python 3.10 "
            "or rebuild/replace the Rokae SDK for this interpreter."
        )
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


def preload_pyrokae(sdk_python_dir: str | None = None) -> Any:
    """Load the Rokae extension before other native stacks such as JAX."""
    return _load_pyrokae(sdk_python_dir)


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
        print(f"[RokaeSDKArm:{self._name}] connecting remote={self._remote_ip} local={self._local_ip}", flush=True)
        self._robot = self._pyrokae.RokaeAR(self._remote_ip, self._local_ip)
        self._connected = True
        print(f"[RokaeSDKArm:{self._name}] connected", flush=True)

    def enable(self) -> None:
        if not self.is_connected:
            raise RuntimeError(f"{self._name} arm is not connected")
        with self._lock:
            print(f"[RokaeSDKArm:{self._name}] recoverState ...", flush=True)
            self._robot.recoverState()
            print(f"[RokaeSDKArm:{self._name}] setOperationMode(automatic) ...", flush=True)
            self._robot.setOperationMode(self._pyrokae.OperateMode.automatic)
            print(f"[RokaeSDKArm:{self._name}] setMotionControlMode(NrtCommand) ...", flush=True)
            self._robot.setMotionControlMode(self._pyrokae.MotionControlMode.NrtCommand)
            print(f"[RokaeSDKArm:{self._name}] setPower(True) ...", flush=True)
            self._robot.setPower(True)
            self._realtime_enabled = False
        print(f"[RokaeSDKArm:{self._name}] enabled in NrtCommand mode", flush=True)

    def _enable_realtime_motion(self) -> None:
        lookahead = float(self._servo_lookahead)
        if lookahead <= self._servo_dt:
            lookahead = self._servo_dt * 5.0
        kp = float(self._servo_kp)
        if kp < 0:
            kp = 1.0
        self._robot.recoverState()
        self._robot.setOperationMode(self._pyrokae.OperateMode.automatic)
        self._robot.setMotionControlMode(self._pyrokae.MotionControlMode.RtCommand)
        self._robot.setPower(True)
        self._robot.enableRealtimeMotion(self._servo_dt, lookahead, kp)
        self._servo_lookahead = lookahead
        self._servo_kp = kp
        self._realtime_enabled = True

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
                self._enable_realtime_motion()
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
            print(
                f"[RokaeSDKArm:{self._name}] moveJ_joint start speed={self._movej_speed if speed is None else speed} "
                f"target_rad={np.round(target, 5).tolist()}",
                flush=True,
            )
            if self._realtime_enabled:
                self._robot.disableRealtimeMotion()
                self._realtime_enabled = False
            self._robot.recoverState()
            self._robot.setOperationMode(self._pyrokae.OperateMode.automatic)
            self._robot.setMotionControlMode(self._pyrokae.MotionControlMode.NrtCommand)
            self._robot.setPower(True)
            self._robot.moveJ_joint(target.tolist(), self._movej_speed if speed is None else speed, self._movej_zone)
            print(f"[RokaeSDKArm:{self._name}] moveJ_joint command sent", flush=True)
        if wait:
            print(f"[RokaeSDKArm:{self._name}] moveWait timeout={timeout}", flush=True)
            self.move_wait(timeout)
            print(f"[RokaeSDKArm:{self._name}] moveWait done", flush=True)
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
        self._servo_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dual_rokae_servo")

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
        self._servo_executor.shutdown(wait=True, cancel_futures=False)

    def get_joint_angles_rad(self) -> np.ndarray:
        return np.concatenate(
            [self.left.get_joint_angles_rad(), self.right.get_joint_angles_rad()],
            dtype=np.float32,
        )

    def servo_j(self, joints_rad: np.ndarray) -> None:
        q = np.asarray(joints_rad, dtype=np.float32).reshape(-1)
        if q.size != constants.ARM_DOF * 2:
            raise ValueError(f"DualRokaeArm expects {constants.ARM_DOF * 2} joints, got {q.size}")
        futures = (
            self._servo_executor.submit(self.left.servo_j, q[: constants.ARM_DOF].copy()),
            self._servo_executor.submit(self.right.servo_j, q[constants.ARM_DOF :].copy()),
        )
        wait(futures)
        for future in futures:
            exc = future.exception()
            if exc is not None:
                raise RuntimeError(f"dual-arm servo_j failed: {exc}") from exc

    def move_j(
        self,
        joints_rad: np.ndarray,
        *,
        wait: bool = True,
        timeout: float = 30.0,
        speed: float | None = None,
        restore_realtime: bool = True,
    ) -> None:
        q = np.asarray(joints_rad, dtype=np.float32).reshape(-1)
        if q.size != constants.ARM_DOF * 2:
            raise ValueError(f"DualRokaeArm expects {constants.ARM_DOF * 2} joints, got {q.size}")
        try:
            print(
                f"[DualRokaeArm] reset/move_j dispatch speed={speed} "
                f"left_rad={np.round(q[: constants.ARM_DOF], 5).tolist()} "
                f"right_rad={np.round(q[constants.ARM_DOF :], 5).tolist()}",
                flush=True,
            )
            self.left.move_j(q[: constants.ARM_DOF], wait=False, speed=speed, restore_realtime=False)
            self.right.move_j(q[constants.ARM_DOF :], wait=False, speed=speed, restore_realtime=False)
            if wait:
                deadline = time.monotonic() + timeout
                print("[DualRokaeArm] waiting left arm move", flush=True)
                self.left.move_wait(max(0.1, deadline - time.monotonic()))
                print("[DualRokaeArm] left arm move done; waiting right arm move", flush=True)
                self.right.move_wait(max(0.1, deadline - time.monotonic()))
                print("[DualRokaeArm] right arm move done", flush=True)
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
        speed: float | None = None,
        restore_realtime: bool = True,
    ) -> None:
        try:
            left_q = self.left.inverse_kinematics(left_end_pose)
            right_q = self.right.inverse_kinematics(right_end_pose)
            self.move_j(
                np.concatenate([left_q, right_q], dtype=np.float32),
                wait=wait,
                timeout=timeout,
                speed=speed,
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
        global rs
        if rs is None:
            try:
                import pyrealsense2 as _rs
            except ImportError as exc:
                raise ImportError("pyrealsense2 不可用。请安装 pyrealsense2 后再连接相机。") from exc
            rs = _rs  # type: ignore[assignment]
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

    def _camera_settings(self, key: str) -> tuple[float | None, float | None]:
        if key == constants.IMAGE_KEY_FRONT:
            return constants.REALSENSE_FRONT_EXPOSURE, constants.REALSENSE_FRONT_GAIN
        if key in (constants.IMAGE_KEY_LEFT_WRIST, constants.IMAGE_KEY_RIGHT_WRIST):
            return constants.REALSENSE_WRIST_EXPOSURE, constants.REALSENSE_WRIST_GAIN
        return None, None

    def _apply_camera_settings(self, profile: Any, *, key: str, serial: str) -> None:
        exposure, gain = self._camera_settings(key)
        if exposure is None and gain is None:
            return

        def set_and_verify(sensor: Any, option: Any, value: float, label: str) -> float:
            if not sensor.supports(option):
                raise RuntimeError(f"{label} is unsupported")
            option_range = sensor.get_option_range(option)
            if not option_range.min <= value <= option_range.max:
                raise ValueError(
                    f"{label}={value} is outside the supported range "
                    f"[{option_range.min}, {option_range.max}]"
                )
            sensor.set_option(option, value)
            actual = float(sensor.get_option(option))
            if not np.isclose(actual, value):
                raise RuntimeError(f"{label} readback={actual}, requested={value}")
            return actual

        try:
            device = profile.get_device()
            color_sensor = None
            for sensor in device.query_sensors():
                name = sensor.get_info(rs.camera_info.name).lower()
                if "rgb" in name or "color" in name:
                    color_sensor = sensor
                    break
            if color_sensor is None:
                print(f"[MultiRealSenseImageRecorder] warning: no color sensor for {key} serial={serial}", flush=True)
                return
            if exposure is not None and color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            actual_exposure = (
                set_and_verify(color_sensor, rs.option.exposure, float(exposure), "exposure")
                if exposure is not None
                else None
            )
            actual_gain = (
                set_and_verify(color_sensor, rs.option.gain, float(gain), "gain")
                if gain is not None
                else None
            )
            print(
                f"[MultiRealSenseImageRecorder] {key} serial={serial} settings verified "
                f"exposure={actual_exposure} gain={actual_gain}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[MultiRealSenseImageRecorder] warning: failed to set {key} serial={serial} "
                f"exposure={exposure} gain={gain}: {exc}",
                flush=True,
            )

    def _make_pipeline(self, serial: str, *, key: str) -> Any:
        pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, self._width, self._height, rs.format.rgb8, self._fps)
        profile = pipeline.start(cfg)
        self._apply_camera_settings(profile, key=key, serial=serial)
        return pipeline

    def start(self) -> None:
        if self._pipelines:
            return
        resolved = self._resolve_serials()
        serial_to_pipeline: dict[str, Any] = {}
        for key, serial in resolved.items():
            if serial not in serial_to_pipeline:
                serial_to_pipeline[serial] = self._make_pipeline(serial, key=key)
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
        import cv2

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
