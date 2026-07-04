"""Real environment for dual TianjiMarvin AR arms with Zhixing grippers.

Action space (16D):
    [left_j1..left_j7, left_gripper_m,
     right_j1..right_j7, right_gripper_m]

Observation qpos has the same layout as the action.
"""
# ruff: noqa
from __future__ import annotations

import collections
from collections.abc import Callable
from typing import Any

import dm_env
import numpy as np

from examples.tianji_marvin_dual import constants
from examples.tianji_marvin_dual import robot_utils as _utils


class DualTianjiMarvinRealEnv:
    def __init__(
        self,
        *,
        robot_ip: str = constants.ROBOT_IP,
        tianji_sdk_python_dir: str = constants.TIANJI_SDK_PYTHON_DIR,
        tianji_kinematics_config_path: str | None = constants.TIANJI_KINEMATICS_CONFIG_PATH,
        left_gripper_port: str = constants.LEFT_GRIPPER_SERIAL_PORT,
        left_gripper_slave_id: int = constants.LEFT_GRIPPER_SLAVE_ID,
        right_gripper_port: str = constants.RIGHT_GRIPPER_SERIAL_PORT,
        right_gripper_slave_id: int = constants.RIGHT_GRIPPER_SLAVE_ID,
        gripper_baudrate: int = constants.GRIPPER_BAUDRATE,
        gripper_speed_pct: int = constants.GRIPPER_SPEED_PCT,
        gripper_force_pct: int = constants.GRIPPER_FORCE_PCT,
        cam_front_serial: str = constants.REALSENSE_FRONT_SERIAL,
        cam_left_wrist_serial: str = constants.REALSENSE_LEFT_WRIST_SERIAL,
        cam_right_wrist_serial: str = constants.REALSENSE_RIGHT_WRIST_SERIAL,
        cam_width: int = constants.REALSENSE_WIDTH,
        cam_height: int = constants.REALSENSE_HEIGHT,
        cam_fps: int = constants.REALSENSE_FPS,
        image_resize_hw: tuple[int, int] | None = None,
        obs_ready_timeout_s: float | None = 10.0,
        enable_tactile: bool = False,
        tactile_image_provider: Callable[[], dict[str, np.ndarray]] | None = None,
        reset_joint_positions: list[float] | None = None,
        left_reset_joint_positions: list[float] | None = None,
        right_reset_joint_positions: list[float] | None = None,
        left_reset_end_pose: list[float] | None = None,
        right_reset_end_pose: list[float] | None = None,
        allow_zero_reset_pose: bool = constants.ALLOW_ZERO_RESET_POSE,
        max_joint_delta_rad: float = constants.MAX_JOINT_DELTA_RAD,
        max_gripper_delta_m: float = constants.MAX_GRIPPER_DELTA_M,
    ) -> None:
        if (
            left_gripper_port == right_gripper_port
            and left_gripper_slave_id == right_gripper_slave_id
        ):
            raise ValueError(
                "left and right Zhixing grippers use the same serial port and slave id. "
                "Use separate adapters or distinct Modbus slave ids."
            )
        self._image_resize_hw = image_resize_hw
        self._enable_tactile = enable_tactile
        self._tactile_image_provider = tactile_image_provider
        self._allow_zero_reset_pose = bool(allow_zero_reset_pose)
        self._max_joint_delta_rad = float(max_joint_delta_rad)
        self._max_gripper_delta_m = float(max_gripper_delta_m)
        self._last_sent_qpos: np.ndarray | None = None
        self._left_reset, self._right_reset, self._left_reset_pose, self._right_reset_pose = self._parse_reset(
            reset_joint_positions,
            left_reset_joint_positions,
            right_reset_joint_positions,
            left_reset_end_pose,
            right_reset_end_pose,
        )

        self.arms = _utils.DualTianjiMarvinArm(
            left=_utils.TianjiMarvinSDKArm(
                arm=constants.LEFT_ARM_ID,
                name="left",
            ),
            right=_utils.TianjiMarvinSDKArm(
                arm=constants.RIGHT_ARM_ID,
                name="right",
            ),
            robot_ip=robot_ip,
            sdk_python_dir=tianji_sdk_python_dir,
            config_path=tianji_kinematics_config_path,
        )
        self.left_gripper = _utils.ZhixingSDKGripper(
            port=left_gripper_port,
            slave_id=left_gripper_slave_id,
            baudrate=gripper_baudrate,
            speed_pct=gripper_speed_pct,
            force_pct=gripper_force_pct,
        )
        self.right_gripper = _utils.ZhixingSDKGripper(
            port=right_gripper_port,
            slave_id=right_gripper_slave_id,
            baudrate=gripper_baudrate,
            speed_pct=gripper_speed_pct,
            force_pct=gripper_force_pct,
        )
        self.image_recorder = _utils.MultiRealSenseImageRecorder(
            serial_front=cam_front_serial,
            serial_left_wrist=cam_left_wrist_serial,
            serial_right_wrist=cam_right_wrist_serial,
            width=cam_width,
            height=cam_height,
            fps=cam_fps,
        )
        self.image_recorder.start()
        if obs_ready_timeout_s is not None:
            self.image_recorder.wait_ready(timeout_s=obs_ready_timeout_s)

    @staticmethod
    def _parse_reset(
        reset_joint_positions: list[float] | None,
        left_reset_joint_positions: list[float] | None,
        right_reset_joint_positions: list[float] | None,
        left_reset_end_pose: list[float] | None,
        right_reset_end_pose: list[float] | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if reset_joint_positions is not None:
            arr = np.asarray(reset_joint_positions, dtype=np.float32).reshape(-1)
            if arr.size == constants.ARM_DOF * 2:
                return arr[: constants.ARM_DOF], arr[constants.ARM_DOF :], None, None
            if arr.size == constants.DUAL_ACTION_DIM:
                return (
                    arr[: constants.ARM_DOF],
                    arr[constants.SINGLE_ARM_ACTION_DIM : constants.SINGLE_ARM_ACTION_DIM + constants.ARM_DOF],
                    None,
                    None,
                )
            raise ValueError(
                "reset_joint_positions must be 14 joints or 16 full action values "
                f"for dual TianjiMarvin arms, got {arr.size}"
            )
        if left_reset_joint_positions is not None or right_reset_joint_positions is not None:
            left = np.asarray(left_reset_joint_positions or constants.LEFT_RESET_JOINT_POSITIONS, dtype=np.float32)
            right = np.asarray(right_reset_joint_positions or constants.RIGHT_RESET_JOINT_POSITIONS, dtype=np.float32)
            if left.size != constants.ARM_DOF or right.size != constants.ARM_DOF:
                raise ValueError("left/right reset joint positions must each contain 7 joints")
            return left, right, None, None

        left_pose_raw = left_reset_end_pose if left_reset_end_pose is not None else constants.LEFT_RESET_END_POSE
        right_pose_raw = right_reset_end_pose if right_reset_end_pose is not None else constants.RIGHT_RESET_END_POSE
        if left_pose_raw is None or right_pose_raw is None:
            raise RuntimeError(
                "No calibrated Tianji reset target configured. Provide reset_joint_positions, "
                "left/right_reset_joint_positions, left/right_reset_end_pose, or set "
                "critical_phase_reset_action/full_task_reset_action in the rollout config before moving real hardware."
            )
        left_pose = np.asarray(left_pose_raw, dtype=np.float32)
        right_pose = np.asarray(right_pose_raw, dtype=np.float32)
        if left_pose.size != 6 or right_pose.size != 6:
            raise ValueError("left/right reset end poses must each contain 6 values: [x, y, z, rx, ry, rz]")
        return None, None, left_pose, right_pose

    def connect(self) -> None:
        try:
            self.arms.connect()
            self.arms.enable()
            if not self.left_gripper.init():
                raise RuntimeError("left Zhixing gripper initialization failed")
            if not self.right_gripper.init():
                raise RuntimeError("right Zhixing gripper initialization failed")
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        try:
            self.arms.disable()
        except Exception:
            pass
        self.arms.disconnect()
        self.left_gripper.release()
        self.right_gripper.release()
        try:
            self.image_recorder.stop()
        except Exception:
            pass

    def get_qpos(self) -> np.ndarray:
        left_q = self.arms.left.get_joint_angles_rad()
        right_q = self.arms.right.get_joint_angles_rad()
        left_g = float(self.left_gripper.get_position_m())
        right_g = float(self.right_gripper.get_position_m())
        return np.concatenate([left_q, [left_g], right_q, [right_g]], dtype=np.float32)

    def get_images(self) -> dict[str, np.ndarray]:
        images = self.image_recorder.get_images(resize_hw=self._image_resize_hw)
        if self._enable_tactile:
            if self._tactile_image_provider is None:
                raise RuntimeError("enable_tactile=True but tactile_image_provider is not configured")
            images.update(self._tactile_image_provider())
        return images

    def get_observation(self) -> dict[str, Any]:
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos()
        obs["images"] = self.get_images()
        return obs

    def send_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != constants.DUAL_ACTION_DIM:
            raise ValueError(f"dual TianjiMarvin action must be {constants.DUAL_ACTION_DIM}D, got {action.size}")
        if not np.all(np.isfinite(action)):
            self.arms.stop()
            self.left_gripper.stop()
            self.right_gripper.stop()
            raise ValueError("dual TianjiMarvin action contains NaN or Inf")

        limited = self._limit_action(action)
        left_q = limited[: constants.ARM_DOF]
        left_g = float(limited[constants.ARM_DOF])
        right_base = constants.SINGLE_ARM_ACTION_DIM
        right_q = limited[right_base : right_base + constants.ARM_DOF]
        right_g = float(limited[right_base + constants.ARM_DOF])

        self.arms.servo_j(np.concatenate([left_q, right_q], dtype=np.float32))
        self.left_gripper.set_opening_m(left_g)
        self.right_gripper.set_opening_m(right_g)
        self._last_sent_qpos = limited.copy()

    def reset(self, *, fake: bool = False) -> dm_env.TimeStep:
        if not fake:
            self._move_to_reset()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=0.0,
            discount=None,
            observation=self.get_observation(),
        )

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        self.send_action(action)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=0.0,
            discount=1.0,
            observation=self.get_observation(),
        )

    def _move_to_reset(self) -> None:
        if self._left_reset is not None and self._right_reset is not None:
            reset_q = np.concatenate([self._left_reset, self._right_reset], dtype=np.float32)
            if not self._allow_zero_reset_pose and np.allclose(reset_q, 0.0, atol=1e-6):
                raise RuntimeError(
                    "Refusing to move both TianjiMarvin arms to all-zero reset joints. "
                    "Set calibrated reset joints or use the configured 6D end-pose reset. "
                    "Only set allow_zero_reset_pose=True after validating that zero joints are physically safe."
                )
            self.arms.move_j(reset_q, wait=True, timeout=60.0)
        elif self._left_reset_pose is not None and self._right_reset_pose is not None:
            self.arms.move_j_pose(self._left_reset_pose, self._right_reset_pose, wait=True, timeout=60.0)
        else:
            raise RuntimeError("No valid TianjiMarvin reset target configured")
        self.left_gripper.open()
        self.right_gripper.open()
        self._last_sent_qpos = None

    def _limit_action(self, action: np.ndarray) -> np.ndarray:
        desired = action.astype(np.float32, copy=True)
        desired[constants.ARM_DOF] = np.clip(
            desired[constants.ARM_DOF],
            constants.GRIPPER_CLOSE_M,
            constants.GRIPPER_OPEN_M,
        )
        right_g_idx = constants.SINGLE_ARM_ACTION_DIM + constants.ARM_DOF
        desired[right_g_idx] = np.clip(
            desired[right_g_idx],
            constants.GRIPPER_CLOSE_M,
            constants.GRIPPER_OPEN_M,
        )

        reference = self._last_sent_qpos
        if reference is None:
            reference = self.get_qpos()
        limited = desired.copy()
        left_joint = slice(0, constants.ARM_DOF)
        right_joint = slice(constants.SINGLE_ARM_ACTION_DIM, constants.SINGLE_ARM_ACTION_DIM + constants.ARM_DOF)
        for slc in (left_joint, right_joint):
            delta = np.clip(
                desired[slc] - reference[slc],
                -self._max_joint_delta_rad,
                self._max_joint_delta_rad,
            )
            limited[slc] = reference[slc] + delta
        for idx in (constants.ARM_DOF, right_g_idx):
            delta_g = float(
                np.clip(
                    desired[idx] - reference[idx],
                    -self._max_gripper_delta_m,
                    self._max_gripper_delta_m,
                )
            )
            limited[idx] = float(
                np.clip(
                    reference[idx] + delta_g,
                    constants.GRIPPER_CLOSE_M,
                    constants.GRIPPER_OPEN_M,
                )
            )
        return limited


def make_real_env(**kwargs) -> DualTianjiMarvinRealEnv:
    return DualTianjiMarvinRealEnv(**kwargs)
