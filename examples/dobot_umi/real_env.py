"""real_env.py — 越疆 Dobot + 知行夹爪 + 双 RealSense 相机的底层真实环境。

硬件接入方式：
  - Dobot 机械臂：SDK 直驱（TCP/IP）
  - 知行夹爪：SDK 直驱（RS-485 串口）
  - RealSense 相机：ROS 话题

Action space (7D):
    [joint1, joint2, joint3, joint4, joint5, joint6,  # 绝对关节角（rad）
     gripper_m]                                         # 夹爪开合距离（m，0=关闭，0.085=完全打开）

Observation space:
    {
        "qpos":   np.ndarray shape (7,)  — [joint1..6 (rad), gripper_m]
        "images": {
            "cam_front": np.ndarray (H, W, 3) uint8
            "cam_wrist": np.ndarray (H, W, 3) uint8
        }
    }
"""
# ruff: noqa
from __future__ import annotations

import collections
import time
from typing import Any

import dm_env
import numpy as np

from examples.dobot_umi import constants
from examples.dobot_umi import robot_utils as _utils


class DobotUMIRealEnv:
    """真实硬件环境：越疆 Dobot CR 系列机械臂 + 知行夹爪 + 双 RealSense。

    使用方式::

        env = DobotUMIRealEnv()
        env.connect()
        ts  = env.reset()
        while True:
            action = policy(ts.observation)   # shape (7,)
            ts = env.step(action)
        env.disconnect()
    """

    def __init__(
        self,
        *,
        # Dobot SDK 参数
        dobot_ip: str = constants.DOBOT_IP,
        dobot_dashboard_port: int = constants.DOBOT_DASHBOARD_PORT,
        dobot_feedback_port: int = constants.DOBOT_FEEDBACK_PORT,
        # 知行夹爪 SDK 参数
        gripper_port: str = constants.GRIPPER_SERIAL_PORT,
        gripper_slave_id: int = constants.GRIPPER_SLAVE_ID,
        gripper_baudrate: int = constants.GRIPPER_BAUDRATE,
        gripper_speed_pct: int = constants.GRIPPER_SPEED_PCT,
        gripper_force_pct: int = constants.GRIPPER_FORCE_PCT,
        # ROS 相机话题参数
        init_ros_node: bool = True,
        cam_front_topic: str = constants.CAM_FRONT_TOPIC,
        cam_wrist_topic: str = constants.CAM_WRIST_TOPIC,
        image_resize_hw: tuple[int, int] | None = None,
        align_image_timestamps: bool = True,
        obs_ready_timeout_s: float | None = 10.0,
        # 复位姿态
        reset_joint_positions: list[float] | None = None,
    ):
        self._reset_joints = (
            list(reset_joint_positions[:6])
            if reset_joint_positions
            else list(constants.DEFAULT_RESET_JOINT_POSITIONS)
        )
        self._image_resize_hw = image_resize_hw
        self._align_timestamps = align_image_timestamps

        # ── 机械臂 SDK ────────────────────────────────────────────────────────
        self.arm = _utils.DobotSDKArm(
            ip=dobot_ip,
            dashboard_port=dobot_dashboard_port,
            feedback_port=dobot_feedback_port,
        )

        # ── 夹爪 SDK ──────────────────────────────────────────────────────────
        self.gripper = _utils.ZhixingSDKGripper(
            port=gripper_port,
            slave_id=gripper_slave_id,
            baudrate=gripper_baudrate,
            speed_pct=gripper_speed_pct,
            force_pct=gripper_force_pct,
        )

        # ── ROS 相机节点 ──────────────────────────────────────────────────────
        import rclpy as _rclpy
        if init_ros_node and not _rclpy.ok():
            _rclpy.init()
        self.image_recorder = _utils.DobotImageRecorder(
            cam_front_topic=cam_front_topic,
            cam_wrist_topic=cam_wrist_topic,
        )

        # 等待相机就绪
        if obs_ready_timeout_s is not None:
            self.image_recorder.wait_ready(timeout_s=obs_ready_timeout_s)

    # ─────────────────────────────────────────────────────────────────────────
    # 连接 / 断开
    # ─────────────────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """连接机械臂和夹爪。阻塞直到硬件就绪。"""
        self.arm.connect()
        self.arm.enable()
        self.gripper.init()

    def disconnect(self) -> None:
        """安全断开机械臂和夹爪。"""
        try:
            self.arm.disable()
        except Exception:
            pass
        self.arm.disconnect()
        self.gripper.release()

    # ─────────────────────────────────────────────────────────────────────────
    # 观测
    # ─────────────────────────────────────────────────────────────────────────

    def get_qpos(self) -> np.ndarray:
        """返回 7D 本体感知状态：[joint1..6 (rad), gripper_m]。"""
        q6 = self.arm.get_joint_angles_rad()
        gripper_m = self.gripper.get_position_m()
        return np.concatenate([q6, [gripper_m]], dtype=np.float32)

    def get_images(self) -> dict[str, np.ndarray]:
        """返回包含 'cam_front' 和 'cam_wrist' 的 RGB uint8 图像字典。"""
        return self.image_recorder.get_images(
            resize_hw=self._image_resize_hw,
            align_timestamps=self._align_timestamps,
        )

    def get_observation(self) -> dict[str, Any]:
        obs = collections.OrderedDict()
        obs["qpos"]   = self.get_qpos()
        obs["images"] = self.get_images()
        return obs

    # ─────────────────────────────────────────────────────────────────────────
    # 动作执行
    # ─────────────────────────────────────────────────────────────────────────

    def send_action(self, action: np.ndarray) -> None:
        """发送 7D 动作：前 6 维为关节角（rad），第 7 维为夹爪距离（m）。"""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        q6 = action[:6]
        gripper_m = float(np.clip(action[6], constants.GRIPPER_CLOSE_M, constants.GRIPPER_OPEN_M))

        # ServoJ 伺服控制机械臂（实时非阻塞）
        self.arm.servo_j(q6, t=constants.DT)

        # 直接设置夹爪位置
        self.gripper.set_opening_m(gripper_m)

    # ─────────────────────────────────────────────────────────────────────────
    # 环境接口（dm_env 风格）
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, *, fake: bool = False) -> dm_env.TimeStep:
        """复位机械臂到默认姿态，返回初始 TimeStep。"""
        if not fake:
            self._move_to_reset()
        obs = self.get_observation()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=0.0,
            discount=None,
            observation=obs,
        )

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        self.send_action(action)
        obs = self.get_observation()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=0.0,
            discount=1.0,
            observation=obs,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 内部辅助
    # ─────────────────────────────────────────────────────────────────────────

    def _move_to_reset(self) -> None:
        """MovJ 复位关节角并阻塞等待到位。"""
        reset_q = np.asarray(self._reset_joints, dtype=np.float32)
        self.arm.move_j(reset_q, wait=True, timeout=30.0)
        self.gripper.open()


# ─────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────────────────────

def make_real_env(
    *,
    reset_joint_positions: list[float] | None = None,
    image_resize_hw: tuple[int, int] | None = None,
    **kwargs,
) -> DobotUMIRealEnv:
    return DobotUMIRealEnv(
        reset_joint_positions=reset_joint_positions,
        image_resize_hw=image_resize_hw,
        **kwargs,
    )
