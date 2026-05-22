"""env.py — openpi_client.runtime.Environment 适配器（越疆 Dobot + UMI）。

将 DobotUMIRealEnv 包装为 openpi Runtime 所需的 Environment 接口，
处理图像尺寸归一化和通道顺序（HWC → CHW）。
"""
from __future__ import annotations

from typing import List, Optional

import einops
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.dobot_umi import constants
from examples.dobot_umi import real_env as _real_env


class DobotUMIEnvironment(_environment.Environment):
    """适配越疆 Dobot CR 机械臂 + 知行夹爪 + 双 RealSense 的 openpi 运行时环境。

    观测格式（送往 Pi0 策略服务器）::

        {
            "state":  np.ndarray (7,)  — [joint1..6 (rad), gripper_m]
            "images": {
                "cam_front": np.ndarray (3, H, W) uint8,
                "cam_wrist": np.ndarray (3, H, W) uint8,
            }
        }

    动作格式（由 Pi0 策略服务器返回）::

        {
            "actions": np.ndarray (7,)  — [joint1..6 (rad), gripper_m]
        }
    """

    def __init__(
        self,
        *,
        reset_joint_positions: Optional[List[float]] = None,
        render_height: int = 224,
        render_width: int = 224,
        cam_front_topic: str = constants.CAM_FRONT_TOPIC,
        cam_wrist_topic: str = constants.CAM_WRIST_TOPIC,
        joint_topic: str = constants.JOINT_STATE_TOPIC,
        joint_cmd_topic: str = constants.JOINT_CMD_TOPIC,
        gripper_ctrl_topic: str = constants.GRIPPER_CTRL_TOPIC,
        enable_gripper_stream: bool = True,
        obs_ready_timeout_s: float | None = 10.0,
    ) -> None:
        self._env = _real_env.make_real_env(
            init_node=True,
            reset_joint_positions=reset_joint_positions,
            image_resize_hw=(render_height, render_width),
            cam_front_topic=cam_front_topic,
            cam_wrist_topic=cam_wrist_topic,
            joint_topic=joint_topic,
            joint_cmd_topic=joint_cmd_topic,
            gripper_ctrl_topic=gripper_ctrl_topic,
            enable_gripper_stream=enable_gripper_stream,
            obs_ready_timeout_s=obs_ready_timeout_s,
        )
        self._render_height = render_height
        self._render_width = render_width
        self._ts = None

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("尚未调用 reset()，TimeStep 为空")

        obs = self._ts.observation
        images_out = {}
        for cam_name, img in obs["images"].items():
            # img 已经是 (H, W, 3) uint8 —— resize 再转 CHW
            img_resized = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, self._render_height, self._render_width)
            )
            images_out[cam_name] = einops.rearrange(img_resized, "h w c -> c h w")

        return {
            "state": obs["qpos"].astype(np.float32),
            "images": images_out,
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action["actions"])
