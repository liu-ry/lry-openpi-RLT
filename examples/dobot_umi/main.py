"""main.py — 越疆 Dobot + UMI 推理入口。

连接 openpi 策略服务器，在真实机械臂上执行推理 rollout。

用法示例::

    python main.py \
        --host 192.168.3.5 \
        --port 8000 \
        --task "pick and place the red block" \
        --num_episodes 5
"""
from __future__ import annotations

import dataclasses
import logging

import tyro
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent

from examples.dobot_umi import constants
from examples.dobot_umi import env as _env
from examples.dobot_umi import robot_utils as _utils


def _zero_tactile_images() -> dict[str, object]:
    import numpy as np

    return {
        constants.IMAGE_KEY_TACTILE_LEFT: np.zeros((240, 240, 3), dtype=np.uint8),
        constants.IMAGE_KEY_TACTILE_RIGHT: np.zeros((240, 240, 3), dtype=np.uint8),
    }


@dataclasses.dataclass
class Args:
    # ── 策略服务器 ────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── 运行参数 ─────────────────────────────────────────────────
    action_horizon: int = 10
    num_episodes: int = 1
    max_episode_steps: int = 500

    # ── 图像分辨率 ────────────────────────────────────────────────
    render_height: int = 224
    render_width: int = 224

    # ── 任务描述 ─────────────────────────────────────────────────
    task: str = "pick and place the object"

    # ── ROS 话题（可按需覆盖）────────────────────────────────────
    cam_front_topic: str = constants.CAM_FRONT_TOPIC
    cam_wrist_topic: str = constants.CAM_WRIST_TOPIC
    joint_topic: str = constants.JOINT_STATE_TOPIC
    joint_cmd_topic: str = constants.JOINT_CMD_TOPIC
    gripper_ctrl_topic: str = constants.GRIPPER_CTRL_TOPIC
    enable_gripper_stream: bool = True
    tactile_sn_left: str = constants.TACTILE_LEFT_SERIAL
    tactile_sn_right: str = constants.TACTILE_RIGHT_SERIAL
    allow_missing_tactile: bool = False


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info("策略服务器元数据: %s", ws_client_policy.get_server_metadata())
    metadata = ws_client_policy.get_server_metadata()
    tactile_camera = None
    tactile_image_provider = None
    if metadata.get("use_tactile", False):
        if not args.tactile_sn_left or not args.tactile_sn_right:
            raise ValueError("策略需要触觉模态，但 tactile_sn_left/right 未配置")
        tactile_camera = _utils.VitAITactileCamera(
            serial_left=args.tactile_sn_left,
            serial_right=args.tactile_sn_right,
        )
        if tactile_camera.connect():
            tactile_image_provider = tactile_camera.get_images
        elif args.allow_missing_tactile:
            logging.warning("触觉传感器连接失败，因 allow_missing_tactile 继续以全零触觉图像运行")
            tactile_camera = None
            tactile_image_provider = _zero_tactile_images
        else:
            raise RuntimeError("策略需要触觉模态，但触觉传感器连接失败")

    environment = _env.DobotUMIEnvironment(
        reset_joint_positions=metadata.get("reset_pose"),
        render_height=args.render_height,
        render_width=args.render_width,
        cam_front_topic=args.cam_front_topic,
        cam_wrist_topic=args.cam_wrist_topic,
        joint_topic=args.joint_topic,
        joint_cmd_topic=args.joint_cmd_topic,
        gripper_ctrl_topic=args.gripper_ctrl_topic,
        enable_gripper_stream=args.enable_gripper_stream,
        enable_tactile=tactile_image_provider is not None,
        tactile_image_provider=tactile_image_provider,
    )

    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=50,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    try:
        runtime.run()
    finally:
        if tactile_camera is not None:
            tactile_camera.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
