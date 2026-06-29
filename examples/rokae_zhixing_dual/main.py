"""Inference entrypoint for dual Rokae AR arms with Zhixing grippers."""
from __future__ import annotations

import dataclasses
import logging

import numpy as np
import tyro
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent

from examples.rokae_zhixing_dual import constants
from examples.rokae_zhixing_dual import env as _env
from examples.rokae_zhixing_dual import robot_utils as _utils


def _zero_tactile_images() -> dict[str, np.ndarray]:
    return {
        constants.IMAGE_KEY_TACTILE_LEFT: np.zeros((240, 240, 3), dtype=np.uint8),
        constants.IMAGE_KEY_TACTILE_RIGHT: np.zeros((240, 240, 3), dtype=np.uint8),
    }


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    action_horizon: int = 10
    num_episodes: int = 1
    max_episode_steps: int = 500
    render_height: int = 224
    render_width: int = 224

    rokae_sdk_python_dir: str = constants.ROKAE_SDK_PYTHON_DIR
    left_arm_remote_ip: str = constants.LEFT_ARM_REMOTE_IP
    left_arm_local_ip: str = constants.LEFT_ARM_LOCAL_IP
    right_arm_remote_ip: str = constants.RIGHT_ARM_REMOTE_IP
    right_arm_local_ip: str = constants.RIGHT_ARM_LOCAL_IP

    left_gripper_port: str = constants.LEFT_GRIPPER_SERIAL_PORT
    left_gripper_slave_id: int = constants.LEFT_GRIPPER_SLAVE_ID
    right_gripper_port: str = constants.RIGHT_GRIPPER_SERIAL_PORT
    right_gripper_slave_id: int = constants.RIGHT_GRIPPER_SLAVE_ID

    cam_front_serial: str = constants.REALSENSE_FRONT_SERIAL
    cam_left_wrist_serial: str = constants.REALSENSE_LEFT_WRIST_SERIAL
    cam_right_wrist_serial: str = constants.REALSENSE_RIGHT_WRIST_SERIAL

    tactile_sn_left: str = constants.TACTILE_LEFT_SERIAL
    tactile_sn_right: str = constants.TACTILE_RIGHT_SERIAL
    allow_missing_tactile: bool = False
    allow_zero_reset_pose: bool = constants.ALLOW_ZERO_RESET_POSE
    max_joint_delta_rad: float = constants.MAX_JOINT_DELTA_RAD
    max_gripper_delta_m: float = constants.MAX_GRIPPER_DELTA_M


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = ws_client_policy.get_server_metadata()
    logging.info("policy server metadata: %s", metadata)

    tactile_camera = None
    tactile_image_provider = None
    if metadata.get("use_tactile", False):
        tactile_camera = _utils.VitAITactileCamera(
            serial_left=args.tactile_sn_left,
            serial_right=args.tactile_sn_right,
        )
        if tactile_camera.connect():
            tactile_image_provider = tactile_camera.get_images
        elif args.allow_missing_tactile:
            logging.warning("tactile connection failed; using zero tactile images")
            tactile_camera = None
            tactile_image_provider = _zero_tactile_images
        else:
            raise RuntimeError("policy requires tactile input, but tactile cameras failed to connect")

    environment = _env.DualRokaeZhixingEnvironment(
        reset_joint_positions=metadata.get("reset_pose"),
        render_height=args.render_height,
        render_width=args.render_width,
        rokae_sdk_python_dir=args.rokae_sdk_python_dir,
        left_arm_remote_ip=args.left_arm_remote_ip,
        left_arm_local_ip=args.left_arm_local_ip,
        right_arm_remote_ip=args.right_arm_remote_ip,
        right_arm_local_ip=args.right_arm_local_ip,
        left_gripper_port=args.left_gripper_port,
        left_gripper_slave_id=args.left_gripper_slave_id,
        right_gripper_port=args.right_gripper_port,
        right_gripper_slave_id=args.right_gripper_slave_id,
        cam_front_serial=args.cam_front_serial,
        cam_left_wrist_serial=args.cam_left_wrist_serial,
        cam_right_wrist_serial=args.cam_right_wrist_serial,
        allow_zero_reset_pose=args.allow_zero_reset_pose,
        max_joint_delta_rad=args.max_joint_delta_rad,
        max_gripper_delta_m=args.max_gripper_delta_m,
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
        max_hz=int(round(1.0 / constants.DT)),
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    try:
        runtime.run()
    finally:
        environment.close()
        if tactile_camera is not None:
            tactile_camera.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
