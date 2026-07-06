#!/usr/bin/env python3
"""RLT rollout entrypoint for dual Rokae AR arms with Zhixing grippers.

This is the robot-side process analogous to dobot_umi_ros.py, but without UMI
human override. It reuses the generic PikaChunkEnvAdapter/EnvDriver stack and
provides a dual-arm SDK bridge.
"""
# ruff: noqa
from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_OUTER = REPO_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC_ROOT, REPO_ROOT_OUTER, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from examples.rokae_zhixing_dual import constants as rokae_constants
from examples.rokae_zhixing_dual import robot_utils as rokae_utils

from manual_signal_bridge import (
    ENTER_CRITICAL_PHASE_SERVICE,
    RECORD_DONE_SERVICE,
    RECORD_FAILURE_SERVICE,
    RECORD_SUCCESS_SERVICE,
    REQUEST_NEXT_EPISODE_SERVICE,
    SET_CRITICAL_POLICY_ACTOR_SERVICE,
    SET_CRITICAL_POLICY_BASE_SERVICE,
    TOGGLE_CRITICAL_PHASE_SERVICE,
    ManualSignalBridge,
)
from pika_sync_ros import (
    ActorClient,
    EnvDriver,
    HumanInterventionState,
    MachineAFeatureClient,
    NullReplayClient,
    PhaseAwareActorClient,
    PikaChunkEnvAdapter,
    ReplayClient,
    RolloutPhaseController,
    RolloutRuntimeContext,
    StaticOnlinePhaseController,
    TaskState,
    _bind_runtime_hook,
    _default_done_fn,
    _default_reward_fn,
    _default_success_fn,
    _load_callable,
    _make_learner_status_reader,
    _override_system_urls,
    _resolve_min_online_actor_version,
)
from rlt_online_rl.config import load_system_config_yaml
from rlt_online_rl.runtime_logging import metrics_path_for, setup_process_logging


logger = logging.getLogger("rokae_zhixing_dual_ros")
DEFAULT_CONFIG = REPO_ROOT / "configs" / "tasks" / "rokae_zhixing_dual" / "online_rl.yaml"


class NullHumanActionRecorder:
    def snapshot_latest(self) -> tuple[np.ndarray | None, int]:
        return None, -1


class DualRokaeRobotBridge:
    # Public RLT/replay/action units follow the converted LeRobot dataset:
    # joints in degrees, grippers in millimeters. Hardware SDK calls stay rad/m.
    DEFAULT_ACTION_DELTA_LIMITS: tuple[float, ...] = (
        1.72, 1.72, 1.72, 1.72, 1.72, 1.72, 1.72, 10.0,
        1.72, 1.72, 1.72, 1.72, 1.72, 1.72, 1.72, 10.0,
    )

    def __init__(
        self,
        args: argparse.Namespace,
        arms: rokae_utils.DualRokaeArm,
        left_gripper: rokae_utils.ZhixingSDKGripper,
        right_gripper: rokae_utils.ZhixingSDKGripper,
        image_recorder: rokae_utils.MultiRealSenseImageRecorder,
    ) -> None:
        self._args = args
        self._arms = arms
        self._left_gripper = left_gripper
        self._right_gripper = right_gripper
        self._image_recorder = image_recorder
        self._lock = threading.Lock()
        self._last_sent_hw: np.ndarray | None = None

    @staticmethod
    def _joint_slices() -> tuple[slice, slice]:
        return (
            slice(0, rokae_constants.ARM_DOF),
            slice(rokae_constants.SINGLE_ARM_ACTION_DIM, rokae_constants.SINGLE_ARM_ACTION_DIM + rokae_constants.ARM_DOF),
        )

    @staticmethod
    def _gripper_indices() -> tuple[int, int]:
        return (
            rokae_constants.ARM_DOF,
            rokae_constants.SINGLE_ARM_ACTION_DIM + rokae_constants.ARM_DOF,
        )

    @classmethod
    def _hardware_to_policy_units(cls, values_hw: np.ndarray) -> np.ndarray:
        """Convert hardware units rad/m to dataset/RLT units degree/mm."""
        converted = np.asarray(values_hw, dtype=np.float32).copy()
        for slc in cls._joint_slices():
            converted[slc] = np.rad2deg(converted[slc])
        for idx in cls._gripper_indices():
            converted[idx] = converted[idx] * 1000.0
        return converted

    @classmethod
    def _policy_to_hardware_units(cls, values_policy: np.ndarray) -> np.ndarray:
        """Convert dataset/RLT units degree/mm to hardware units rad/m."""
        converted = np.asarray(values_policy, dtype=np.float32).copy()
        for slc in cls._joint_slices():
            converted[slc] = np.deg2rad(converted[slc])
        for idx in cls._gripper_indices():
            converted[idx] = converted[idx] * 0.001
        return converted

    @classmethod
    def _clip_joint_limits_hw(cls, values_hw: np.ndarray) -> np.ndarray:
        clipped = np.asarray(values_hw, dtype=np.float32).copy()
        limits_deg = np.asarray(rokae_constants.JOINT_LIMITS_DEG, dtype=np.float32)
        margin = float(rokae_constants.JOINT_LIMIT_MARGIN_DEG)
        lower = np.deg2rad(limits_deg[:, 0] + margin)
        upper = np.deg2rad(limits_deg[:, 1] - margin)
        for slc in cls._joint_slices():
            clipped[slc] = np.clip(clipped[slc], lower, upper)
        return clipped

    def _get_hardware_state(self) -> np.ndarray:
        left_q = self._arms.left.get_joint_angles_rad()
        right_q = self._arms.right.get_joint_angles_rad()
        left_g = float(np.clip(self._left_gripper.get_position_m(), 0.0, rokae_constants.GRIPPER_OPEN_M))
        right_g = float(np.clip(self._right_gripper.get_position_m(), 0.0, rokae_constants.GRIPPER_OPEN_M))
        return np.concatenate([left_q, [left_g], right_q, [right_g]], dtype=np.float32)

    def shutdown(self) -> None:
        try:
            self._arms.stop()
        except Exception:
            pass
        try:
            self._arms.disconnect()
        except Exception:
            pass
        for gripper in (self._left_gripper, self._right_gripper):
            try:
                gripper.release()
            except Exception:
                pass
        try:
            self._image_recorder.stop()
        except Exception:
            pass

    def set_policy_control_active(self, _enabled: bool) -> None:
        return

    def wait_for_observation_ready(self, timeout_s: float | None = None) -> None:
        self._image_recorder.wait_ready(timeout_s=timeout_s or 10.0)

    def get_observation(self, resize_hw: tuple[int, int], task: str) -> dict[str, Any]:
        images = self._image_recorder.get_images(resize_hw=resize_hw)
        state = self._hardware_to_policy_units(self._get_hardware_state())
        return {
            "state": state,
            "images": {
                key: images[key]
                for key in (
                    rokae_constants.IMAGE_KEY_FRONT,
                    rokae_constants.IMAGE_KEY_LEFT_WRIST,
                    rokae_constants.IMAGE_KEY_RIGHT_WRIST,
                    rokae_constants.IMAGE_KEY_TACTILE_LEFT,
                    rokae_constants.IMAGE_KEY_TACTILE_RIGHT,
                )
                if key in images
            },
            "prompt": task,
        }

    def send_action(self, action16: np.ndarray) -> np.ndarray:
        action_policy = np.asarray(action16, dtype=np.float32).reshape(-1)
        if action_policy.shape[0] < rokae_constants.DUAL_ACTION_DIM:
            raise ValueError(f"Expected 16D dual Rokae action, got {action_policy.shape}")
        action_policy = action_policy[: rokae_constants.DUAL_ACTION_DIM]
        if not np.all(np.isfinite(action_policy)):
            self._arms.stop()
            raise ValueError("Action contains NaN or Inf")
        with self._lock:
            desired_hw = self._policy_to_hardware_units(action_policy)
            limited_hw = self._limit_action(desired_hw)
            left_q = limited_hw[: rokae_constants.ARM_DOF]
            left_g = float(limited_hw[rokae_constants.ARM_DOF])
            right_base = rokae_constants.SINGLE_ARM_ACTION_DIM
            right_q = limited_hw[right_base : right_base + rokae_constants.ARM_DOF]
            right_g = float(limited_hw[right_base + rokae_constants.ARM_DOF])
            self._arms.servo_j(np.concatenate([left_q, right_q], dtype=np.float32))
            self._left_gripper.set_opening_m(left_g)
            self._right_gripper.set_opening_m(right_g)
            self._last_sent_hw = limited_hw.copy()
            return self._hardware_to_policy_units(limited_hw)

    def _limit_action(self, desired_hw: np.ndarray) -> np.ndarray:
        desired = self._clip_joint_limits_hw(desired_hw)
        left_g_idx = rokae_constants.ARM_DOF
        right_g_idx = rokae_constants.SINGLE_ARM_ACTION_DIM + rokae_constants.ARM_DOF
        desired[left_g_idx] = np.clip(desired[left_g_idx], rokae_constants.GRIPPER_CLOSE_M, rokae_constants.GRIPPER_OPEN_M)
        desired[right_g_idx] = np.clip(desired[right_g_idx], rokae_constants.GRIPPER_CLOSE_M, rokae_constants.GRIPPER_OPEN_M)
        reference = self._last_sent_hw
        if reference is None:
            reference = self._get_hardware_state()
        max_dq = float(self._args.policy_max_delta_rad)
        max_dg = float(self._args.policy_max_delta_gripper_m)
        limited = desired.copy()
        for slc in self._joint_slices():
            limited[slc] = reference[slc] + np.clip(desired[slc] - reference[slc], -max_dq, max_dq)
        for idx in self._gripper_indices():
            limited[idx] = np.clip(reference[idx] + np.clip(desired[idx] - reference[idx], -max_dg, max_dg),
                                   rokae_constants.GRIPPER_CLOSE_M, rokae_constants.GRIPPER_OPEN_M)
        return limited

    def reset_control_state(self) -> None:
        with self._lock:
            self._last_sent_hw = None


class DualRokaeEnvAdapter(PikaChunkEnvAdapter):
    def _sample_latest_human_action(self, observation) -> np.ndarray:
        # Human override is intentionally not implemented for dual Rokae yet.
        return np.asarray(observation["state"], dtype=np.float32)[: self._system.rl.action_dim]

    def _reset_robot_to_mode_start(self) -> None:
        target_raw = (
            self._system.env_driver.critical_phase_reset_action
            if self._task_mode == "critical_phase"
            else self._system.env_driver.full_task_reset_action
        )
        if target_raw is not None:
            target_policy = np.asarray(target_raw, dtype=np.float32).reshape(-1)
            if target_policy.shape[0] == rokae_constants.DUAL_ACTION_DIM:
                target_hw = self._robot._clip_joint_limits_hw(self._robot._policy_to_hardware_units(target_policy))
                left_q = target_hw[: rokae_constants.ARM_DOF]
                right_base = rokae_constants.SINGLE_ARM_ACTION_DIM
                right_q = target_hw[right_base : right_base + rokae_constants.ARM_DOF]
                self._robot._arms.move_j(
                    np.concatenate([left_q, right_q], dtype=np.float32),
                    wait=True,
                    timeout=60.0,
                    restore_realtime=True,
                )
                self._robot._left_gripper.set_opening_m(float(target_hw[rokae_constants.ARM_DOF]))
                self._robot._right_gripper.set_opening_m(float(target_hw[right_base + rokae_constants.ARM_DOF]))
            else:
                raise ValueError(f"Dual Rokae reset action must be 16D if configured, got {target_policy.shape[0]}")
        else:
            self._robot._arms.move_j_pose(
                np.asarray(rokae_constants.LEFT_RESET_END_POSE, dtype=np.float32),
                np.asarray(rokae_constants.RIGHT_RESET_END_POSE, dtype=np.float32),
                wait=True,
                timeout=60.0,
                restore_realtime=True,
            )
            self._robot._left_gripper.open()
            self._robot._right_gripper.open()
        self._robot.reset_control_state()
        self._last_sent_action = None
        time.sleep(0.3)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dual Rokae + Zhixing RLT robot rollout")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--task", type=str, default="dual arm manipulation")
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--max_chunk_steps_per_episode", type=int, default=200)
    parser.add_argument("--idle_sleep_sec", type=float, default=0.02)
    parser.add_argument("--machine_a_ws_url", type=str, default=None)
    parser.add_argument("--actor_service_url", type=str, default=None)
    parser.add_argument("--replay_service_url", type=str, default=None)
    parser.add_argument("--reward_factory", type=str, default=None)
    parser.add_argument("--success_factory", type=str, default=None)
    parser.add_argument("--done_factory", type=str, default=None)
    parser.add_argument("--safe_action_filter_factory", type=str, default=None)
    parser.add_argument("--action_delta_limits", type=float, nargs=16, default=DualRokaeRobotBridge.DEFAULT_ACTION_DELTA_LIMITS)

    parser.add_argument("--image_h", type=int, default=224)
    parser.add_argument("--image_w", type=int, default=224)
    parser.add_argument("--obs_ready_timeout_s", type=float, default=None)

    parser.add_argument("--rokae_sdk_python_dir", type=str, default=rokae_constants.ROKAE_SDK_PYTHON_DIR)
    parser.add_argument("--left_arm_remote_ip", type=str, default=rokae_constants.LEFT_ARM_REMOTE_IP)
    parser.add_argument("--left_arm_local_ip", type=str, default=rokae_constants.LEFT_ARM_LOCAL_IP)
    parser.add_argument("--right_arm_remote_ip", type=str, default=rokae_constants.RIGHT_ARM_REMOTE_IP)
    parser.add_argument("--right_arm_local_ip", type=str, default=rokae_constants.RIGHT_ARM_LOCAL_IP)

    parser.add_argument("--left_gripper_port", type=str, default=rokae_constants.LEFT_GRIPPER_SERIAL_PORT)
    parser.add_argument("--left_gripper_slave_id", type=int, default=rokae_constants.LEFT_GRIPPER_SLAVE_ID)
    parser.add_argument("--right_gripper_port", type=str, default=rokae_constants.RIGHT_GRIPPER_SERIAL_PORT)
    parser.add_argument("--right_gripper_slave_id", type=int, default=rokae_constants.RIGHT_GRIPPER_SLAVE_ID)
    parser.add_argument("--gripper_baudrate", type=int, default=rokae_constants.GRIPPER_BAUDRATE)
    parser.add_argument("--gripper_speed_pct", type=int, default=rokae_constants.GRIPPER_SPEED_PCT)
    parser.add_argument("--gripper_force_pct", type=int, default=rokae_constants.GRIPPER_FORCE_PCT)

    parser.add_argument("--realsense_front_serial", type=str, default=rokae_constants.REALSENSE_FRONT_SERIAL)
    parser.add_argument("--realsense_left_wrist_serial", type=str, default=rokae_constants.REALSENSE_LEFT_WRIST_SERIAL)
    parser.add_argument("--realsense_right_wrist_serial", type=str, default=rokae_constants.REALSENSE_RIGHT_WRIST_SERIAL)
    parser.add_argument("--realsense_width", type=int, default=rokae_constants.REALSENSE_WIDTH)
    parser.add_argument("--realsense_height", type=int, default=rokae_constants.REALSENSE_HEIGHT)
    parser.add_argument("--realsense_fps", type=int, default=rokae_constants.REALSENSE_FPS)

    parser.add_argument("--policy_max_delta_rad", type=float, default=rokae_constants.MAX_JOINT_DELTA_RAD)
    parser.add_argument("--policy_max_delta_gripper_m", type=float, default=rokae_constants.MAX_GRIPPER_DELTA_M)
    parser.add_argument("--require_online_approval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--step_trace_stride", type=int, default=None)
    parser.add_argument("--eval_actor_only", action="store_true")
    parser.add_argument("--ros_domain_id", type=int, default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    system = _override_system_urls(load_system_config_yaml(args.config), args)
    if system.rl.action_dim != rokae_constants.DUAL_ACTION_DIM:
        raise ValueError(f"Dual Rokae rollout requires action_dim=16, got {system.rl.action_dim}")

    configured_stride = int(system.env_driver.step_trace_stride)
    override_stride = configured_stride if args.step_trace_stride is None else max(int(args.step_trace_stride), 0)
    effective_step_trace_stride = 0 if args.eval_actor_only else override_stride
    effective_actor_deterministic = True if args.eval_actor_only else system.env_driver.actor_deterministic
    system = dataclasses.replace(
        system,
        env_driver=dataclasses.replace(
            system.env_driver,
            actor_deterministic=effective_actor_deterministic,
            step_trace_stride=effective_step_trace_stride,
            enable_human_override=False,
        ),
    )
    log_path = setup_process_logging("rokae_zhixing_dual_ros", system, console_level=logging.INFO)

    reward_fn = _load_callable(args.reward_factory) or _default_reward_fn
    success_fn = _load_callable(args.success_factory) or _default_success_fn
    done_fn = _load_callable(args.done_factory) or _default_done_fn
    safe_action_filter = _load_callable(args.safe_action_filter_factory)

    rclpy.init(domain_id=args.ros_domain_id) if args.ros_domain_id is not None else rclpy.init()
    task_state = TaskState(args.task)
    intervention_state = HumanInterventionState(policy_enabled=True)

    arms = rokae_utils.DualRokaeArm(
        left=rokae_utils.RokaeSDKArm(
            remote_ip=args.left_arm_remote_ip,
            local_ip=args.left_arm_local_ip,
            name="left",
            sdk_python_dir=args.rokae_sdk_python_dir,
        ),
        right=rokae_utils.RokaeSDKArm(
            remote_ip=args.right_arm_remote_ip,
            local_ip=args.right_arm_local_ip,
            name="right",
            sdk_python_dir=args.rokae_sdk_python_dir,
        ),
    )
    left_gripper = rokae_utils.ZhixingSDKGripper(
        port=args.left_gripper_port,
        slave_id=args.left_gripper_slave_id,
        baudrate=args.gripper_baudrate,
        speed_pct=args.gripper_speed_pct,
        force_pct=args.gripper_force_pct,
    )
    right_gripper = rokae_utils.ZhixingSDKGripper(
        port=args.right_gripper_port,
        slave_id=args.right_gripper_slave_id,
        baudrate=args.gripper_baudrate,
        speed_pct=args.gripper_speed_pct,
        force_pct=args.gripper_force_pct,
    )
    image_recorder = rokae_utils.MultiRealSenseImageRecorder(
        serial_front=args.realsense_front_serial,
        serial_left_wrist=args.realsense_left_wrist_serial,
        serial_right_wrist=args.realsense_right_wrist_serial,
        width=args.realsense_width,
        height=args.realsense_height,
        fps=args.realsense_fps,
    )

    image_recorder.start()
    arms.connect()
    arms.enable()
    if not left_gripper.init():
        raise RuntimeError("left gripper initialization failed")
    if not right_gripper.init():
        raise RuntimeError("right gripper initialization failed")

    robot = DualRokaeRobotBridge(args, arms, left_gripper, right_gripper, image_recorder)
    runtime_context = RolloutRuntimeContext(
        system=system,
        obs_node=None,  # type: ignore[arg-type]
        task_state=task_state,
        intervention_state=intervention_state,
        robot=robot,  # type: ignore[arg-type]
    )
    manual_signal_bridge = ManualSignalBridge()
    nodes = []
    nodes.extend(_bind_runtime_hook(manual_signal_bridge, runtime_context))
    nodes.extend(_bind_runtime_hook(reward_fn, runtime_context))
    nodes.extend(_bind_runtime_hook(success_fn, runtime_context))
    nodes.extend(_bind_runtime_hook(done_fn, runtime_context))
    nodes.extend(_bind_runtime_hook(safe_action_filter, runtime_context))

    executor = MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    feature_provider = MachineAFeatureClient(
        system.env_driver.machine_a_ws_url,
        connect_timeout_sec=system.env_driver.machine_a_connect_timeout_sec,
        recv_timeout_sec=system.env_driver.machine_a_recv_timeout_sec,
        retry_interval_sec=system.env_driver.machine_a_retry_interval_sec,
    )
    replay_client = (
        NullReplayClient()
        if args.eval_actor_only
        else ReplayClient(system.env_driver.replay_service_url, timeout_sec=system.env_driver.replay_request_timeout_sec)
    )
    min_online_actor_version = 0 if args.eval_actor_only else _resolve_min_online_actor_version(system)
    learner_status_path = metrics_path_for(system, "learner_status.json")
    phase_controller = (
        StaticOnlinePhaseController()
        if args.eval_actor_only
        else RolloutPhaseController(
            replay_client,
            system.rl.warmup_min_size,
            min_online_actor_version=min_online_actor_version,
            require_online_approval=args.require_online_approval,
            logger_=logger,
        )
    )
    base_actor_client = ActorClient(system.env_driver.actor_service_url, timeout_sec=system.env_driver.actor_request_timeout_sec)
    phase_controller.bind_actor_version_getter(base_actor_client.get_actor_param_version)
    phase_controller.bind_learner_status_getter(_make_learner_status_reader(learner_status_path))
    phase_controller.bind_online_approval_getter(runtime_context.has_online_approval)
    phase_controller.bind_online_approval_consumer(runtime_context.consume_online_approval)
    phase_controller.bind_next_episode_request_getter(runtime_context.has_pending_next_episode_request)
    actor_client = PhaseAwareActorClient(base_actor_client, phase_controller, runtime_context)

    env = DualRokaeEnvAdapter(
        system=system,
        robot=robot,  # type: ignore[arg-type]
        task_state=task_state,
        intervention_state=intervention_state,
        human_action_recorder=NullHumanActionRecorder(),  # type: ignore[arg-type]
        phase_controller=phase_controller,  # type: ignore[arg-type]
        runtime_context=runtime_context,
        reward_fn=reward_fn,
        success_fn=success_fn,
        done_fn=done_fn,
        safe_action_filter=safe_action_filter,
        max_chunk_steps_per_episode=args.max_chunk_steps_per_episode,
        idle_sleep_sec=args.idle_sleep_sec,
        action_delta_limits=args.action_delta_limits,
        resize_hw=(args.image_h, args.image_w),
        obs_ready_timeout_s=args.obs_ready_timeout_s,
    )
    driver = EnvDriver(
        env=env,
        feature_provider=feature_provider,
        actor_client=actor_client,
        replay_client=replay_client,
        rl_config=system.rl,
        env_config=system.env_driver,
        eval_actor_only=args.eval_actor_only,
        metrics_path=str(metrics_path_for(system, "robot_rollout_metrics.jsonl")),
    )

    logger.info("Starting dual Rokae rollout log=%s config=%s", log_path, args.config)
    logger.info("Manual services next=%s success=%s failure=%s done=%s critical=%s toggle=%s actor=%s base=%s",
                REQUEST_NEXT_EPISODE_SERVICE, RECORD_SUCCESS_SERVICE, RECORD_FAILURE_SERVICE,
                RECORD_DONE_SERVICE, ENTER_CRITICAL_PHASE_SERVICE, TOGGLE_CRITICAL_PHASE_SERVICE,
                SET_CRITICAL_POLICY_ACTOR_SERVICE, SET_CRITICAL_POLICY_BASE_SERVICE)
    try:
        driver.run_forever(num_episodes=args.num_episodes)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, shutting down")
    finally:
        robot.shutdown()
        executor.shutdown()
        for node in nodes:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
