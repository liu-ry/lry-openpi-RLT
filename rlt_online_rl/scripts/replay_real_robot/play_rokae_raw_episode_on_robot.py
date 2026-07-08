#!/usr/bin/env python3
"""Replay a raw dual-Rokae episode on the real robot via the rollout bridge.

The command path intentionally matches rollout:
raw episode frame -> 16D policy action [deg, gripper_raw/100] ->
DualRokaeRobotBridge.send_action() -> rollout unit conversion, safety clamps,
DualRokaeArm.servo_j(), ZhixingSDKGripper.set_raw_position().
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RLT_ROOT = REPO_ROOT / "rlt_online_rl"
SRC_ROOT = RLT_ROOT / "src"
SCRIPT_DIR = RLT_ROOT / "train_deploy_alignment"
for path in (REPO_ROOT, SRC_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from examples.rokae_zhixing_dual import constants as rokae_constants
from examples.rokae_zhixing_dual import robot_utils as rokae_utils


class NullImageRecorder:
    def wait_ready(self, timeout_s: float | None = None) -> None:
        return

    def stop(self) -> None:
        return


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Play a raw /home/lry/temp/rokae_dual episode on the real dual Rokae setup. "
            "This uses the same DualRokaeRobotBridge.send_action path as rollout."
        )
    )
    parser.add_argument(
        "--episode-dir",
        type=Path,
        default=Path("/home/lry/temp/rokae_dual/episode_100"),
        help="Raw episode directory containing robot_rokae_dual_*_joint.npy and gripper_*_width.npy.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually connect to hardware and replay. Default is dry-run.")
    parser.add_argument("--start-frame", type=int, default=0, help="First raw frame to play.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum number of frames to play.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride before playback.")
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=None,
        help="Playback rate. Defaults to meta devices.total.actual_fps, then timestamp median FPS, then 30 Hz.",
    )
    parser.add_argument(
        "--resample-hz",
        type=float,
        default=None,
        help="Linearly resample the raw trajectory to this fixed command rate before playback, e.g. 50.",
    )
    parser.add_argument("--servo-dt", type=float, default=rokae_constants.ROKAE_SERVO_DT)
    parser.add_argument("--servo-lookahead", type=float, default=rokae_constants.ROKAE_SERVO_LOOKAHEAD)
    parser.add_argument("--servo-kp", type=float, default=rokae_constants.ROKAE_SERVO_KP)
    parser.add_argument(
        "--skip-gripper-commands",
        action="store_true",
        help="Diagnostic mode: keep rollout arm servo path but bypass per-frame gripper serial writes.",
    )
    parser.add_argument(
        "--use-timestamps",
        action="store_true",
        help="Replay using timestamp deltas instead of a fixed --rate-hz.",
    )
    parser.add_argument("--initial-wait-sec", type=float, default=1.0, help="Wait before first replay frame.")
    parser.add_argument(
        "--startup-move",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move both arms to the first selected frame with non-realtime moveJ before streaming.",
    )
    parser.add_argument(
        "--startup-move-speed",
        type=float,
        default=rokae_constants.ROKAE_RESET_MOVEJ_SPEED,
        help="Rokae moveJ speed used for startup move.",
    )
    parser.add_argument("--post-startup-hold-sec", type=float, default=1.0)
    parser.add_argument("--joint-units", choices=("auto", "deg", "rad"), default="auto")
    parser.add_argument(
        "--gripper-units",
        choices=("policy_raw_div100", "raw"),
        default="policy_raw_div100",
        help="Raw episode gripper unit. Current RokaeDual data is SDK raw position divided by 100.",
    )
    parser.add_argument("--left-arm-remote-ip", type=str, default=rokae_constants.LEFT_ARM_REMOTE_IP)
    parser.add_argument("--left-arm-local-ip", type=str, default=rokae_constants.LEFT_ARM_LOCAL_IP)
    parser.add_argument("--right-arm-remote-ip", type=str, default=rokae_constants.RIGHT_ARM_REMOTE_IP)
    parser.add_argument("--right-arm-local-ip", type=str, default=rokae_constants.RIGHT_ARM_LOCAL_IP)
    parser.add_argument("--rokae-sdk-python-dir", type=str, default=rokae_constants.ROKAE_SDK_PYTHON_DIR)
    parser.add_argument("--left-gripper-port", type=str, default=rokae_constants.LEFT_GRIPPER_SERIAL_PORT)
    parser.add_argument("--left-gripper-slave-id", type=int, default=rokae_constants.LEFT_GRIPPER_SLAVE_ID)
    parser.add_argument("--right-gripper-port", type=str, default=rokae_constants.RIGHT_GRIPPER_SERIAL_PORT)
    parser.add_argument("--right-gripper-slave-id", type=int, default=rokae_constants.RIGHT_GRIPPER_SLAVE_ID)
    parser.add_argument("--gripper-baudrate", type=int, default=rokae_constants.GRIPPER_BAUDRATE)
    parser.add_argument("--gripper-speed-pct", type=int, default=rokae_constants.GRIPPER_SPEED_PCT)
    parser.add_argument("--gripper-force-pct", type=int, default=rokae_constants.GRIPPER_FORCE_PCT)
    parser.add_argument("--policy-max-delta-rad", type=float, default=rokae_constants.MAX_JOINT_DELTA_RAD)
    parser.add_argument("--policy-max-delta-gripper-m", type=float, default=rokae_constants.MAX_GRIPPER_DELTA_M)
    parser.add_argument("--teleop-max-delta-rad", type=float, default=0.05)
    parser.add_argument(
        "--teleop-max-delta-gripper-m",
        type=float,
        default=rokae_constants.GRIPPER_POS_CLOSE / rokae_constants.ROKAE_POLICY_GRIPPER_RAW_SCALE,
    )
    parser.add_argument(
        "--confirm-each-action",
        action="store_true",
        help="Forwarded to DualRokaeRobotBridge; prints each action and waits for Enter before sending.",
    )
    return parser.parse_args()


def _load_episode(episode_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    required = {
        "left_joint": episode_dir / "robot_rokae_dual_left_joint.npy",
        "right_joint": episode_dir / "robot_rokae_dual_right_joint.npy",
        "left_gripper": episode_dir / "gripper_left_width.npy",
        "right_gripper": episode_dir / "gripper_right_width.npy",
        "timestamps": episode_dir / "timestamps.npy",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw episode files: {missing}")

    left_joint = np.load(required["left_joint"]).astype(np.float32)
    right_joint = np.load(required["right_joint"]).astype(np.float32)
    left_gripper = np.load(required["left_gripper"]).astype(np.float32).reshape(-1, 1)
    right_gripper = np.load(required["right_gripper"]).astype(np.float32).reshape(-1, 1)
    timestamps = np.load(required["timestamps"]).astype(np.float64).reshape(-1)
    lengths = {
        "left_joint": len(left_joint),
        "right_joint": len(right_joint),
        "left_gripper": len(left_gripper),
        "right_gripper": len(right_gripper),
        "timestamps": len(timestamps),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Episode arrays have mismatched lengths: {lengths}")
    if left_joint.ndim != 2 or left_joint.shape[1] < rokae_constants.ARM_DOF:
        raise ValueError(f"left_joint must be [T, >=7], got {left_joint.shape}")
    if right_joint.ndim != 2 or right_joint.shape[1] < rokae_constants.ARM_DOF:
        raise ValueError(f"right_joint must be [T, >=7], got {right_joint.shape}")

    meta_path = episode_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    actions = np.concatenate(
        [
            left_joint[:, : rokae_constants.ARM_DOF],
            left_gripper,
            right_joint[:, : rokae_constants.ARM_DOF],
            right_gripper,
        ],
        axis=1,
    ).astype(np.float32)
    return actions, timestamps, meta


def _convert_units(actions: np.ndarray, *, joint_units: str, gripper_units: str) -> np.ndarray:
    converted = np.asarray(actions, dtype=np.float32).copy()
    joint_indices = [*range(0, rokae_constants.ARM_DOF), *range(8, 8 + rokae_constants.ARM_DOF)]
    if joint_units == "auto":
        max_abs_joint = float(np.nanmax(np.abs(converted[:, joint_indices])))
        joint_units = "rad" if max_abs_joint <= 2.0 * np.pi + 1e-3 else "deg"
    if joint_units == "rad":
        converted[:, joint_indices] = np.rad2deg(converted[:, joint_indices])
    elif joint_units != "deg":
        raise ValueError(f"Unsupported joint_units={joint_units!r}")

    if gripper_units == "raw":
        converted[:, [7, 15]] /= float(rokae_constants.ROKAE_POLICY_GRIPPER_RAW_SCALE)
    elif gripper_units != "policy_raw_div100":
        raise ValueError(f"Unsupported gripper_units={gripper_units!r}")
    return converted


def _select_frames(actions: np.ndarray, timestamps: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.start_frame < 0 or args.start_frame >= len(actions):
        raise ValueError(f"--start-frame={args.start_frame} out of range for {len(actions)} frames")
    indices = np.arange(args.start_frame, len(actions), args.stride, dtype=np.int64)
    if args.max_frames is not None:
        indices = indices[: max(int(args.max_frames), 0)]
    if indices.size == 0:
        raise ValueError("No frames selected for playback")
    return actions[indices], timestamps[indices]


def _resample_actions(actions: np.ndarray, timestamps: np.ndarray, rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    if rate_hz <= 0:
        raise ValueError("--resample-hz must be > 0")
    if len(actions) <= 1:
        return actions, timestamps
    t = np.asarray(timestamps, dtype=np.float64)
    t = t - t[0]
    if not np.all(np.diff(t) > 0):
        t = np.arange(len(actions), dtype=np.float64) / rate_hz
    duration = float(t[-1])
    period = 1.0 / float(rate_hz)
    new_t = np.arange(0.0, duration + period * 0.5, period, dtype=np.float64)
    new_actions = np.empty((len(new_t), actions.shape[1]), dtype=np.float32)
    for dim in range(actions.shape[1]):
        new_actions[:, dim] = np.interp(new_t, t, actions[:, dim]).astype(np.float32)
    return new_actions, timestamps[0] + new_t


def _resolve_rate_hz(timestamps: np.ndarray, meta: dict[str, Any], args: argparse.Namespace) -> float:
    if args.rate_hz is not None:
        return float(args.rate_hz)
    try:
        rate = float(meta["devices"]["total"]["actual_fps"])
        if rate > 0:
            return rate
    except Exception:
        pass
    if len(timestamps) > 1:
        dt = np.diff(timestamps)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            return float(1.0 / np.median(dt))
    return 30.0


def _print_summary(actions: np.ndarray, timestamps: np.ndarray, meta: dict[str, Any], args: argparse.Namespace) -> None:
    rate_hz = _resolve_rate_hz(timestamps, meta, args)
    diff = np.diff(actions, axis=0) if len(actions) > 1 else np.zeros((0, actions.shape[1]), dtype=np.float32)
    joint_indices = np.asarray([*range(0, 7), *range(8, 15)], dtype=np.int64)
    print(f"Episode dir: {args.episode_dir}")
    print(f"Episode id: {meta.get('episode_id', 'unknown')}")
    print(f"Selected frames: {len(actions)}")
    print(f"Playback timing: {'timestamps' if args.use_timestamps else f'fixed {rate_hz:.3f} Hz'}")
    print(f"ServoJ params: dt={args.servo_dt:.4f} lookahead={args.servo_lookahead:.4f} kp={args.servo_kp:.3f}")
    print(f"Execute: {args.execute}")
    print(f"Startup move: {args.startup_move} speed={args.startup_move_speed}")
    print(f"Action units sent to rollout bridge: joints=deg gripper=raw/100")
    print(f"First action: {np.round(actions[0], 3).tolist()}")
    print(f"Last action:  {np.round(actions[-1], 3).tolist()}")
    if diff.size:
        print(f"Joint step delta mean_abs_deg={float(np.mean(np.abs(diff[:, joint_indices]))):.4f}")
        print(f"Joint step delta p95_abs_deg={float(np.percentile(np.abs(diff[:, joint_indices]), 95)):.4f}")
        print(f"Joint step delta max_abs_deg={float(np.max(np.abs(diff[:, joint_indices]))):.4f}")
    print("Hardware path: DualRokaeRobotBridge.send_action(), same as rollout policy execution.")


def _build_robot(args: argparse.Namespace):
    from rokae_zhixing_dual_ros import DualRokaeRobotBridge

    rokae_utils.preload_pyrokae(args.rokae_sdk_python_dir)
    arms = rokae_utils.DualRokaeArm(
        left=rokae_utils.RokaeSDKArm(
            remote_ip=args.left_arm_remote_ip,
            local_ip=args.left_arm_local_ip,
            name="left",
            sdk_python_dir=args.rokae_sdk_python_dir,
            servo_dt=args.servo_dt,
            servo_lookahead=args.servo_lookahead,
            servo_kp=args.servo_kp,
        ),
        right=rokae_utils.RokaeSDKArm(
            remote_ip=args.right_arm_remote_ip,
            local_ip=args.right_arm_local_ip,
            name="right",
            sdk_python_dir=args.rokae_sdk_python_dir,
            servo_dt=args.servo_dt,
            servo_lookahead=args.servo_lookahead,
            servo_kp=args.servo_kp,
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
    bridge_args = argparse.Namespace(**vars(args))
    robot = DualRokaeRobotBridge(
        bridge_args,
        arms,
        left_gripper,
        right_gripper,
        NullImageRecorder(),
    )
    return robot


def _startup_move(robot, first_action: np.ndarray, args: argparse.Namespace) -> None:
    from rokae_zhixing_dual_ros import DualRokaeRobotBridge

    target_hw = robot._clip_joint_limits_hw(DualRokaeRobotBridge._policy_to_hardware_units(first_action))
    left_q = target_hw[: rokae_constants.ARM_DOF]
    right_base = rokae_constants.SINGLE_ARM_ACTION_DIM
    right_q = target_hw[right_base : right_base + rokae_constants.ARM_DOF]
    robot._arms.move_j(
        np.concatenate([left_q, right_q], dtype=np.float32),
        wait=True,
        timeout=60.0,
        speed=float(args.startup_move_speed),
        restore_realtime=False,
    )
    scale = float(rokae_constants.ROKAE_POLICY_GRIPPER_RAW_SCALE)
    robot._left_gripper.set_raw_position(float(target_hw[rokae_constants.ARM_DOF]) * scale)
    robot._right_gripper.set_raw_position(float(target_hw[right_base + rokae_constants.ARM_DOF]) * scale)
    robot.reset_control_state()


def _sleep_until_next_frame(
    *,
    frame_idx: int,
    send_started_at: float,
    timestamps: np.ndarray,
    fixed_period_s: float,
    use_timestamps: bool,
) -> None:
    if frame_idx <= 0:
        return
    if use_timestamps:
        target_elapsed = float(timestamps[frame_idx] - timestamps[0])
    else:
        target_elapsed = frame_idx * fixed_period_s
    remaining = send_started_at + target_elapsed - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def _execute(actions: np.ndarray, timestamps: np.ndarray, meta: dict[str, Any], args: argparse.Namespace) -> None:
    robot = _build_robot(args)
    try:
        print("[play_rokae_raw_episode] connecting arms ...", flush=True)
        robot._arms.connect()
        robot._arms.enable()
        print("[play_rokae_raw_episode] initializing grippers ...", flush=True)
        if not robot._left_gripper.init():
            raise RuntimeError("left gripper initialization failed")
        if not robot._right_gripper.init():
            raise RuntimeError("right gripper initialization failed")
        if args.startup_move:
            print("[play_rokae_raw_episode] startup move to first frame ...", flush=True)
            _startup_move(robot, actions[0], args)
            if args.post_startup_hold_sec > 0:
                time.sleep(float(args.post_startup_hold_sec))
        if args.initial_wait_sec > 0:
            print(f"[play_rokae_raw_episode] initial wait {args.initial_wait_sec:.2f}s ...", flush=True)
            time.sleep(float(args.initial_wait_sec))

        rate_hz = _resolve_rate_hz(timestamps, meta, args)
        fixed_period_s = 1.0 / max(rate_hz, 1e-6)
        print(f"[play_rokae_raw_episode] replay start frames={len(actions)} rate_hz={rate_hz:.3f}", flush=True)
        if args.skip_gripper_commands:
            print("[play_rokae_raw_episode] diagnostic: per-frame gripper commands are disabled", flush=True)
            robot._left_gripper.set_raw_position = lambda _raw_position: None
            robot._right_gripper.set_raw_position = lambda _raw_position: None
        send_durations: list[float] = []
        intervals: list[float] = []
        overruns = 0
        max_overrun = 0.0
        last_send_start: float | None = None
        send_started_at = time.perf_counter()
        for idx, action in enumerate(actions):
            _sleep_until_next_frame(
                frame_idx=idx,
                send_started_at=send_started_at,
                timestamps=timestamps,
                fixed_period_s=fixed_period_s,
                use_timestamps=bool(args.use_timestamps),
            )
            send_start = time.perf_counter()
            if last_send_start is not None:
                intervals.append(send_start - last_send_start)
            last_send_start = send_start
            sent = robot.send_action(action, _source="policy")
            send_duration = time.perf_counter() - send_start
            send_durations.append(send_duration)
            overrun = send_duration - fixed_period_s
            if overrun > 0:
                overruns += 1
                max_overrun = max(max_overrun, overrun)
            if idx == 0 or (idx + 1) % 50 == 0 or idx == len(actions) - 1:
                print(
                    f"[play_rokae_raw_episode] sent frame {idx + 1}/{len(actions)} "
                    f"send_ms={send_duration * 1000.0:.2f} action={np.round(sent, 3).tolist()}",
                    flush=True,
                )
        if send_durations:
            send_ms = np.asarray(send_durations, dtype=np.float64) * 1000.0
            print(
                "[play_rokae_raw_episode] timing send_action_ms "
                f"mean={send_ms.mean():.2f} p95={np.percentile(send_ms, 95):.2f} max={send_ms.max():.2f} "
                f"overruns={overruns} max_overrun_ms={max_overrun * 1000.0:.2f}",
                flush=True,
            )
        if intervals:
            interval_ms = np.asarray(intervals, dtype=np.float64) * 1000.0
            print(
                "[play_rokae_raw_episode] timing send_interval_ms "
                f"target={fixed_period_s * 1000.0:.2f} mean={interval_ms.mean():.2f} "
                f"p95={np.percentile(interval_ms, 95):.2f} max={interval_ms.max():.2f}",
                flush=True,
            )
        print("[play_rokae_raw_episode] replay done", flush=True)
    finally:
        robot.shutdown()


def main() -> None:
    args = _parse_args()
    actions_raw, timestamps, meta = _load_episode(args.episode_dir)
    actions_policy = _convert_units(actions_raw, joint_units=args.joint_units, gripper_units=args.gripper_units)
    actions, selected_timestamps = _select_frames(actions_policy, timestamps, args)
    if args.resample_hz is not None:
        actions, selected_timestamps = _resample_actions(actions, selected_timestamps, args.resample_hz)
        args.rate_hz = float(args.resample_hz)
        args.use_timestamps = False
    _print_summary(actions, selected_timestamps, meta, args)
    if not args.execute:
        print("Dry-run only. Pass --execute to connect to hardware and replay.")
        return
    print("About to connect to real hardware and replay through rollout bridge. Press Enter to continue.", flush=True)
    input()
    _execute(actions, selected_timestamps, meta, args)


if __name__ == "__main__":
    main()
