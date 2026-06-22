#!/usr/bin/env python3
"""Play Dobot robot_joint.npy trajectories through the Dobot SDK ServoJ path.

Usage:
  # 1) Inspect a recorded episode without connecting to hardware.
  python rlt_online_rl/scripts/replay_real_robot/play_dobot_joint_npy.py \
    --episode-dir /home/lry/temp/Dobot/episode_28 \
    --dry-run

  # 2) Replay a near-static episode at 20 Hz, arm only.
  python rlt_online_rl/scripts/replay_real_robot/play_dobot_joint_npy.py \
    --episode-dir /home/lry/temp/Dobot/episode_28 \
    --hz 20 \
    --disable-gripper

  # 3) Replay a moving episode at 20 Hz, with gripper if gripper.npy exists.
  python rlt_online_rl/scripts/replay_real_robot/play_dobot_joint_npy.py \
    --episode-dir /home/lry/temp/Dobot/episode_0 \
    --hz 20

Notes:
  - robot_joint.npy must be [T, 6] joint angles in radians.
  - gripper.npy, when enabled, is interpreted as gripper_travel in [0, 12000].
  - Playback uses MovJ to reach the first frame, then ServoJ at --hz.
  - Ctrl+C stops the arm command stream and disconnects; it does not DisableRobot.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_DOBOT_IP = "192.168.5.1"
DEFAULT_DOBOT_DASHBOARD_PORT = 29999
DEFAULT_DOBOT_FEEDBACK_PORT = 30004
DEFAULT_GRIPPER_PORT = "/dev/ttyUSB1"
DEFAULT_GRIPPER_SLAVE_ID = 1
DEFAULT_GRIPPER_BAUDRATE = 115200
DEFAULT_GRIPPER_SPEED_PCT = 30
DEFAULT_GRIPPER_FORCE_PCT = 50
DEFAULT_GRIPPER_POS_CLOSE = 12000
GRIPPER_MAX_M = 0.085


@dataclass(frozen=True)
class PlaybackData:
    episode_dir: Path
    joints_rad: np.ndarray
    gripper_travel: np.ndarray | None
    timestamps: np.ndarray | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Play ~/temp/Dobot/episode_x/robot_joint.npy on a Dobot arm using the same "
            "SDK ServoJ path as the online rollout adapter."
        )
    )
    parser.add_argument("--episode-dir", type=Path, default=Path("/home/lry/temp/Dobot/episode_0"))
    parser.add_argument("--hz", type=float, default=20.0, help="ServoJ command send frequency.")
    parser.add_argument("--dobot-ip", type=str, default=DEFAULT_DOBOT_IP)
    parser.add_argument("--dashboard-port", type=int, default=DEFAULT_DOBOT_DASHBOARD_PORT)
    parser.add_argument("--feedback-port", type=int, default=DEFAULT_DOBOT_FEEDBACK_PORT)
    parser.add_argument("--gripper-port", type=str, default=DEFAULT_GRIPPER_PORT)
    parser.add_argument("--gripper-slave-id", type=int, default=DEFAULT_GRIPPER_SLAVE_ID)
    parser.add_argument("--gripper-baudrate", type=int, default=DEFAULT_GRIPPER_BAUDRATE)
    parser.add_argument("--gripper-speed-pct", type=int, default=DEFAULT_GRIPPER_SPEED_PCT)
    parser.add_argument("--gripper-force-pct", type=int, default=DEFAULT_GRIPPER_FORCE_PCT)
    parser.add_argument("--max-gripper-m", type=float, default=GRIPPER_MAX_M)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--startup-speed-pct", type=int, default=20)
    parser.add_argument("--startup-hold-sec", type=float, default=1.0)
    parser.add_argument(
        "--max-delta-rad",
        type=float,
        default=0.0,
        help="Optional per-joint command delta clamp. 0 disables clipping.",
    )
    parser.add_argument("--disable-gripper", action="store_true")
    parser.add_argument("--no-enable", action="store_true", help="Do not call EnableRobot before playback.")
    parser.add_argument("--dry-run", action="store_true", help="Load and print stats only; do not connect hardware.")
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.hz <= 0:
        raise ValueError("--hz must be > 0")
    if args.max_gripper_m <= 0:
        raise ValueError("--max-gripper-m must be > 0")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0 when provided")
    if args.startup_hold_sec < 0:
        raise ValueError("--startup-hold-sec must be >= 0")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be > 0")
    if args.max_delta_rad < 0:
        raise ValueError("--max-delta-rad must be >= 0")


def _gripper_travel_to_m(value: float, *, max_gripper_m: float = GRIPPER_MAX_M) -> float:
    travel = float(np.clip(value, 0.0, DEFAULT_GRIPPER_POS_CLOSE))
    ratio_closed = travel / max(float(DEFAULT_GRIPPER_POS_CLOSE), 1.0)
    return float(np.clip((1.0 - ratio_closed) * max_gripper_m, 0.0, max_gripper_m))


def _load_data(args: argparse.Namespace) -> PlaybackData:
    episode_dir = args.episode_dir.expanduser().resolve()
    joint_path = episode_dir / "robot_joint.npy"
    gripper_path = episode_dir / "gripper.npy"
    timestamps_path = episode_dir / "timestamps.npy"
    if not joint_path.exists():
        raise FileNotFoundError(f"Missing {joint_path}")

    joints = np.asarray(np.load(joint_path), dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 6:
        raise ValueError(f"robot_joint.npy must have shape [T, 6] in radians, got {joints.shape}")
    if not np.all(np.isfinite(joints)):
        raise ValueError("robot_joint.npy contains NaN or inf values")

    gripper = None
    if gripper_path.exists() and not args.disable_gripper:
        gripper = np.asarray(np.load(gripper_path), dtype=np.float64).reshape(-1)
        if gripper.shape[0] != joints.shape[0]:
            raise ValueError(f"gripper.npy length {gripper.shape[0]} does not match joint length {joints.shape[0]}")
        if not np.all(np.isfinite(gripper)):
            raise ValueError("gripper.npy contains NaN or inf values")

    timestamps = None
    if timestamps_path.exists():
        timestamps = np.asarray(np.load(timestamps_path), dtype=np.float64).reshape(-1)
        if timestamps.shape[0] != joints.shape[0]:
            timestamps = None

    start = max(int(args.start_index), 0)
    if start >= joints.shape[0]:
        raise ValueError(f"start-index={start} out of range for trajectory length {joints.shape[0]}")
    stop = joints.shape[0] if args.max_steps is None else min(joints.shape[0], start + max(int(args.max_steps), 0))
    if stop <= start:
        raise ValueError(f"Selected empty range start={start}, stop={stop}")

    return PlaybackData(
        episode_dir=episode_dir,
        joints_rad=joints[start:stop].copy(),
        gripper_travel=None if gripper is None else gripper[start:stop].copy(),
        timestamps=None if timestamps is None else timestamps[start:stop].copy(),
    )


def _print_stats(data: PlaybackData, args: argparse.Namespace) -> None:
    q = data.joints_rad
    dq = np.diff(q, axis=0) if q.shape[0] > 1 else np.zeros((0, 6), dtype=np.float64)
    abs_dq = np.abs(dq)
    duration_from_hz = (q.shape[0] - 1) / max(float(args.hz), 1e-6)
    print(f"Episode dir: {data.episode_dir}")
    print(f"Selected steps: {q.shape[0]}")
    print(f"Playback Hz: {args.hz:.3f}")
    print(f"Playback duration from Hz: {duration_from_hz:.3f}s")
    if data.timestamps is not None and data.timestamps.shape[0] > 1:
        dt = np.diff(data.timestamps)
        print(
            "Recorded timestamps: "
            f"duration={data.timestamps[-1] - data.timestamps[0]:.3f}s "
            f"mean_hz={1.0 / max(float(np.mean(dt)), 1e-9):.3f} "
            f"dt_mean={np.mean(dt) * 1000.0:.2f}ms"
        )
    print(f"Joint min deg: {np.round(np.rad2deg(np.min(q, axis=0)), 3).tolist()}")
    print(f"Joint max deg: {np.round(np.rad2deg(np.max(q, axis=0)), 3).tolist()}")
    if abs_dq.size:
        print(f"Joint max step deg: {np.round(np.rad2deg(np.max(abs_dq, axis=0)), 4).tolist()}")
        print(f"Joint mean step deg: {np.round(np.rad2deg(np.mean(abs_dq, axis=0)), 4).tolist()}")
        print(f"Global max joint step deg: {float(np.rad2deg(np.max(abs_dq))):.4f}")
    if data.gripper_travel is None:
        print("Gripper: disabled or gripper.npy missing")
    else:
        g = data.gripper_travel
        openings = np.asarray([_gripper_travel_to_m(x, max_gripper_m=args.max_gripper_m) for x in g])
        print(f"Gripper travel min/max: {float(np.min(g)):.1f} / {float(np.max(g)):.1f}")
        print(f"Gripper opening min/max mm: {float(np.min(openings) * 1000.0):.2f} / {float(np.max(openings) * 1000.0):.2f}")
    print(f"Startup MovJ speed: {args.startup_speed_pct}%")
    print(f"Startup hold: {args.startup_hold_sec:.3f}s")
    print(f"Max delta clamp: {'disabled' if args.max_delta_rad <= 0 else f'{args.max_delta_rad:.6f} rad'}")


def _import_dobot_classes():
    repo_root = Path(__file__).resolve().parents[3]
    adapter_dir = repo_root / "rlt_online_rl" / "train_deploy_alignment"
    src_root = repo_root / "rlt_online_rl" / "src"
    openpi_client_src = repo_root / "packages" / "openpi-client" / "src"
    for path in (adapter_dir, src_root, openpi_client_src):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from dobot_umi_ros import DobotSDKArm, ZhixingSDKGripper

    return DobotSDKArm, ZhixingSDKGripper


def _maybe_clamp_delta(target: np.ndarray, previous: np.ndarray, max_delta_rad: float) -> np.ndarray:
    if max_delta_rad <= 0:
        return target
    delta = np.clip(target - previous, -max_delta_rad, max_delta_rad)
    return previous + delta


def _sleep_until(deadline: float) -> None:
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.005))


def _play(data: PlaybackData, args: argparse.Namespace) -> None:
    DobotSDKArm, ZhixingSDKGripper = _import_dobot_classes()
    arm = DobotSDKArm(args.dobot_ip, args.dashboard_port, args.feedback_port)
    gripper = None
    servo_t = max(0.02, 1.0 / max(float(args.hz), 1e-6))
    step_period = 1.0 / max(float(args.hz), 1e-6)
    sent = 0
    failed = 0
    started_at = None
    try:
        arm.connect()
        if not args.no_enable and not arm.enable():
            raise RuntimeError("EnableRobot failed")

        if data.gripper_travel is not None and not args.disable_gripper:
            gripper = ZhixingSDKGripper(
                args.gripper_port,
                args.gripper_slave_id,
                args.gripper_baudrate,
                args.gripper_speed_pct,
                args.gripper_force_pct,
            )
            if not gripper.init():
                raise RuntimeError("Gripper init failed")

        first = data.joints_rad[0]
        print("Moving to first frame with MovJ...")
        if not arm.move_j(first, wait=True, timeout=30.0, speed_pct=args.startup_speed_pct):
            raise RuntimeError("Startup MovJ failed or timed out")
        if gripper is not None and data.gripper_travel is not None:
            gripper.set_opening_m(_gripper_travel_to_m(data.gripper_travel[0], max_gripper_m=args.max_gripper_m))
        if args.startup_hold_sec > 0:
            print(f"Holding first frame for {args.startup_hold_sec:.3f}s...")
            time.sleep(args.startup_hold_sec)

        print("Starting ServoJ replay...")
        previous_cmd = first.copy()
        started_at = time.perf_counter()
        next_tick = started_at
        progress_interval = max(int(args.progress_interval), 1)
        for step_idx, raw_target in enumerate(data.joints_rad):
            target = _maybe_clamp_delta(raw_target, previous_cmd, float(args.max_delta_rad))
            previous_cmd = target.copy()
            # breakpoint()  # For debugging: inspect target, previous_cmd, raw_target
            ok = arm.servo_j(target, t=servo_t)
            sent += 1
            if not ok:
                failed += 1
            if gripper is not None and data.gripper_travel is not None:
                gripper.set_opening_m(
                    _gripper_travel_to_m(data.gripper_travel[step_idx], max_gripper_m=args.max_gripper_m)
                )
            if step_idx == 0 or (step_idx + 1) % progress_interval == 0 or step_idx + 1 == data.joints_rad.shape[0]:
                elapsed = time.perf_counter() - started_at
                actual_hz = sent / max(elapsed, 1e-9)
                print(f"step={step_idx + 1}/{data.joints_rad.shape[0]} elapsed={elapsed:.2f}s actual_hz={actual_hz:.2f} failed={failed}")
            next_tick += step_period
            _sleep_until(next_tick)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        try:
            arm.stop()
        except Exception:
            pass
        try:
            arm.disconnect()
        except Exception:
            pass
        if gripper is not None:
            try:
                gripper.release()
            except Exception:
                pass
        if started_at is not None:
            elapsed = time.perf_counter() - started_at
            print(f"Finished playback sent={sent} failed={failed} elapsed={elapsed:.3f}s avg_hz={sent / max(elapsed, 1e-9):.3f}")


def main() -> None:
    args = _parse_args()
    data = _load_data(args)
    _print_stats(data, args)
    if args.dry_run:
        return
    _play(data, args)


if __name__ == "__main__":
    main()
