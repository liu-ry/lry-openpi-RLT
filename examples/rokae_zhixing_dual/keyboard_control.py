#!/usr/bin/env python3
"""Keyboard Cartesian jog for dual Rokae arms.

Controls:
  l / r             switch selected arm
  Arrow Up/Down    +/- X end-pose position
  Arrow Left/Right +/- Y end-pose position
  PageUp/PageDown  +/- Z end-pose position
  p                print current end pose of both arms
  space            stop both arms
  q or Ctrl-C      quit

Each key press sends one small MoveJ step through inverseKinematics().
Default step is 0.5 mm; use a smaller --step-mm for first contact tests.
"""
# ruff: noqa
from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rokae_zhixing_dual import constants
from examples.rokae_zhixing_dual import robot_utils as _utils


class RawKeyboard:
    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._old: list | None = None

    def __enter__(self) -> "RawKeyboard":
        self._old = termios.tcgetattr(self._fd)
        tty.setraw(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def read_key(self, timeout_s: float = 0.1) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if not ready:
            return None
        ch = sys.stdin.read(1)
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch != "\x1b":
            return ch

        ready, _, _ = select.select([sys.stdin], [], [], 0.02)
        if not ready:
            return "\x1b"
        seq = ch + sys.stdin.read(1)
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.02)
            if not ready:
                break
            seq += sys.stdin.read(1)
            if seq.endswith("~") or len(seq) >= 6:
                break
        return {
            "\x1b[A": "up",
            "\x1b[B": "down",
            "\x1b[C": "right",
            "\x1b[D": "left",
            "\x1b[5~": "pageup",
            "\x1b[6~": "pagedown",
        }.get(seq, seq)


class DualRokaeKeyboardControl:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.step_m = float(np.clip(args.step_mm, 0.01, args.max_step_mm)) * 1e-3
        self.selected = "left"
        self.arms = _utils.DualRokaeArm(
            left=_utils.RokaeSDKArm(
                remote_ip=args.left_arm_remote_ip,
                local_ip=args.left_arm_local_ip,
                name="left",
                sdk_python_dir=args.rokae_sdk_python_dir,
                movej_speed=args.movej_speed,
            ),
            right=_utils.RokaeSDKArm(
                remote_ip=args.right_arm_remote_ip,
                local_ip=args.right_arm_local_ip,
                name="right",
                sdk_python_dir=args.rokae_sdk_python_dir,
                movej_speed=args.movej_speed,
            ),
        )
        self.key_handlers = {
            "up": lambda: self.jog(dx=self.step_m),
            "down": lambda: self.jog(dx=-self.step_m),
            "left": lambda: self.jog(dy=self.step_m),
            "right": lambda: self.jog(dy=-self.step_m),
            "pageup": lambda: self.jog(dz=self.step_m),
            "pagedown": lambda: self.jog(dz=-self.step_m),
            "l": lambda: self.select("left"),
            "L": lambda: self.select("left"),
            "r": lambda: self.select("right"),
            "R": lambda: self.select("right"),
            "p": self.print_poses,
            "P": self.print_poses,
            " ": self.stop,
        }

    def connect(self) -> None:
        if self.args.left_arm_remote_ip == self.args.right_arm_remote_ip:
            raise ValueError("left and right remote_ip are identical")
        self.arms.connect()
        if self.args.power_on:
            self.arms.left.move_j(
                self.arms.left.get_joint_angles_rad(),
                wait=True,
                timeout=self.args.timeout_s,
                restore_realtime=False,
            )
            self.arms.right.move_j(
                self.arms.right.get_joint_angles_rad(),
                wait=True,
                timeout=self.args.timeout_s,
                restore_realtime=False,
            )
        self.print_poses()

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.arms.disconnect()

    def active_arm(self) -> _utils.RokaeSDKArm:
        return self.arms.left if self.selected == "left" else self.arms.right

    def select(self, name: str) -> None:
        self.selected = name
        print(f"\nSelected arm: {self.selected}", flush=True)

    def print_poses(self) -> None:
        left_pose = self.arms.left.get_end_pose()
        right_pose = self.arms.right.get_end_pose()
        print(
            "\nCurrent end poses:\n"
            f"  left : {np.round(left_pose, 6).tolist()}\n"
            f"  right: {np.round(right_pose, 6).tolist()}\n"
            f"Selected: {self.selected}, step={self.step_m * 1000.0:.3f} mm",
            flush=True,
        )

    def jog(self, *, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        arm = self.active_arm()
        current = arm.get_end_pose()
        if current.size != 6 or not np.all(np.isfinite(current)):
            raise RuntimeError(f"invalid current pose for {self.selected}: {current}")
        target = current.copy()
        target[0] += dx
        target[1] += dy
        target[2] += dz
        print(
            f"\n{self.selected} target pose: {np.round(target, 6).tolist()}",
            flush=True,
        )
        arm.move_j_pose(
            target,
            wait=True,
            timeout=self.args.timeout_s,
            speed=self.args.movej_speed,
            restore_realtime=False,
        )

    def stop(self) -> None:
        self.arms.stop()
        print("\nStop sent to both arms", flush=True)

    def run(self) -> None:
        print_controls()
        with RawKeyboard() as keyboard:
            while True:
                key = keyboard.read_key(timeout_s=0.1)
                if key is None:
                    continue
                if key in ("q", "Q"):
                    break
                handler = self.key_handlers.get(key)
                if handler is not None:
                    try:
                        handler()
                    except Exception as exc:
                        self.stop()
                        print(f"\nCommand failed: {exc}", flush=True)


def print_controls() -> None:
    print(
        "\nControls:\n"
        "  l / r: switch selected arm\n"
        "  Arrow Up/Down: +/- X end-pose jog\n"
        "  Arrow Left/Right: +/- Y end-pose jog\n"
        "  PageUp/PageDown: +/- Z end-pose jog\n"
        "  p: print current poses\n"
        "  space: stop both arms\n"
        "  q: quit\n",
        flush=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dual Rokae keyboard Cartesian jog.")
    parser.add_argument("--rokae-sdk-python-dir", default=constants.ROKAE_SDK_PYTHON_DIR)
    parser.add_argument("--left-arm-remote-ip", default=constants.LEFT_ARM_REMOTE_IP)
    parser.add_argument("--left-arm-local-ip", default=constants.LEFT_ARM_LOCAL_IP)
    parser.add_argument("--right-arm-remote-ip", default=constants.RIGHT_ARM_REMOTE_IP)
    parser.add_argument("--right-arm-local-ip", default=constants.RIGHT_ARM_LOCAL_IP)
    parser.add_argument("--step-mm", type=float, default=0.5, help="Cartesian step per key press in mm.")
    parser.add_argument("--max-step-mm", type=float, default=5.0, help="Upper safety clamp for --step-mm.")
    parser.add_argument("--movej-speed", type=float, default=5.0, help="Rokae moveJ_joint speed.")
    parser.add_argument("--timeout-s", type=float, default=10.0, help="Per-step moveWait timeout.")
    parser.add_argument(
        "--power-on",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recover, set automatic/NrtCommand, and power on each arm before keyboard control.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    app = DualRokaeKeyboardControl(args)
    try:
        app.connect()
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()


if __name__ == "__main__":
    main()
