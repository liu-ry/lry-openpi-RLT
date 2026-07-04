"""Hardware connectivity checks for dual Tianji Marvin arms.

Examples:
    python examples/tianji_marvin_dual/check_hardware.py
"""
# ruff: noqa
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.tianji_marvin_dual import constants
from examples.tianji_marvin_dual import robot_utils


_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _ok(msg: str) -> str:
    return f"{_GREEN}PASS  {msg}{_RESET}"


def _fail(msg: str) -> str:
    return f"{_RED}FAIL  {msg}{_RESET}"


def _warn(msg: str) -> str:
    return f"{_YELLOW}WARN  {msg}{_RESET}"


def _title(msg: str) -> str:
    return f"\n{_BOLD}{msg}{_RESET}"


def check_tianji_arms(args: argparse.Namespace) -> bool:
    print(_title("[1] Tianji Marvin dual arms"))
    arms = robot_utils.DualTianjiMarvinArm(
        left=robot_utils.TianjiMarvinSDKArm(arm=constants.LEFT_ARM_ID, name="left"),
        right=robot_utils.TianjiMarvinSDKArm(arm=constants.RIGHT_ARM_ID, name="right"),
        robot_ip=args.robot_ip,
        sdk_python_dir=args.tianji_sdk_python_dir,
        config_path=args.tianji_kinematics_config_path,
    )
    try:
        print(f"   Connecting robot_ip={args.robot_ip} ...")
        arms.connect()
        print(_ok("MarvinSDK connected"))
        for label, arm in (("left/A", arms.left), ("right/B", arms.right)):
            joints = arm.get_joint_angles_rad()
            pose = arm.get_end_pose()
            print(_ok(f"{label} joints rad: {np.round(joints, 5).tolist()}"))
            print(_ok(f"{label} tcp pose m/rad: {np.round(pose, 5).tolist()}"))
        if args.enable_position_state:
            arms.enable()
            print(_ok("position state enabled for both arms"))
        else:
            print(_warn("position-state enable skipped; pass --enable_position_state to test mode switch"))
        return True
    except Exception as exc:
        print(_fail(f"Tianji Marvin check failed: {exc}"))
        return False
    finally:
        if args.enable_position_state:
            try:
                arms.disable()
            except Exception as exc:
                print(_warn(f"disable after position-state check failed: {exc}"))
        arms.disconnect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Tianji Marvin dual-arm SDK connectivity")
    parser.add_argument("--robot_ip", default=constants.ROBOT_IP)
    parser.add_argument("--tianji_sdk_python_dir", default=constants.TIANJI_SDK_PYTHON_DIR)
    parser.add_argument("--tianji_kinematics_config_path", default=constants.TIANJI_KINEMATICS_CONFIG_PATH)
    parser.add_argument("--enable_position_state", action="store_true")
    return parser.parse_args()


def main() -> None:
    ok = check_tianji_arms(_parse_args())
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
