"""Hardware connectivity checks for dual Rokae arms and Zhixing grippers.

Checks:
  1. Two Rokae arms over xCoreSDK/pyrokae. Read-only: no power, mode, or motion commands.
  2. Two Zhixing grippers over RS-485. Optional open/close test only with --gripper_test.
  3. RealSense cameras through pyrealsense2.
  4. Optional VitAI GF225 tactile sensors.

Examples:
    python examples/rokae_zhixing_dual/check_hardware.py
    python examples/rokae_zhixing_dual/check_hardware.py --gripper_test
    python examples/rokae_zhixing_dual/check_hardware.py --skip_tactile
    python examples/rokae_zhixing_dual/check_hardware.py --skip_arm --skip_gripper
"""
# ruff: noqa
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOBOT_SDK_ROOT = _REPO_ROOT / "third_party" / "dobot_umi_sdk"
for _p in (
    str(_REPO_ROOT),
    str(_DOBOT_SDK_ROOT),
    str(_DOBOT_SDK_ROOT / "adaptive_sdk"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from examples.rokae_zhixing_dual import constants


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


def _load_pyrokae(sdk_python_dir: str) -> Any:
    sdk_dir = Path(sdk_python_dir)
    if str(sdk_dir) not in sys.path:
        sys.path.insert(0, str(sdk_dir))
    try:
        import pyrokae  # type: ignore
    except ImportError as exc:
        print(_fail(f"pyrokae import failed from {sdk_dir}: {exc}"))
        print(_warn("Check Python version. The provided SDK file name suggests cpython-310."))
        return None
    return pyrokae


def check_rokae_arm(
    *,
    label: str,
    remote_ip: str,
    local_ip: str,
    sdk_python_dir: str,
) -> bool:
    print(_title(f"[1] Rokae arm: {label}"))
    pyrokae = _load_pyrokae(sdk_python_dir)
    if pyrokae is None:
        return False

    robot = None
    try:
        print(f"   Connecting RokaeAR remote={remote_ip}, local={local_ip} ...")
        robot = pyrokae.RokaeAR(remote_ip, local_ip)
        print(_ok("RokaeAR object created"))
    except Exception as exc:
        print(_fail(f"RokaeAR connection failed: {exc}"))
        return False

    ok = True
    read_checks = (
        ("sdkVersion", lambda: robot.sdkVersion()),
        ("robotInfo", lambda: robot.robotInfo()),
        ("powerState", lambda: robot.getPowerState()),
        ("operateMode", lambda: robot.getOperateMode()),
        ("operationState", lambda: robot.getOperationState()),
        ("jointPos", lambda: robot.getJointPos()),
        ("jointVel", lambda: robot.getJointVel()),
        ("endPose", lambda: robot.getEndPose()),
    )
    for name, getter in read_checks:
        try:
            value = getter()
            if name == "jointPos":
                arr = np.asarray(value, dtype=np.float64)
                print(_ok(f"{name}: {np.round(arr, 5).tolist()} rad"))
            elif name == "jointVel":
                arr = np.asarray(value, dtype=np.float64)
                print(_ok(f"{name}: {np.round(arr, 5).tolist()} rad/s"))
            elif name == "endPose":
                arr = np.asarray(value, dtype=np.float64)
                print(_ok(f"{name}: {np.round(arr, 5).tolist()}"))
            elif name == "robotInfo":
                joint_num = getattr(value, "joint_num", "?")
                robot_type = getattr(value, "type", "?")
                robot_id = getattr(value, "id", "?")
                version = getattr(value, "version", "?")
                print(_ok(f"{name}: id={robot_id}, type={robot_type}, version={version}, joint_num={joint_num}"))
                if joint_num != constants.ARM_DOF:
                    print(_warn(f"Expected {constants.ARM_DOF} joints, got {joint_num}"))
            else:
                print(_ok(f"{name}: {value}"))
        except Exception as exc:
            print(_fail(f"{name} read failed: {exc}"))
            ok = False

    if ok:
        print(_ok(f"{label} Rokae arm check passed; no motion command was sent"))
    return ok


def check_dual_rokae_arms(args: argparse.Namespace) -> bool:
    if args.left_arm_remote_ip == args.right_arm_remote_ip:
        print(_title("[1] Rokae dual arms"))
        print(_fail("left and right remote_ip are identical; this is unsafe for dual-arm rollout"))
        return False

    left_ok = check_rokae_arm(
        label="left",
        remote_ip=args.left_arm_remote_ip,
        local_ip=args.left_arm_local_ip,
        sdk_python_dir=args.rokae_sdk_python_dir,
    )
    right_ok = check_rokae_arm(
        label="right",
        remote_ip=args.right_arm_remote_ip,
        local_ip=args.right_arm_local_ip,
        sdk_python_dir=args.rokae_sdk_python_dir,
    )
    return left_ok and right_ok


def _encoder_to_m(pos: float) -> float:
    ratio = float(pos) / max(constants.GRIPPER_POS_CLOSE, 1)
    return (1.0 - ratio) * constants.GRIPPER_OPEN_M


def check_gripper(
    *,
    label: str,
    port: str,
    slave_id: int,
    baudrate: int,
    speed_pct: int,
    force_pct: int,
    do_motion_test: bool,
) -> bool:
    print(_title(f"[2] Zhixing gripper: {label}"))
    try:
        from changingtek_p_rtu_Servo import MotorController  # type: ignore
    except ImportError as exc:
        print(_fail(f"MotorController import failed: {exc}"))
        return False

    motor = None
    try:
        print(f"   Opening {port}, slave_id={slave_id}, baudrate={baudrate} ...")
        motor = MotorController(port, slave_id, baudrate, 0.5)
        motor.set_target_speed(speed_pct)
        motor.set_target_force(force_pct)
        motor.set_target_acceleration(2000)
        motor.set_target_deceleration(2000)
        print(_ok("serial initialized"))
    except Exception as exc:
        print(_fail(f"gripper initialization failed: {exc}"))
        return False

    try:
        pos = motor.read_real_position()
        print(_ok(f"position encoder={pos}, opening={_encoder_to_m(pos) * 1000.0:.1f} mm"))
    except Exception as exc:
        print(_fail(f"position read failed: {exc}"))
        return False

    if not do_motion_test:
        print(_warn("motion test skipped; add --gripper_test to open/close the gripper"))
        return True

    try:
        print("   Motion test: open ...")
        motor.set_target_position(constants.GRIPPER_POS_OPEN)
        motor.trigger_motion()
        time.sleep(2.0)
        pos_open = motor.read_real_position()
        print(_ok(f"open encoder={pos_open}"))

        print("   Motion test: close ...")
        motor.set_target_position(constants.GRIPPER_POS_CLOSE)
        motor.trigger_motion()
        time.sleep(2.0)
        pos_close = motor.read_real_position()
        print(_ok(f"close encoder={pos_close}"))

        print("   Motion test: restore open ...")
        motor.set_target_position(constants.GRIPPER_POS_OPEN)
        motor.trigger_motion()
        time.sleep(1.5)
        print(_ok("gripper motion test passed"))
        return True
    except Exception as exc:
        print(_fail(f"gripper motion test failed: {exc}"))
        return False


def check_dual_grippers(args: argparse.Namespace) -> bool:
    if (
        args.left_gripper_port == args.right_gripper_port
        and args.left_gripper_slave_id == args.right_gripper_slave_id
    ):
        print(_title("[2] Zhixing dual grippers"))
        print(_fail("left and right grippers share the same port and slave id"))
        return False
    left_ok = check_gripper(
        label="left",
        port=args.left_gripper_port,
        slave_id=args.left_gripper_slave_id,
        baudrate=args.gripper_baudrate,
        speed_pct=args.gripper_speed_pct,
        force_pct=args.gripper_force_pct,
        do_motion_test=args.gripper_test,
    )
    right_ok = check_gripper(
        label="right",
        port=args.right_gripper_port,
        slave_id=args.right_gripper_slave_id,
        baudrate=args.gripper_baudrate,
        speed_pct=args.gripper_speed_pct,
        force_pct=args.gripper_force_pct,
        do_motion_test=args.gripper_test,
    )
    return left_ok and right_ok


def _resolve_camera_serials(
    connected: list[str],
    requested: dict[str, str],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    used: set[str] = set()
    for key, serial in requested.items():
        if serial:
            if serial not in connected:
                raise RuntimeError(f"{key} serial {serial} is not connected; connected={connected}")
            resolved[key] = serial
            used.add(serial)

    for key in (
        constants.IMAGE_KEY_FRONT,
        constants.IMAGE_KEY_LEFT_WRIST,
        constants.IMAGE_KEY_RIGHT_WRIST,
    ):
        if key in resolved:
            continue
        remaining = [serial for serial in connected if serial not in used]
        if remaining:
            resolved[key] = remaining[0]
            used.add(remaining[0])

    if constants.IMAGE_KEY_FRONT not in resolved and connected:
        resolved[constants.IMAGE_KEY_FRONT] = connected[0]
    if constants.IMAGE_KEY_LEFT_WRIST not in resolved:
        resolved[constants.IMAGE_KEY_LEFT_WRIST] = resolved[constants.IMAGE_KEY_FRONT]
    if constants.IMAGE_KEY_RIGHT_WRIST not in resolved:
        resolved[constants.IMAGE_KEY_RIGHT_WRIST] = resolved[constants.IMAGE_KEY_LEFT_WRIST]
    return resolved


def check_cameras(args: argparse.Namespace) -> bool:
    print(_title("[3] RealSense cameras"))
    try:
        import pyrealsense2 as rs  # type: ignore
    except ImportError:
        print(_fail("pyrealsense2 is not installed"))
        return False

    ctx = rs.context()
    devices = list(ctx.query_devices())
    connected = [device.get_info(rs.camera_info.serial_number) for device in devices]
    names = {
        device.get_info(rs.camera_info.serial_number): device.get_info(rs.camera_info.name)
        for device in devices
    }
    print(f"   Detected {len(connected)} RealSense devices: {connected}")
    if not connected:
        print(_fail("no RealSense device detected"))
        return False

    try:
        resolved = _resolve_camera_serials(
            connected,
            {
                constants.IMAGE_KEY_FRONT: args.cam_front_serial,
                constants.IMAGE_KEY_LEFT_WRIST: args.cam_left_wrist_serial,
                constants.IMAGE_KEY_RIGHT_WRIST: args.cam_right_wrist_serial,
            },
        )
    except Exception as exc:
        print(_fail(str(exc)))
        return False

    ok = True
    tested_serials: set[str] = set()
    for label, serial in resolved.items():
        if serial in tested_serials:
            print(_warn(f"{label}: shares SN={serial}; frame capture already checked"))
            continue
        tested_serials.add(serial)
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, args.cam_width, args.cam_height, rs.format.rgb8, args.cam_fps)
        try:
            pipeline.start(cfg)
            frames = pipeline.wait_for_frames(timeout_ms=3000)
            color = frames.get_color_frame()
            if color is None:
                print(_fail(f"{label}: SN={serial} no color frame"))
                ok = False
                continue
            img = np.asanyarray(color.get_data())
            print(_ok(f"{label}: SN={serial}, {names.get(serial, '')}, {img.shape[1]}x{img.shape[0]} RGB8"))
        except Exception as exc:
            print(_fail(f"{label}: SN={serial} capture failed: {exc}"))
            ok = False
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass
    return ok


def check_tactile(args: argparse.Namespace) -> bool:
    print(_title("[4] VitAI GF225 tactile sensors"))
    if not args.tactile_sn_left or not args.tactile_sn_right:
        print(_fail("tactile serial numbers are not configured"))
        return False
    try:
        from examples.rokae_zhixing_dual.robot_utils import VitAITactileCamera
    except ImportError as exc:
        print(_fail(f"VitAITactileCamera import failed: {exc}"))
        return False

    tactile = VitAITactileCamera(
        serial_left=args.tactile_sn_left,
        serial_right=args.tactile_sn_right,
    )
    try:
        print(f"   Connecting tactile left={args.tactile_sn_left}, right={args.tactile_sn_right} ...")
        if not tactile.connect():
            print(_fail("tactile connection failed"))
            return False
        images = tactile.get_images()
        ok = True
        for key in (constants.IMAGE_KEY_TACTILE_LEFT, constants.IMAGE_KEY_TACTILE_RIGHT):
            img = images.get(key)
            if img is None:
                print(_fail(f"{key}: missing image"))
                ok = False
                continue
            if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
                print(_fail(f"{key}: unexpected image shape/dtype {img.shape} {img.dtype}"))
                ok = False
                continue
            print(_ok(f"{key}: {img.shape[1]}x{img.shape[0]} RGB uint8"))
        return ok
    except Exception as exc:
        print(_fail(f"tactile capture failed: {exc}"))
        return False
    finally:
        try:
            tactile.disconnect()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dual Rokae + Zhixing + RealSense hardware connectivity check"
    )
    parser.add_argument("--rokae_sdk_python_dir", default=constants.ROKAE_SDK_PYTHON_DIR)
    parser.add_argument("--left_arm_remote_ip", default=constants.LEFT_ARM_REMOTE_IP)
    parser.add_argument("--left_arm_local_ip", default=constants.LEFT_ARM_LOCAL_IP)
    parser.add_argument("--right_arm_remote_ip", default=constants.RIGHT_ARM_REMOTE_IP)
    parser.add_argument("--right_arm_local_ip", default=constants.RIGHT_ARM_LOCAL_IP)

    parser.add_argument("--left_gripper_port", default=constants.LEFT_GRIPPER_SERIAL_PORT)
    parser.add_argument("--left_gripper_slave_id", default=constants.LEFT_GRIPPER_SLAVE_ID, type=int)
    parser.add_argument("--right_gripper_port", default=constants.RIGHT_GRIPPER_SERIAL_PORT)
    parser.add_argument("--right_gripper_slave_id", default=constants.RIGHT_GRIPPER_SLAVE_ID, type=int)
    parser.add_argument("--gripper_baudrate", default=constants.GRIPPER_BAUDRATE, type=int)
    parser.add_argument("--gripper_speed_pct", default=constants.GRIPPER_SPEED_PCT, type=int)
    parser.add_argument("--gripper_force_pct", default=constants.GRIPPER_FORCE_PCT, type=int)
    parser.add_argument("--gripper_test", action="store_true", help="Actually open/close both grippers.")

    parser.add_argument("--cam_front_serial", default=constants.REALSENSE_FRONT_SERIAL)
    parser.add_argument("--cam_left_wrist_serial", default=constants.REALSENSE_LEFT_WRIST_SERIAL)
    parser.add_argument("--cam_right_wrist_serial", default=constants.REALSENSE_RIGHT_WRIST_SERIAL)
    parser.add_argument("--cam_width", default=constants.REALSENSE_WIDTH, type=int)
    parser.add_argument("--cam_height", default=constants.REALSENSE_HEIGHT, type=int)
    parser.add_argument("--cam_fps", default=constants.REALSENSE_FPS, type=int)

    parser.add_argument("--tactile_sn_left", default=constants.TACTILE_LEFT_SERIAL)
    parser.add_argument("--tactile_sn_right", default=constants.TACTILE_RIGHT_SERIAL)

    parser.add_argument("--skip_arm", action="store_true")
    parser.add_argument("--skip_gripper", action="store_true")
    parser.add_argument("--skip_cameras", action="store_true")
    parser.add_argument("--skip_tactile", action="store_true")
    args = parser.parse_args()

    print(f"\n{'=' * 72}")
    print("  Dual Rokae / Zhixing / RealSense / VitAI hardware connectivity check")
    print(f"{'=' * 72}")
    print(_warn("Rokae arm checks are read-only. No arm motion command will be sent."))

    results: dict[str, bool | None] = {}
    if args.skip_arm:
        print(_title("[1] Rokae dual arms"))
        print(_warn("skipped by --skip_arm"))
        results["arms"] = None
    else:
        results["arms"] = check_dual_rokae_arms(args)

    if args.skip_gripper:
        print(_title("[2] Zhixing dual grippers"))
        print(_warn("skipped by --skip_gripper"))
        results["grippers"] = None
    else:
        results["grippers"] = check_dual_grippers(args)

    if args.skip_cameras:
        print(_title("[3] RealSense cameras"))
        print(_warn("skipped by --skip_cameras"))
        results["cameras"] = None
    else:
        results["cameras"] = check_cameras(args)

    if args.skip_tactile:
        print(_title("[4] VitAI GF225 tactile sensors"))
        print(_warn("skipped by --skip_tactile"))
        results["tactile"] = None
    else:
        results["tactile"] = check_tactile(args)

    print(f"\n{'=' * 72}")
    print("  Summary")
    print(f"{'=' * 72}")
    labels = {
        "arms": "Rokae dual arms",
        "grippers": "Zhixing grippers",
        "cameras": "RealSense cameras",
        "tactile": "VitAI tactile",
    }
    all_pass = True
    for key, label in labels.items():
        value = results.get(key)
        if value is True:
            print(f"  {label:<20} {_GREEN}PASS{_RESET}")
        elif value is False:
            print(f"  {label:<20} {_RED}FAIL{_RESET}")
            all_pass = False
        else:
            print(f"  {label:<20} {_YELLOW}SKIP{_RESET}")
    print(f"{'=' * 72}")
    if all_pass:
        print(f"  {_GREEN}{_BOLD}All required checks passed.{_RESET}\n")
    else:
        print(f"  {_RED}{_BOLD}Some checks failed. Inspect the failed hardware path before rollout.{_RESET}\n")
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
