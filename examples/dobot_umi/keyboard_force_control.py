#!/usr/bin/env python3
"""Keyboard Cartesian jog and Dobot force-compliance test.

Usage:
  python3 examples/dobot_umi/keyboard_force_control.py

Safer first-run example:
  python3 examples/dobot_umi/keyboard_force_control.py \
      --step-mm 0.2 \
      --motion-speed 5 \
      --speed-factor 10 \
      --force-print-hz 2

What one key press does:
  - Arrow keys send one RelMovLUser command in the selected user frame.
  - The default step is 0.5 mm per key press.
  - The default motion speed is 10 mm/s.
  - Use --step-mm to make this smaller, e.g. --step-mm 0.2.
  - Use --motion-speed and --speed-factor to slow the robot down further.

Safety notes:
  - Keep an emergency stop reachable.
  - Start with --step-mm 0.2 in open space before testing near fixtures.
  - Press space to send Stop().
  - Press q or Ctrl-C to quit; the script sends Stop() and FCOff() on exit.
  - The force readout only represents a real six-axis F/T sensor if the Dobot
    controller reports an online force sensor. Without one, treat it as a
    controller/firmware value, not a calibrated external contact force.

Controls:
  Arrow Up/Down    : +/- X in the selected user coordinate frame
  Arrow Left/Right : +/- Y in the selected user coordinate frame
  PageUp/PageDown  : +/- Z
  f                : toggle force-compliance mode
  h                : zero the six-axis force sensor
  g                : print current force sensor reading
  p                : pause/resume continuous force printing
  space            : stop robot motion
  q or Ctrl-C      : quit

The force-compliance mode uses FCForceMode with zero target force. Dobot's SDK
documents zero target force as a compliant mode similar to force drag.
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
SDK_ROOT = REPO_ROOT / "third_party" / "dobot_umi_sdk"
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)
for path in (SDK_ROOT / "dobot_sdk", SDK_ROOT / "adaptive_sdk"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from dobot_api import DobotApiDashboard  # type: ignore  # noqa: E402
from examples.dobot_umi import constants  # noqa: E402


def parse_dobot_response(resp: str | bytes | None) -> tuple[int | None, list[float]]:
    """Parse standard Dobot response, e.g. '0,{...},Command();'."""
    if resp is None:
        return None, []
    if isinstance(resp, bytes):
        resp = resp.decode("utf-8", errors="replace")
    text = resp.strip()
    if not text:
        return None, []

    code: int | None = None
    try:
        code = int(text.split(",", 1)[0])
    except Exception:
        code = None

    values: list[float] = []
    if "{" in text and "}" in text:
        body = text[text.find("{") + 1 : text.find("}")]
        for item in body.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                values.append(float(item))
            except ValueError:
                pass
    return code, values


def require_ok(resp: str | bytes | None, action: str) -> bool:
    code, _ = parse_dobot_response(resp)
    if code == 0:
        return True
    print(f"[WARN] {action} failed: {resp!r}")
    return False


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

        # Decode common ANSI escape sequences emitted by arrow/page keys.
        ready, _, _ = select.select([sys.stdin], [], [], 0.01)
        if not ready:
            return "\x1b"
        seq = ch + sys.stdin.read(1)
        ready, _, _ = select.select([sys.stdin], [], [], 0.01)
        if ready:
            seq += sys.stdin.read(1)
        ready, _, _ = select.select([sys.stdin], [], [], 0.01)
        if seq in ("\x1b[5", "\x1b[6") and ready:
            seq += sys.stdin.read(1)
        return {
            "\x1b[A": "up",
            "\x1b[B": "down",
            "\x1b[C": "right",
            "\x1b[D": "left",
            "\x1b[5~": "pageup",
            "\x1b[6~": "pagedown",
        }.get(seq, seq)


class DobotKeyboardForceControl:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dashboard = DobotApiDashboard(args.dobot_ip, args.dashboard_port)
        sock = getattr(self.dashboard, "socket_dobot", None)
        if sock is None or isinstance(sock, int):
            raise RuntimeError("Dashboard socket is not connected")
        sock.getpeername()

        self.force_mode = False
        self.print_force_stream = args.print_force
        self._last_force_print = 0.0
        self.key_handlers: dict[str, Callable[[], None]] = {
            "up": lambda: self.jog(dx=self.args.step_mm),
            "down": lambda: self.jog(dx=-self.args.step_mm),
            "left": lambda: self.jog(dy=self.args.step_mm),
            "right": lambda: self.jog(dy=-self.args.step_mm),
            "pageup": lambda: self.jog(dz=self.args.step_mm),
            "pagedown": lambda: self.jog(dz=-self.args.step_mm),
            " ": self.stop,
            "f": self.toggle_force_mode,
            "h": self.zero_force_sensor,
            "g": self.print_force,
            "p": self.toggle_force_stream,
        }

    def close(self) -> None:
        try:
            if self.force_mode:
                self.force_off()
        finally:
            try:
                self.dashboard.close()
            except Exception:
                pass

    def initialize_robot(self) -> None:
        print(f"Connecting Dobot Dashboard {self.args.dobot_ip}:{self.args.dashboard_port}")
        if self.args.clear_error:
            require_ok(self.dashboard.ClearError(), "ClearError")
            time.sleep(0.2)
        if self.args.enable_robot:
            require_ok(self.dashboard.EnableRobot(), "EnableRobot")
            time.sleep(0.5)
        if self.args.speed_factor > 0:
            require_ok(self.dashboard.SpeedFactor(self.args.speed_factor), "SpeedFactor")

        mode_resp = self.dashboard.RobotMode()
        pose_resp = self.dashboard.GetPose()
        print(f"RobotMode: {mode_resp.strip() if mode_resp else '(empty)'}")
        print(f"GetPose:   {pose_resp.strip() if pose_resp else '(empty)'}")

    def jog(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        resp = self.dashboard.RelMovLUser(
            dx,
            dy,
            dz,
            0.0,
            0.0,
            0.0,
            user=self.args.user,
            tool=self.args.tool,
            speed=self.args.motion_speed,
        )
        label = f"RelMovLUser dx={dx:+.1f} dy={dy:+.1f} dz={dz:+.1f}"
        if require_ok(resp, label) and not self.args.quiet:
            print(f"{label} | force_mode={self.force_mode}")

    def stop(self) -> None:
        require_ok(self.dashboard.Stop(), "Stop")
        print("Stop")

    def zero_force_sensor(self) -> None:
        require_ok(self.dashboard.EnableFTSensor(1), "EnableFTSensor")
        time.sleep(0.2)
        require_ok(self.dashboard.SixForceHome(), "SixForceHome")
        print("Six-axis force sensor zeroed")

    def print_force(self) -> None:
        resp = self.dashboard.GetForce(self.args.tool if self.args.tool >= 0 else -1)
        _, values = parse_dobot_response(resp)
        print(self.format_force_line(values, resp))

    def format_force_line(self, values: list[float], raw_resp: str | bytes | None) -> str:
        stamp = time.strftime("%H:%M:%S")
        if len(values) >= 6:
            fx, fy, fz, tx, ty, tz = values[:6]
            return (
                f"[{stamp}] Force "
                f"Fx={fx:+8.3f} Fy={fy:+8.3f} Fz={fz:+8.3f} "
                f"Tx={tx:+8.3f} Ty={ty:+8.3f} Tz={tz:+8.3f} "
                f"force_mode={self.force_mode}"
            )
        return f"[{stamp}] Force raw={raw_resp!r} force_mode={self.force_mode}"

    def maybe_print_force_stream(self) -> None:
        if not self.print_force_stream or self.args.force_print_hz <= 0:
            return
        now = time.monotonic()
        period = 1.0 / self.args.force_print_hz
        if now - self._last_force_print < period:
            return
        self._last_force_print = now
        self.print_force()

    def toggle_force_stream(self) -> None:
        self.print_force_stream = not self.print_force_stream
        state = "ON" if self.print_force_stream else "OFF"
        print(f"Continuous force printing {state}")

    def force_on(self) -> None:
        mask = self.args.force_mask
        target = self.args.target_force

        require_ok(self.dashboard.EnableFTSensor(1), "EnableFTSensor")
        time.sleep(0.2)
        if self.args.auto_zero_force:
            require_ok(self.dashboard.SixForceHome(), "SixForceHome")
            time.sleep(0.2)

        require_ok(
            self.dashboard.FCSetForceLimit(
                *self.args.force_limit,
            ),
            "FCSetForceLimit",
        )
        require_ok(self.dashboard.FCSetMass(*self.args.force_mass), "FCSetMass")
        require_ok(self.dashboard.FCSetStiffness(*self.args.force_stiffness), "FCSetStiffness")
        require_ok(self.dashboard.FCSetDamping(*self.args.force_damping), "FCSetDamping")
        require_ok(
            self.dashboard.FCSetForceSpeedLimit(*self.args.force_speed_limit),
            "FCSetForceSpeedLimit",
        )
        require_ok(
            self.dashboard.FCSetDeviation(*self.args.force_deviation, self.args.force_deviation_action),
            "FCSetDeviation",
        )
        ok = require_ok(
            self.dashboard.FCForceMode(
                mask[0],
                mask[1],
                mask[2],
                mask[3],
                mask[4],
                mask[5],
                target[0],
                target[1],
                target[2],
                target[3],
                target[4],
                target[5],
                reference=self.args.force_reference,
                user=self.args.user,
                tool=self.args.tool,
            ),
            "FCForceMode",
        )
        self.force_mode = ok
        if ok:
            print(
                "Force-compliance ON "
                f"mask={mask}, target_force={target}, reference={self.args.force_reference}"
            )

    def force_off(self) -> None:
        require_ok(self.dashboard.FCOff(), "FCOff")
        self.force_mode = False
        print("Force-compliance OFF")

    def toggle_force_mode(self) -> None:
        if self.force_mode:
            self.force_off()
        else:
            self.force_on()

    def run(self) -> None:
        print_controls()
        with RawKeyboard() as keyboard:
            while True:
                self.maybe_print_force_stream()
                timeout_s = min(0.1, 1.0 / max(self.args.force_print_hz, 1.0))
                key = keyboard.read_key(timeout_s=timeout_s)
                if key is None:
                    continue
                if key in ("q", "Q"):
                    break
                handler = self.key_handlers.get(key)
                if handler is not None:
                    handler()


def csv_ints(text: str, count: int, name: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != count:
        raise argparse.ArgumentTypeError(f"{name} must contain {count} comma-separated ints")
    return values


def print_controls() -> None:
    print(
        "\nControls:\n"
        "  arrows: +/- X/Y relative linear jog in user frame\n"
        "  PageUp/PageDown: +/- Z jog\n"
        "  f: toggle zero-force compliance mode\n"
        "  h: zero six-axis force sensor\n"
        "  g: print force sensor value\n"
        "  p: pause/resume continuous force printing\n"
        "  space: stop motion\n"
        "  q: quit\n"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dobot keyboard Cartesian jog with FCForceMode compliance test."
    )
    parser.add_argument("--dobot-ip", default=constants.DOBOT_IP)
    parser.add_argument("--dashboard-port", type=int, default=constants.DOBOT_DASHBOARD_PORT)
    parser.add_argument("--step-mm", type=float, default=0.5, help="Cartesian jog step in mm.")
    parser.add_argument("--motion-speed", type=int, default=10, help="RelMovLUser target speed in mm/s.")
    parser.add_argument("--speed-factor", type=int, default=10, help="Global speed factor percent; <=0 skips.")
    parser.add_argument("--user", type=int, default=0, help="Dobot user coordinate index.")
    parser.add_argument("--tool", type=int, default=0, help="Dobot tool coordinate index.")
    parser.add_argument("--enable-robot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clear-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-zero-force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--print-force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continuously print GetForce() readings while the keyboard loop is running.",
    )
    parser.add_argument(
        "--force-print-hz",
        type=float,
        default=5.0,
        help="Continuous force print refresh rate. Set <=0 to disable.",
    )

    parser.add_argument(
        "--force-mask",
        type=lambda s: csv_ints(s, 6, "force-mask"),
        default=[1, 1, 1, 0, 0, 0],
        help="FCForceMode axis enable mask x,y,z,rx,ry,rz. Default opens XYZ only.",
    )
    parser.add_argument(
        "--target-force",
        type=lambda s: csv_ints(s, 6, "target-force"),
        default=[0, 0, 0, 0, 0, 0],
        help="Target force/torque x,y,z,rx,ry,rz. Zero means compliance.",
    )
    parser.add_argument(
        "--force-reference",
        type=int,
        default=1,
        choices=(0, 1),
        help="0: tool frame, 1: user frame.",
    )
    parser.add_argument(
        "--force-limit",
        type=lambda s: csv_ints(s, 6, "force-limit"),
        default=[40, 40, 40, 8, 8, 8],
        help="Max force/torque limit for FC mode.",
    )
    parser.add_argument(
        "--force-mass",
        type=lambda s: csv_ints(s, 6, "force-mass"),
        default=[20, 20, 20, 10, 10, 10],
    )
    parser.add_argument(
        "--force-stiffness",
        type=lambda s: csv_ints(s, 6, "force-stiffness"),
        default=[0, 0, 0, 0, 0, 0],
        help="Zero stiffness gives softer free compliance on enabled axes.",
    )
    parser.add_argument(
        "--force-damping",
        type=lambda s: csv_ints(s, 6, "force-damping"),
        default=[60, 60, 60, 60, 60, 60],
    )
    parser.add_argument(
        "--force-speed-limit",
        type=lambda s: csv_ints(s, 6, "force-speed-limit"),
        default=[20, 20, 20, 5, 5, 5],
    )
    parser.add_argument(
        "--force-deviation",
        type=lambda s: csv_ints(s, 6, "force-deviation"),
        default=[100, 100, 100, 30, 30, 30],
        help="Allowed force-control position/orientation deviation.",
    )
    parser.add_argument(
        "--force-deviation-action",
        type=int,
        default=0,
        choices=(0, 1),
        help="0: alarm on deviation, 1: stop searching and continue original trajectory.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    app: DobotKeyboardForceControl | None = None
    try:
        app = DobotKeyboardForceControl(args)
        app.initialize_robot()
        app.run()
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    finally:
        if app is not None:
            try:
                app.stop()
            except Exception:
                pass
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
