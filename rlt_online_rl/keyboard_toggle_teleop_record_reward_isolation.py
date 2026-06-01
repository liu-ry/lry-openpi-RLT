#!/usr/bin/env python3
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from train_deploy_alignment.manual_signal_bridge import ENTER_CRITICAL_PHASE_SERVICE
from train_deploy_alignment.manual_signal_bridge import RECORD_FAILURE_SERVICE
from train_deploy_alignment.manual_signal_bridge import RECORD_SUCCESS_SERVICE
from train_deploy_alignment.manual_signal_bridge import REQUEST_NEXT_EPISODE_SERVICE

RL_TELEOP_TRIGGER_SERVICE_DEFAULT = "/umi/teleop_trigger"
HW_TELEOP_TRIGGER_SERVICE_DEFAULT = "/umi/teleop_trigger"
TELEOP_STATUS_SERVICE = "/teleop_status"
HW_TELEOP_SETTLE_SEC = 1.0


def _parse_cli_args():
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--rl_service", default=RL_TELEOP_TRIGGER_SERVICE_DEFAULT,
                   help="RL 侧 teleop toggle 服务名（Pika 默认 /teleop_trigger_rl；"
                        "Dobot+UMI 用 /umi/teleop_trigger）")
    p.add_argument("--hw_service", default=HW_TELEOP_TRIGGER_SERVICE_DEFAULT,
                   help="硬件侧 teleop toggle 服务名（Pika 默认 /teleop_trigger；"
                        "Dobot+UMI 用 /umi/teleop_trigger）")
    p.add_argument("--domain_id", type=int, default=9,
                   help="ROS_DOMAIN_ID（UMI 默认 9）")
    args, _ = p.parse_known_args()
    return args


def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


class KeyboardTeleopRecordRewardToggle(Node):
    def __init__(self, rl_service: str, hw_service: str):
        super().__init__("keyboard_teleop_record_reward_toggle")

        self.rl_teleop_cli = self.create_client(Trigger, rl_service)
        self.hw_teleop_cli = self.create_client(Trigger, hw_service)
        self.teleop_status_cli = self.create_client(Trigger, TELEOP_STATUS_SERVICE)
        self.next_episode_cli = self.create_client(Trigger, REQUEST_NEXT_EPISODE_SERVICE)
        self.success_cli = self.create_client(Trigger, RECORD_SUCCESS_SERVICE)
        self.failure_cli = self.create_client(Trigger, RECORD_FAILURE_SERVICE)
        self.critical_phase_cli = self.create_client(Trigger, ENTER_CRITICAL_PHASE_SERVICE)
        self.control_mode = "unknown"

        self.get_logger().info(f"Waiting for local teleop service {rl_service}...")
        self.rl_teleop_cli.wait_for_service()
        self.get_logger().info(f"Waiting for hardware teleop service {hw_service}...")
        self.hw_teleop_cli.wait_for_service()
        self.get_logger().info(f"Waiting for {TELEOP_STATUS_SERVICE} service...")
        self.teleop_status_cli.wait_for_service()

        self.get_logger().info("Waiting for next-episode / manual signal services...")
        self.next_episode_cli.wait_for_service()

        self.success_cli.wait_for_service()
        self.failure_cli.wait_for_service()
        self.critical_phase_cli.wait_for_service()

        self.refresh_teleop_mode(retries=5, timeout_sec=1.5)
        self.get_logger().info(self._ready_message())

    def _ready_message(self) -> str:
        return (
            "Ready. Press 't' to toggle teleop. Press 'o' to start the next episode. "
            "Press 's' to end the episode with success. Press 'f' to end the episode with failure. "
            "Press 'c' to enter the critical phase. Press 'q' to quit."
        )

    @staticmethod
    def _parse_mode_message(message: str | None):
        text = (message or "").strip().lower()
        if "mode=teleop" in text:
            return "teleop"
        if "mode=policy" in text:
            return "policy"
        if "mode=reset" in text:
            return "reset"
        return None

    def log_teleop_mode(self):
        self.get_logger().info(f"Current control mode: {self.control_mode}")

    def refresh_teleop_mode(self, *, retries: int = 3, timeout_sec: float = 1.0):
        total_attempts = max(int(retries), 1)
        for attempt in range(1, total_attempts + 1):
            req = Trigger.Request()
            future = self.teleop_status_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
            if future.result() is None:
                if attempt < total_attempts:
                    self.get_logger().warn(
                        f"Failed to query teleop status (attempt {attempt}/{total_attempts}); retrying."
                    )
                    continue
                self.control_mode = "unknown"
                self.get_logger().error("Failed to query teleop status.")
                return False

            resp = future.result()
            parsed = self._parse_mode_message(resp.message)
            if parsed is None:
                self.control_mode = "unknown"
                self.get_logger().error(f"Unexpected teleop status response: {resp.message!r}")
                return False

            self.control_mode = parsed
            self.log_teleop_mode()
            return True
        return False

    def _call_trigger(self, client, failure_message: str, *, timeout_sec: float = 5.0):
        req = Trigger.Request()
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)

        if future.result() is None:
            self.get_logger().error(failure_message)
            return None
        return future.result()

    def _toggle_local_teleop(self) -> bool:
        resp = self._call_trigger(self.rl_teleop_cli, "Failed to toggle local teleop state.")
        if resp is None:
            return False
        if not resp.success:
            self.get_logger().warn(resp.message if resp.message else "Local teleop toggle failed.")
            self.refresh_teleop_mode()
            return False

        parsed = self._parse_mode_message(resp.message)
        if parsed is None:
            self.get_logger().warn(f"Could not parse teleop mode from response: {resp.message!r}")
            self.refresh_teleop_mode()
        else:
            self.control_mode = parsed
            self.log_teleop_mode()
        self.get_logger().info(resp.message if resp.message else "Local teleop toggle succeeded.")
        return True

    def _toggle_hardware_teleop(self, *, reason: str) -> bool:
        resp = self._call_trigger(
            self.hw_teleop_cli,
            f"Failed to toggle hardware teleop for {reason}.",
        )
        if resp is None:
            return False
        message = resp.message if resp.message else f"Hardware teleop toggled for {reason}."
        self.get_logger().info(message)
        return True

    def _record_terminal(self, client, label: str) -> None:
        if not self.refresh_teleop_mode():
            return
        if self.control_mode == "reset":
            self.get_logger().warn(f"Cannot record {label}: episode inactive/reset in progress.")
            return
        if self.control_mode == "teleop":
            if not self._toggle_hardware_teleop(reason=f"{label} end"):
                return
            time.sleep(HW_TELEOP_SETTLE_SEC)

        resp = self._call_trigger(client, f"Failed to record {label}.")
        if resp is None:
            return
        if resp.success:
            self.get_logger().info(resp.message if resp.message else f"Recorded {label}.")
        else:
            self.get_logger().warn(resp.message if resp.message else f"Recording {label} failed.")
        self.refresh_teleop_mode()

    def _is_unified_service(self) -> bool:
        """rl_service 和 hw_service 是同一个服务（如 Dobot+UMI 的 /umi/teleop_trigger）。"""
        return self.rl_teleop_cli.srv_name == self.hw_teleop_cli.srv_name

    def toggle_teleop(self):
        if not self.refresh_teleop_mode():
            return
        if self.control_mode == "reset":
            self.get_logger().warn("Episode inactive/reset in progress; teleop toggle ignored.")
            return

        if self._is_unified_service():
            # 单服务模式（Dobot+UMI）：只调一次，服务内部同时处理 RL 暂停 + 对齐参考帧
            if not self._toggle_local_teleop():
                return
        elif self.control_mode == "teleop":
            if not self._toggle_hardware_teleop(reason="teleop exit"):
                return
            time.sleep(HW_TELEOP_SETTLE_SEC)
            if not self._toggle_local_teleop():
                return
        else:
            if not self._toggle_local_teleop():
                return
            if not self._toggle_hardware_teleop(reason="teleop entry"):
                return
        self.refresh_teleop_mode()

    def request_next_episode(self):
        resp = self._call_trigger(self.next_episode_cli, "Failed to request next episode start.")
        if resp is None:
            return

        if resp.success:
            self.get_logger().info(resp.message if resp.message else "Requested next episode start.")
            return
        self.get_logger().warn(resp.message if resp.message else "Next episode request failed.")

    def record_success(self):
        self._record_terminal(self.success_cli, "success")

    def record_failure(self):
        self._record_terminal(self.failure_cli, "failure")

    def enter_critical_phase(self):
        resp = self._call_trigger(self.critical_phase_cli, "Failed to enter the critical phase.")
        if resp is None:
            return
        if resp.success:
            self.get_logger().info(resp.message if resp.message else "Entered the critical phase.")
            return
        self.get_logger().warn(resp.message if resp.message else "Entering the critical phase failed.")


def main():
    cli = _parse_cli_args()
    rclpy.init(domain_id=cli.domain_id)
    node = KeyboardTeleopRecordRewardToggle(
        rl_service=cli.rl_service,
        hw_service=cli.hw_service,
    )

    try:
        while rclpy.ok():
            ch = getch()
            if ch in ("\x03", "\x04", "q"):  # Ctrl+C / Ctrl+D / q 均退出
                break
            elif ch == "t":
                node.toggle_teleop()
            elif ch == "o":
                node.request_next_episode()
            elif ch == "s":
                node.record_success()
            elif ch == "f":
                node.record_failure()
            elif ch == "c":
                node.enter_critical_phase()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
