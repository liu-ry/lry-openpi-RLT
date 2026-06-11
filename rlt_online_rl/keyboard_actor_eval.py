#!/usr/bin/env python3
from keyboard_toggle_teleop_record_reward_isolation import KeyboardTeleopRecordRewardToggle
from keyboard_toggle_teleop_record_reward_isolation import _parse_cli_args
from keyboard_toggle_teleop_record_reward_isolation import getch
import rclpy
from std_srvs.srv import Trigger
from train_deploy_alignment.manual_signal_bridge import RECORD_DONE_SERVICE
from train_deploy_alignment.manual_signal_bridge import SET_CRITICAL_POLICY_ACTOR_SERVICE
from train_deploy_alignment.manual_signal_bridge import SET_CRITICAL_POLICY_BASE_SERVICE
from train_deploy_alignment.manual_signal_bridge import TOGGLE_CRITICAL_PHASE_SERVICE


'''Keyboard Actor Eval Node
- 允许用户通过键盘输入控制评测流程，专注于 Actor 策略的评测。
- 功能：
  - 'a'：切换到 Actor 策略模式（Machine A）。
  - 'b'：切换到基础策略模式（Machine B）。
  - 'o'：开始下一轮评测。
  - 's'：结束或重置当前评测。
  - 'c'：切换关键阶段的开关。
  - 't'：切换人工操作模式。
  - 'q'：退出评测。
'''

class KeyboardActorEval(KeyboardTeleopRecordRewardToggle):
    def _ready_message(self) -> str:
        return (
            "Eval ready. Press 'a' for actor refine, 'b' for Machine A only, 'o' to start the next episode, "
            "'c' to toggle critical on/off in full_task, 's' to end/reset the episode, "
            "'t' to toggle teleop, 'q' to quit."
        )

    def __init__(self, rl_service: str, hw_service: str):
        super().__init__(rl_service=rl_service, hw_service=hw_service)
        self.done_cli = self.create_client(Trigger, RECORD_DONE_SERVICE)
        self.toggle_critical_cli = self.create_client(Trigger, TOGGLE_CRITICAL_PHASE_SERVICE)
        self.select_actor_cli = self.create_client(Trigger, SET_CRITICAL_POLICY_ACTOR_SERVICE)
        self.select_base_cli = self.create_client(Trigger, SET_CRITICAL_POLICY_BASE_SERVICE)

        self.done_cli.wait_for_service()
        self.toggle_critical_cli.wait_for_service()
        self.select_actor_cli.wait_for_service()
        self.select_base_cli.wait_for_service()

    def reset_episode(self):
        self._record_terminal(self.done_cli, "done")

    def select_actor(self):
        resp = self._call_trigger(self.select_actor_cli, "Failed to select actor critical policy mode.")
        if resp is None:
            return
        self.get_logger().info(resp.message if resp.message else "Selected critical policy mode=actor.")

    def select_base(self):
        resp = self._call_trigger(self.select_base_cli, "Failed to select base critical policy mode.")
        if resp is None:
            return
        self.get_logger().info(resp.message if resp.message else "Selected critical policy mode=base.")

    def toggle_critical_phase(self):
        resp = self._call_trigger(self.toggle_critical_cli, "Failed to toggle the critical phase.")
        if resp is None:
            return
        self.get_logger().info(resp.message if resp.message else "Toggled critical phase.")


def main():
    cli = _parse_cli_args()
    rclpy.init(domain_id=cli.domain_id)
    node = KeyboardActorEval(
        rl_service=cli.rl_service,
        hw_service=cli.hw_service,
    )

    try:
        while rclpy.ok():
            ch = getch()
            if ch in ("\x03", "\x04", "q"):
                break
            elif ch == "t":
                node.toggle_teleop()
            elif ch == "o":
                node.request_next_episode()
            elif ch == "s":
                node.reset_episode()
            elif ch == "c":
                node.toggle_critical_phase()
            elif ch == "a":
                node.select_actor()
            elif ch == "b":
                node.select_base()
            elif ch in ("\r", "\n"):
                node.approve_online_transition()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
