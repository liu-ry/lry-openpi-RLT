"""check_hardware.py — 插上所有硬件后的连通性检测脚本。

检测项目：
  1. 越疆 Dobot 机械臂（TCP/IP 连接 + 读取关节角）—— 不执行任何运动
  2. 知行夹爪（RS-485 串口初始化 + 读取当前位置 + 可选开合测试）
  3. RealSense 正面相机（ROS 话题是否有图像帧到达）
  4. RealSense 腕部相机（同上）
  5. UMI 示教设备（ROS 话题是否有动作帧到达，可不接）

用法::

    # 仅连通性检测（不动夹爪）
    python examples/dobot_umi/check_hardware.py

    # 同时做夹爪开合测试
    python examples/dobot_umi/check_hardware.py --gripper_test

    # 跳过 UMI 检测（UMI 设备未连接时）
    python examples/dobot_umi/check_hardware.py --skip_umi

    # 修改超时时间（等待 ROS 话题，默认 8s）
    python examples/dobot_umi/check_hardware.py --ros_timeout 15
"""
# ruff: noqa
from __future__ import annotations

import argparse
import sys
import time
import threading
from pathlib import Path

# ── 路径注入 ──────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SDK_ROOT  = _REPO_ROOT / "third_party" / "dobot_umi_sdk"
for _p in (
    str(_REPO_ROOT),
    str(_SDK_ROOT),
    str(_SDK_ROOT / "dobot_sdk"),
    str(_SDK_ROOT / "adaptive_sdk"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from examples.dobot_umi import constants


# ─────────────────────────────────────────────────────────────────────────────
# ANSI 颜色
# ─────────────────────────────────────────────────────────────────────────────
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def _ok(msg: str)   -> str: return f"{_GREEN}✓  {msg}{_RESET}"
def _fail(msg: str) -> str: return f"{_RED}✗  {msg}{_RESET}"
def _warn(msg: str) -> str: return f"{_YELLOW}⚠  {msg}{_RESET}"
def _title(msg: str)-> str: return f"\n{_BOLD}{msg}{_RESET}"


# ─────────────────────────────────────────────────────────────────────────────
# 检测 1：Dobot 机械臂 TCP/IP 连接（只读，不动）
# ─────────────────────────────────────────────────────────────────────────────

def check_dobot_arm(
    ip: str,
    dashboard_port: int,
    feedback_port: int,
) -> bool:
    print(_title("[1] 越疆 Dobot 机械臂"))
    try:
        from dobot_sdk.dobot_api import DobotApiDashboard, DobotApiFeedBack
    except ImportError:
        print(_fail("Dobot SDK 导入失败，请确认 third_party/dobot_umi_sdk/ 存在"))
        return False

    # ── Dashboard 连接 ────────────────────────────────────────────────────────
    dashboard = None
    try:
        print(f"   连接 Dashboard {ip}:{dashboard_port} ...")
        dashboard = DobotApiDashboard(ip, dashboard_port)
        sock = getattr(dashboard, "socket_dobot", None)
        if sock is None or isinstance(sock, int):
            print(_fail(f"Dashboard 连接失败（socket 未就绪）"))
            return False
        try:
            sock.getpeername()
        except Exception:
            print(_fail("Dashboard TCP 连接未建立"))
            return False
        print(_ok(f"Dashboard 连接成功 {ip}:{dashboard_port}"))
    except Exception as e:
        print(_fail(f"Dashboard 连接异常: {e}"))
        return False

    # ── 读取机器人模式 ─────────────────────────────────────────────────────────
    try:
        resp = dashboard.RobotMode()
        mode = resp.strip().split(",")[0] if resp else "?"
        mode_desc = {
            "1": "初始化", "2": "拖动", "3": "运行中", "4": "录制",
            "5": "空闲", "6": "暂停", "7": "JOG", "8": "Home",
            "9": "报警", "10": "校准", "11": "保留",
        }.get(mode, "未知")
        print(_ok(f"机器人模式: {mode} ({mode_desc})"))
        if mode == "9":
            print(_warn("机械臂当前处于报警状态，建议先手动 ClearError"))
    except Exception as e:
        print(_warn(f"RobotMode 查询失败: {e}"))

    # ── 读取当前关节角（GetAngle，°） ─────────────────────────────────────────
    try:
        resp = dashboard.GetAngle()
        print(_ok(f"GetAngle 响应: {resp.strip() if resp else '(空)'}"))
    except Exception as e:
        print(_warn(f"GetAngle 查询失败: {e}"))

    # ── FeedBack 端口 ─────────────────────────────────────────────────────────
    fb = None
    try:
        print(f"   连接 FeedBack {ip}:{feedback_port} ...")
        fb = DobotApiFeedBack(ip, feedback_port)
        sock_fb = getattr(fb, "socket_dobot", None)
        if sock_fb and not isinstance(sock_fb, int):
            sock_fb.getpeername()
            print(_ok(f"FeedBack 连接成功 {ip}:{feedback_port}"))
            # 尝试读一帧反馈数据
            data = fb.feedBackData()
            if data and hex(data["TestValue"][0]) == "0x123456789abcdef":
                q = list(data["QActual"][0])
                import math
                q_deg = [round(math.degrees(v), 2) for v in q]
                print(_ok(f"FeedBack 关节角（°）: {q_deg}"))
            else:
                print(_warn("FeedBack 数据校验未通过（TestValue 不匹配），可能固件版本差异"))
        else:
            print(_warn("FeedBack 连接未就绪（将回退到 Dashboard GetAngle 查询）"))
    except Exception as e:
        print(_warn(f"FeedBack 连接异常: {e}（将回退到 GetAngle）"))
    finally:
        if fb is not None:
            try:
                del fb
            except Exception:
                pass
        if dashboard is not None:
            try:
                dashboard.close()
            except Exception:
                pass

    print(_ok("Dobot 机械臂检测通过（未执行任何运动指令）"))
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 检测 2：知行夹爪 RS-485
# ─────────────────────────────────────────────────────────────────────────────

def check_gripper(
    port: str,
    slave_id: int,
    baudrate: int,
    speed_pct: int,
    force_pct: int,
    do_motion_test: bool = False,
) -> bool:
    print(_title("[2] 知行夹爪（RS-485）"))
    try:
        from adaptive_sdk.changingtek_p_rtu_Servo import MotorController
    except ImportError:
        print(_fail("MotorController SDK 导入失败，请确认 third_party/dobot_umi_sdk/adaptive_sdk/ 存在"))
        return False

    motor = None
    try:
        print(f"   打开串口 {port}（波特率 {baudrate}）...")
        motor = MotorController(port, slave_id, baudrate, 0.5)
        motor.set_target_speed(speed_pct)
        motor.set_target_force(force_pct)
        motor.set_target_acceleration(2000)
        motor.set_target_deceleration(2000)
        print(_ok(f"串口 {port} 初始化成功"))
    except Exception as e:
        print(_fail(f"夹爪初始化失败: {e}"))
        return False

    # ── 读取当前位置 ──────────────────────────────────────────────────────────
    try:
        pos = motor.read_real_position()
        ratio = pos / max(constants.GRIPPER_POS_CLOSE, 1)
        dist_m = (1.0 - ratio) * constants.GRIPPER_OPEN_M
        print(_ok(f"当前编码器位置: {pos}  →  开口距离: {dist_m*1000:.1f} mm"))
    except Exception as e:
        print(_warn(f"读取夹爪位置失败: {e}"))

    # ── 可选：开合运动测试 ────────────────────────────────────────────────────
    if do_motion_test:
        print("   开合测试：打开夹爪...")
        try:
            motor.set_target_position(constants.GRIPPER_POS_OPEN)
            motor.trigger_motion()
            time.sleep(2.0)
            pos_open = motor.read_real_position()
            print(_ok(f"张开完成，编码器位置: {pos_open}"))

            print("   开合测试：关闭夹爪...")
            motor.set_target_position(constants.GRIPPER_POS_CLOSE)
            motor.trigger_motion()
            time.sleep(2.0)
            pos_close = motor.read_real_position()
            print(_ok(f"闭合完成，编码器位置: {pos_close}"))

            print("   开合测试：恢复打开（复位）...")
            motor.set_target_position(constants.GRIPPER_POS_OPEN)
            motor.trigger_motion()
            time.sleep(1.5)
            print(_ok("夹爪开合测试通过"))
        except Exception as e:
            print(_fail(f"夹爪开合测试失败: {e}"))
            return False
    else:
        print(_warn("跳过开合运动测试（添加 --gripper_test 可启用）"))

    print(_ok("知行夹爪检测通过"))
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 检测 3 & 4：RealSense 相机（ROS 话题）
# ─────────────────────────────────────────────────────────────────────────────

def check_cameras(
    front_topic: str,
    wrist_topic: str,
    timeout_s: float = 8.0,
) -> bool:
    print(_title("[3] RealSense 相机（ROS 话题）"))
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image as ROSImage
    except ImportError:
        print(_fail("rclpy 未安装，无法检测相机话题"))
        return False

    received = {"front": None, "wrist": None}
    lock = threading.Lock()

    class _CamChecker(Node):
        def __init__(self):
            super().__init__("hw_check_cam")
            qos = qos_profile_sensor_data
            self.create_subscription(ROSImage, front_topic, self._on_front, qos)
            self.create_subscription(ROSImage, wrist_topic, self._on_wrist, qos)

        def _on_front(self, msg):
            with lock:
                if received["front"] is None:
                    received["front"] = msg

        def _on_wrist(self, msg):
            with lock:
                if received["wrist"] is None:
                    received["wrist"] = msg

    if not rclpy.ok():
        rclpy.init()

    node = _CamChecker()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    print(f"   等待相机话题（最多 {timeout_s:.0f}s）...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        executor.spin_once(timeout_sec=0.1)
        with lock:
            if received["front"] is not None and received["wrist"] is not None:
                break

    ok = True
    with lock:
        for key, topic in (("front", front_topic), ("wrist", wrist_topic)):
            msg = received[key]
            if msg is not None:
                h = msg.height
                w = msg.width
                enc = msg.encoding
                print(_ok(f"{topic}  →  {w}×{h}  encoding={enc}"))
            else:
                print(_fail(f"{topic}  →  {timeout_s:.0f}s 内未收到图像（检查相机是否启动）"))
                ok = False

    node.destroy_node()
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 检测 5：UMI 示教设备（ROS 话题）
# ─────────────────────────────────────────────────────────────────────────────

def check_umi(
    action_topic: str,
    timeout_s: float = 5.0,
) -> bool:
    print(_title("[4] UMI 示教设备（ROS 话题）"))
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
    except ImportError:
        print(_fail("rclpy 未安装，无法检测 UMI 话题"))
        return False

    received = [None]
    lock = threading.Lock()

    class _UMIChecker(Node):
        def __init__(self):
            super().__init__("hw_check_umi")
            self.create_subscription(JointState, action_topic, self._on_action, 10)

        def _on_action(self, msg):
            with lock:
                if received[0] is None:
                    received[0] = msg

    if not rclpy.ok():
        rclpy.init()

    node = _UMIChecker()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    print(f"   等待 UMI 话题 {action_topic}（最多 {timeout_s:.0f}s）...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        executor.spin_once(timeout_sec=0.1)
        with lock:
            if received[0] is not None:
                break

    ok = False
    with lock:
        msg = received[0]
        if msg is not None:
            import numpy as np
            action = list(msg.position)
            print(_ok(f"{action_topic}  →  7D 动作: {[round(v, 4) for v in action[:7]]}"))
            ok = True
        else:
            print(_warn(f"{action_topic}  →  {timeout_s:.0f}s 内未收到帧（UMI 设备未连接或未发布？）"))

    node.destroy_node()
    return ok  # UMI 不连接时不算强制失败


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="越疆 Dobot + 知行夹爪 + 双 RealSense + UMI 硬件连通性检测"
    )
    parser.add_argument("--dobot_ip",            default=constants.DOBOT_IP)
    parser.add_argument("--dobot_dashboard_port", default=constants.DOBOT_DASHBOARD_PORT, type=int)
    parser.add_argument("--dobot_feedback_port",  default=constants.DOBOT_FEEDBACK_PORT,  type=int)
    parser.add_argument("--gripper_port",         default=constants.GRIPPER_SERIAL_PORT)
    parser.add_argument("--gripper_slave_id",     default=constants.GRIPPER_SLAVE_ID,     type=int)
    parser.add_argument("--gripper_baudrate",     default=constants.GRIPPER_BAUDRATE,     type=int)
    parser.add_argument("--gripper_speed_pct",    default=constants.GRIPPER_SPEED_PCT,    type=int)
    parser.add_argument("--gripper_force_pct",    default=constants.GRIPPER_FORCE_PCT,    type=int)
    parser.add_argument("--gripper_test",  action="store_true",
                        help="执行夹爪开合运动测试（会实际移动夹爪）")
    parser.add_argument("--skip_umi",     action="store_true",
                        help="跳过 UMI 设备检测")
    parser.add_argument("--skip_arm",     action="store_true",
                        help="跳过机械臂检测（仅测其他设备）")
    parser.add_argument("--skip_gripper", action="store_true",
                        help="跳过夹爪检测")
    parser.add_argument("--skip_cameras", action="store_true",
                        help="跳过相机检测")
    parser.add_argument("--ros_timeout",  default=8.0, type=float,
                        help="等待 ROS 话题的超时秒数（默认 8s）")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  越疆 Dobot / 知行夹爪 / RealSense / UMI  硬件连通性检测")
    print(f"{'='*60}")

    results: dict[str, bool | None] = {}

    # 1. 机械臂
    if not args.skip_arm:
        results["arm"] = check_dobot_arm(
            args.dobot_ip,
            args.dobot_dashboard_port,
            args.dobot_feedback_port,
        )
    else:
        print(_title("[1] 越疆 Dobot 机械臂"))
        print(_warn("已跳过（--skip_arm）"))
        results["arm"] = None

    # 2. 夹爪
    if not args.skip_gripper:
        results["gripper"] = check_gripper(
            args.gripper_port,
            args.gripper_slave_id,
            args.gripper_baudrate,
            args.gripper_speed_pct,
            args.gripper_force_pct,
            do_motion_test=args.gripper_test,
        )
    else:
        print(_title("[2] 知行夹爪（RS-485）"))
        print(_warn("已跳过（--skip_gripper）"))
        results["gripper"] = None

    # 3 & 4. 相机（ROS 节点只初始化一次）
    if not args.skip_cameras:
        results["cameras"] = check_cameras(
            constants.CAM_FRONT_TOPIC,
            constants.CAM_WRIST_TOPIC,
            timeout_s=args.ros_timeout,
        )
    else:
        print(_title("[3] RealSense 相机（ROS 话题）"))
        print(_warn("已跳过（--skip_cameras）"))
        results["cameras"] = None

    # 5. UMI
    if not args.skip_umi:
        results["umi"] = check_umi(
            constants.UMI_HUMAN_ACTION_TOPIC,
            timeout_s=args.ros_timeout,
        )
    else:
        print(_title("[4] UMI 示教设备"))
        print(_warn("已跳过（--skip_umi）"))
        results["umi"] = None

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  检测结果汇总")
    print(f"{'='*60}")
    labels = {
        "arm":     "越疆机械臂",
        "gripper": "知行夹爪  ",
        "cameras": "RealSense ",
        "umi":     "UMI 设备  ",
    }
    all_pass = True
    for key, label in labels.items():
        v = results.get(key)
        if v is True:
            print(f"  {label}  {_GREEN}PASS{_RESET}")
        elif v is False:
            print(f"  {label}  {_RED}FAIL{_RESET}")
            all_pass = False
        else:
            print(f"  {label}  {_YELLOW}SKIP{_RESET}")

    print(f"{'='*60}")
    if all_pass:
        print(f"  {_GREEN}{_BOLD}所有检测项通过，硬件就绪！{_RESET}\n")
    else:
        print(f"  {_RED}{_BOLD}存在检测失败项，请检查对应硬件连接。{_RESET}\n")

    # 清理 ROS
    try:
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
