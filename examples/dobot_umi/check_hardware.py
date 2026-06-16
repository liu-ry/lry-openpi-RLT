"""check_hardware.py — 插上所有硬件后的连通性检测脚本。

检测项目：
  1. 越疆 Dobot 机械臂（TCP/IP 连接 + 读取关节角）—— 不执行任何运动
  2. 知行夹爪（RS-485 串口初始化 + 读取当前位置 + 可选开合测试）
  3. RealSense 正面相机（ROS 话题是否有图像帧到达）
  4. RealSense 腕部相机（同上）
  5. VitAI GF225 触觉传感器（pyvitaisdk 连接 + 读取 warped image）
  6. UMI 示教设备（ROS 话题是否有动作帧到达，可不接）

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
import os
import sys
import time
import threading
from pathlib import Path

import numpy as np

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

# 确保 third_party 可作为包被识别（dobot_sdk/__init__.py 内部用了 third_party.dobot_sdk 前缀）
_third_party_init = _REPO_ROOT / "third_party" / "__init__.py"
if not _third_party_init.exists():
    _third_party_init.touch()

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
        from dobot_api import DobotApiDashboard, DobotApiFeedBack  # noqa: PLC0415
    except ImportError:
        print(_fail("Dobot SDK 导入失败，请确认 third_party/dobot_umi_sdk/dobot_sdk/dobot_api.py 存在"))
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
                # QActual 单位已是度，直接显示，不需要 math.degrees() 转换
                q_deg = [round(v, 2) for v in q]
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
        from changingtek_p_rtu_Servo import MotorController  # noqa: PLC0415
    except ImportError:
        print(_fail("MotorController SDK 导入失败，请确认 third_party/dobot_umi_sdk/adaptive_sdk/changingtek_p_rtu_Servo.py 存在"))
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

# ─────────────────────────────────────────────────────────────────────────────
# 检测 3 & 4：RealSense 相机（pyrealsense2 SDK 直驱）
# ─────────────────────────────────────────────────────────────────────────────

def check_cameras(
    serial_front: str,
    serial_wrist: str,
    width: int = constants.REALSENSE_WIDTH,
    height: int = constants.REALSENSE_HEIGHT,
    fps: int = constants.REALSENSE_FPS,
) -> bool:
    print(_title("[3] RealSense 相机（pyrealsense2 SDK）"))
    try:
        import pyrealsense2 as rs
    except ImportError:
        print(_fail("pyrealsense2 未安装，请 pip install pyrealsense2"))
        return False

    # ── 枚举设备 ──────────────────────────────────────────────────────────────
    ctx = rs.context()
    connected = [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]
    names     = {d.get_info(rs.camera_info.serial_number): d.get_info(rs.camera_info.name)
                 for d in ctx.query_devices()}
    print(f"   检测到 {len(connected)} 台 RealSense 设备: {connected}")

    if not connected:
        print(_fail("未检测到任何 RealSense 设备，请检查 USB 连接"))
        return False

    # 解析正面 / 腕部序列号
    front_sn = serial_front if serial_front else connected[0]
    remaining = [s for s in connected if s != front_sn]
    wrist_sn  = serial_wrist if serial_wrist else (remaining[0] if remaining else front_sn)
    single_cam = (front_sn == wrist_sn)

    ok = True
    for label, sn in (("cam_front", front_sn), ("cam_wrist", wrist_sn)):
        if single_cam and label == "cam_wrist":
            print(_warn(f"cam_wrist  →  仅 1 台设备，与 cam_front 共用 (SN={sn})"))
            continue
        if sn not in connected:
            print(_fail(f"{label}  →  序列号 {sn} 未连接"))
            ok = False
            continue

        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(sn)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        try:
            pipeline.start(cfg)
            frames = pipeline.wait_for_frames(timeout_ms=3000)
            color  = frames.get_color_frame()
            if color:
                img = np.asanyarray(color.get_data())
                print(_ok(f"{label}  →  SN={sn}  {names.get(sn,'')}  "
                           f"{img.shape[1]}×{img.shape[0]}  RGB8"))
            else:
                print(_fail(f"{label}  →  SN={sn} 未能获取彩色帧"))
                ok = False
        except Exception as e:
            print(_fail(f"{label}  →  SN={sn} 采集失败: {e}"))
            ok = False
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass

    if ok:
        print(_ok("RealSense 相机检测通过"))
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 检测 5：VitAI GF225 触觉传感器（pyvitaisdk）
# ─────────────────────────────────────────────────────────────────────────────

def check_tactile(
    serial_left: str,
    serial_right: str,
) -> bool:
    print(_title("[4] VitAI GF225 触觉传感器（pyvitaisdk）"))
    if not serial_left or not serial_right:
        print(_fail("触觉传感器序列号未配置，请设置 --tactile_sn_left/--tactile_sn_right"))
        return False

    try:
        from examples.dobot_umi.robot_utils import VitAITactileCamera  # noqa: PLC0415
    except ImportError as e:
        print(_fail(f"VitAITactileCamera 导入失败: {e}"))
        return False

    tactile = VitAITactileCamera(serial_left=serial_left, serial_right=serial_right)
    try:
        print(f"   连接 GF225 left={serial_left}, right={serial_right} ...")
        if not tactile.connect():
            print(_fail("触觉传感器连接或校准失败"))
            return False
        images = tactile.get_images()
        ok = True
        for key in (constants.IMAGE_KEY_TACTILE_LEFT, constants.IMAGE_KEY_TACTILE_RIGHT):
            img = images.get(key)
            if img is None:
                print(_fail(f"{key}  →  未返回图像"))
                ok = False
                continue
            if img.ndim != 3 or img.shape[2] != 3:
                print(_fail(f"{key}  →  图像形状异常: {img.shape}，期望 H×W×3"))
                ok = False
                continue
            if img.dtype != np.uint8:
                print(_fail(f"{key}  →  图像 dtype 异常: {img.dtype}，期望 uint8"))
                ok = False
                continue
            print(_ok(f"{key}  →  {img.shape[1]}×{img.shape[0]} RGB uint8"))
        if ok:
            print(_ok("VitAI GF225 触觉传感器检测通过"))
        return ok
    except Exception as e:
        print(_fail(f"触觉传感器采集异常: {e}"))
        return False
    finally:
        try:
            tactile.disconnect()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 检测 6：UMI 示教设备（ROS 话题）
# ─────────────────────────────────────────────────────────────────────────────

def check_umi(
    action_topic: str,
    timeout_s: float = 5.0,
    domain_id: int = 9,
) -> bool:
    print(_title("[5] UMI 示教设备（ROS 话题）"))
    try:
        import rclpy
        import rclpy.context
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped as ROSPoseStamped
    except ImportError:
        print(_fail("rclpy 未安装，无法检测 UMI 话题"))
        return False

    received = [None]
    lock = threading.Lock()

    # UMI 使用独立 Context + domain_id=9，与相机检测的 Context 互不干扰
    umi_context = rclpy.context.Context()
    rclpy.init(context=umi_context, domain_id=domain_id)

    class _UMIChecker(Node):
        def __init__(self):
            super().__init__("hw_check_umi", context=umi_context)
            self.create_subscription(ROSPoseStamped, action_topic, self._on_pose, 50)

        def _on_pose(self, msg):
            with lock:
                if received[0] is None:
                    received[0] = msg

    node = _UMIChecker()
    executor = rclpy.executors.SingleThreadedExecutor(context=umi_context)
    executor.add_node(node)

    print(f"   等待 UMI 话题 {action_topic}（ROS_DOMAIN_ID={domain_id}，最多 {timeout_s:.0f}s）...")
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
            p = msg.pose.position
            q = msg.pose.orientation
            print(_ok(
                f"{action_topic}  →  "
                f"pos=[{p.x:.4f}, {p.y:.4f}, {p.z:.4f}]  "
                f"quat=[{q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f}]"
            ))
            ok = True
        else:
            print(_warn(f"{action_topic}  →  {timeout_s:.0f}s 内未收到帧（UMI 设备未连接或未发布？）"))

    node.destroy_node()
    umi_context.try_shutdown()
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
    parser.add_argument("--skip_tactile", action="store_true",
                        help="跳过 VitAI GF225 触觉传感器检测")
    parser.add_argument("--ros_timeout",  default=8.0, type=float,
                        help="等待 ROS 话题的超时秒数（默认 8s）")
    parser.add_argument("--umi_domain_id", default=9, type=int,
                        help="UMI ROS 消息的 DOMAIN_ID（默认 9）")
    parser.add_argument("--cam_front_serial", default=constants.REALSENSE_FRONT_SERIAL,
                        help="正面 RealSense 序列号（默认读取 constants.py）")
    parser.add_argument("--cam_wrist_serial", default=constants.REALSENSE_WRIST_SERIAL,
                        help="腕部 RealSense 序列号（默认读取 constants.py；为空时自动选第二台）")
    parser.add_argument("--tactile_sn_left", default=constants.TACTILE_LEFT_SERIAL,
                        help="左触觉 VitAI GF225 序列号（默认读取 constants.py）")
    parser.add_argument("--tactile_sn_right", default=constants.TACTILE_RIGHT_SERIAL,
                        help="右触觉 VitAI GF225 序列号（默认读取 constants.py）")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  越疆 Dobot / 知行夹爪 / RealSense / VitAI 触觉 / UMI  硬件连通性检测")
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

    # 3 & 4. 相机（pyrealsense2 SDK 直驱）
    if not args.skip_cameras:
        results["cameras"] = check_cameras(
            serial_front=args.cam_front_serial,
            serial_wrist=args.cam_wrist_serial,
        )
    else:
        print(_title("[3] RealSense 相机（pyrealsense2 SDK）"))
        print(_warn("已跳过（--skip_cameras）"))
        results["cameras"] = None

    # 5. 触觉传感器
    if not args.skip_tactile:
        results["tactile"] = check_tactile(
            serial_left=args.tactile_sn_left,
            serial_right=args.tactile_sn_right,
        )
    else:
        print(_title("[4] VitAI GF225 触觉传感器"))
        print(_warn("已跳过（--skip_tactile）"))
        results["tactile"] = None

    # 6. UMI
    if not args.skip_umi:
        results["umi"] = check_umi(
            constants.UMI_VIO_POSE_TOPIC,
            timeout_s=args.ros_timeout,
            domain_id=args.umi_domain_id,
        )
    else:
        print(_title("[5] UMI 示教设备"))
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
        "tactile": "VitAI触觉 ",
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
