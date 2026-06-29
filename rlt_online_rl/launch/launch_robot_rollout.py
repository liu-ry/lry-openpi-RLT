#!/usr/bin/env python3
from __future__ import annotations

import http.client
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib import error as urllib_error
from urllib import request as urllib_request


def _apply_ros_workspace(setup_bash: str) -> bool:
    """在 os.execv 之前 source ROS2 工作空间，确保子进程继承正确的
    LD_LIBRARY_PATH / PYTHONPATH 等环境变量（动态链接器在进程启动时读取）。
    """
    setup_path = Path(setup_bash).expanduser().resolve()
    if not setup_path.exists():
        return False
    try:
        result = subprocess.run(
            ["bash", "-c", f"source {setup_path} && env -0"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return False
        for item in result.stdout.split("\0"):
            if "=" in item:
                k, _, v = item.partition("=")
                if k in ("LD_LIBRARY_PATH", "PYTHONPATH", "AMENT_PREFIX_PATH",
                         "PATH", "AMENT_CURRENT_PREFIX", "ROS_DISTRO",
                         "AMENT_PYTHON_EXECUTABLE"):
                    os.environ[k] = v
        # 同步 PYTHONPATH → sys.path（对当前进程也生效，便于后续 import）
        for p in os.environ.get("PYTHONPATH", "").split(":"):
            if p and p not in sys.path:
                sys.path.insert(0, p)
        return True
    except Exception as e:
        print(f"[launch_robot_rollout] _apply_ros_workspace 失败: {e}", flush=True)
        return False

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_OUTER = REPO_ROOT.parent  # lry-openpi-RLT 仓库根目录（third_party 在此）
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PIKA_SYNC_ROS = REPO_ROOT / "train_deploy_alignment" / "pika_sync_ros.py"
DOBOT_UMI_ROS = REPO_ROOT / "train_deploy_alignment" / "dobot_umi_ros.py"
ROKAE_ZHIXING_DUAL_ROS = REPO_ROOT / "train_deploy_alignment" / "rokae_zhixing_dual_ros.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "tasks" / "agilex_ethernet" / "online_rl.yaml"

# 内置 ROS 脚本映射表，通过 --ros_script 选项引用
_ROS_SCRIPTS: dict[str, Path] = {
    "pika":      PIKA_SYNC_ROS,
    "dobot_umi": DOBOT_UMI_ROS,
    "rokae_zhixing_dual": ROKAE_ZHIXING_DUAL_ROS,
}

from rlt_online_rl.config import load_system_config_yaml


def _peek_option(argv: list[str], flag: str) -> str | None:
    for idx, token in enumerate(argv):
        if token == flag:
            if idx + 1 >= len(argv):
                raise ValueError(f"{flag} requires a value.")
            return argv[idx + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def _resolve_ros_script_from_config(config_path: str) -> Path:
    config_parts = Path(config_path).expanduser().resolve().parts
    if "dobot_umi" in config_parts:
        return DOBOT_UMI_ROS
    if "rokae_zhixing_dual" in config_parts:
        return ROKAE_ZHIXING_DUAL_ROS
    return PIKA_SYNC_ROS


def _wait_for_http(url: str, *, timeout_sec: float = 30.0) -> None:
    deadline = time.time() + timeout_sec
    last_error: Exception | None = None
    while time.time() < deadline:
        req = urllib_request.Request(url, method="GET")
        try:
            with urllib_request.urlopen(req, timeout=1.0) as response:
                if response.status == 200:
                    return
        except (urllib_error.URLError, ConnectionError, http.client.HTTPException) as exc:
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"Service not ready at {url}") from last_error


def main() -> None:
    argv = sys.argv[1:]

    # ── 解析 --ros_script（不传给 ROS 脚本本身，仅用于选择入口） ─────────────
    ros_script_key = _peek_option(argv, "--ros_script")
    if ros_script_key is not None:
        # 从 argv 中移除 --ros_script <value> 这两个 token
        idx = argv.index("--ros_script")
        argv = argv[:idx] + argv[idx + 2:]
        if ros_script_key in _ROS_SCRIPTS:
            ros_script = _ROS_SCRIPTS[ros_script_key]
        else:
            ros_script = Path(ros_script_key).expanduser().resolve()
    else:
        config_for_script = _peek_option(argv, "--config") or str(DEFAULT_CONFIG)
        ros_script = _resolve_ros_script_from_config(config_for_script)

    config_path = _peek_option(argv, "--config") or str(DEFAULT_CONFIG)
    system = load_system_config_yaml(config_path)

    actor_service_url = _peek_option(argv, "--actor_service_url") or system.env_driver.actor_service_url
    replay_service_url = _peek_option(argv, "--replay_service_url") or system.env_driver.replay_service_url

    _wait_for_http(f"{actor_service_url.rstrip('/')}/version")
    _wait_for_http(f"{replay_service_url.rstrip('/')}/stats")

    # ── 在 execv 前 source ROS2 工作空间，让子进程继承完整的 LD_LIBRARY_PATH ──
    # 候选顺序：仓库内自带 → 外部 handheld-umi_ws
    _ws_candidates = [
        REPO_ROOT_OUTER / "third_party" / "ros2_msgs_ws" / "install" / "setup.bash",
        REPO_ROOT_OUTER.parent / "handheld-umi_ws" / "install" / "setup.bash",
        Path.home() / "handheld-umi_ws" / "install" / "setup.bash",
    ]
    for _ws in _ws_candidates:
        if _apply_ros_workspace(str(_ws)):
            break

    os.chdir(REPO_ROOT)
    os.execv(sys.executable, [sys.executable, str(ros_script), *argv])


if __name__ == "__main__":
    main()
