#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.client
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib import error as urllib_error
from urllib import request as urllib_request

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_OUTER = REPO_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PIKA_SYNC_ROS = REPO_ROOT / "train_deploy_alignment" / "pika_sync_ros.py"
DOBOT_UMI_ROS = REPO_ROOT / "train_deploy_alignment" / "dobot_umi_ros.py"
ROKAE_ZHIXING_DUAL_ROS = REPO_ROOT / "train_deploy_alignment" / "rokae_zhixing_dual_ros.py"
TIANJI_MARVIN_DUAL_ROS = REPO_ROOT / "train_deploy_alignment" / "tianji_marvin_dual_ros.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "tasks" / "agilex_ethernet" / "online_rl.yaml"

_ROS_SCRIPTS: dict[str, Path] = {
    "pika": PIKA_SYNC_ROS,
    "dobot_umi": DOBOT_UMI_ROS,
    "rokae_zhixing_dual": ROKAE_ZHIXING_DUAL_ROS,
    "tianji_marvin_dual": TIANJI_MARVIN_DUAL_ROS,
}

from rlt_online_rl.config import load_system_config_yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch eval rollout with actor_service only.")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--ros_script",
        type=str,
        default=None,
        help="Robot rollout entrypoint: pika, dobot_umi, rokae_zhixing_dual, tianji_marvin_dual, or an explicit script path.",
    )
    args, remaining = parser.parse_known_args()
    args.remaining = remaining
    return args


def _resolve_config_path(run_dir: str | None, config: str | None) -> str:
    if config is not None:
        return config
    if run_dir is None:
        return str(DEFAULT_CONFIG)
    resolved = Path(run_dir) / "checkpoints" / "online_rl_config.yaml"
    if resolved.exists():
        return str(resolved)
    return str(DEFAULT_CONFIG)


def _peek_option(argv: list[str], flag: str) -> str | None:
    for idx, token in enumerate(argv):
        if token == flag:
            if idx + 1 >= len(argv):
                raise ValueError(f"{flag} requires a value.")
            return argv[idx + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def _apply_ros_workspace(setup_bash: str) -> bool:
    setup_path = Path(setup_bash).expanduser().resolve()
    if not setup_path.exists():
        return False
    try:
        result = subprocess.run(
            ["bash", "-c", f"source {setup_path} && env -0"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        print(f"[launch_actor_eval] failed to source {setup_path}: {exc}", flush=True)
        return False
    if result.returncode != 0:
        return False
    for item in result.stdout.split("\0"):
        if "=" not in item:
            continue
        key, _, value = item.partition("=")
        if key in {
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "AMENT_PREFIX_PATH",
            "PATH",
            "AMENT_CURRENT_PREFIX",
            "ROS_DISTRO",
            "AMENT_PYTHON_EXECUTABLE",
        }:
            os.environ[key] = value
    for path in os.environ.get("PYTHONPATH", "").split(":"):
        if path and path not in sys.path:
            sys.path.insert(0, path)
    return True


def _resolve_ros_script(config_path: str, ros_script: str | None) -> Path:
    if ros_script:
        if ros_script in _ROS_SCRIPTS:
            return _ROS_SCRIPTS[ros_script]
        return Path(ros_script).expanduser().resolve()

    config_parts = Path(config_path).expanduser().resolve().parts
    if "dobot_umi" in config_parts:
        return DOBOT_UMI_ROS
    if "rokae_zhixing_dual" in config_parts:
        return ROKAE_ZHIXING_DUAL_ROS
    if "tianji_marvin_dual" in config_parts:
        return TIANJI_MARVIN_DUAL_ROS
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
    args = _parse_args()
    config_path = _resolve_config_path(args.run_dir, args.config)
    ros_script = _resolve_ros_script(config_path, args.ros_script)
    system = load_system_config_yaml(config_path)
    actor_service_url = _peek_option(args.remaining, "--actor_service_url") or system.env_driver.actor_service_url

    print(f"[launch_actor_eval] using config {config_path}", flush=True)
    print(f"[launch_actor_eval] using robot script {ros_script}", flush=True)
    print(f"[launch_actor_eval] waiting for actor_service at {actor_service_url}", flush=True)
    _wait_for_http(f"{actor_service_url.rstrip('/')}/version")
    print(f"[launch_actor_eval] actor_service ready, starting {ros_script.name} eval rollout.", flush=True)

    for setup_bash in [
        REPO_ROOT_OUTER / "third_party" / "ros2_msgs_ws" / "install" / "setup.bash",
        REPO_ROOT_OUTER.parent / "handheld-umi_ws" / "install" / "setup.bash",
        Path.home() / "handheld-umi_ws" / "install" / "setup.bash",
    ]:
        if _apply_ros_workspace(str(setup_bash)):
            break

    argv = [
        sys.executable,
        str(ros_script),
        "--config",
        config_path,
        "--eval_actor_only",
        *args.remaining,
    ]
    os.chdir(REPO_ROOT)
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
