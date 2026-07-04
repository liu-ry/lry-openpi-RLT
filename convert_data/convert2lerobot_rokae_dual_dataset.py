"""
Convert dual Rokae raw episodes to a LeRobot dataset.

Example:
    uv run --no-sync convert_data/convert2lerobot_rokae_dual_dataset.py \
        --root-dir /home/lry/RokaeDual \
        --output-dir /home/lry/temp/rokae_dual_lerobot

Dry-run the first few episodes without writing a dataset:
    uv run --no-sync convert_data/convert2lerobot_rokae_dual_dataset.py \
        --root-dir /home/lry/RokaeDual \
        --dry-run \
        --max-episodes 3

Input episode layout:
    episode_*/gripper_left_width.npy
    episode_*/gripper_right_width.npy
    episode_*/realsense_rgb_top.mp4
    episode_*/realsense_rgb_wrist_left.mp4
    episode_*/realsense_rgb_wrist_right.mp4
    episode_*/robot_rokae_dual_left_joint.npy
    episode_*/robot_rokae_dual_right_joint.npy
    episode_*/timestamps.npy
    episode_*/meta.json or metadata.json
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


ARM_DOF = 7
STATE_DIM = 16
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

STATE_NAMES = (
    [f"left_joint_{i}" for i in range(ARM_DOF)]
    + ["left_gripper"]
    + [f"right_joint_{i}" for i in range(ARM_DOF)]
    + ["right_gripper"]
)

VIDEO_FILES = {
    "observation.images.cam_high": "realsense_rgb_top.mp4",
    "observation.images.cam_left_wrist": "realsense_rgb_wrist_left.mp4",
    "observation.images.cam_right_wrist": "realsense_rgb_wrist_right.mp4",
}


def find_existing_file(base_dir: Path, candidates: list[str]) -> Path:
    for name in candidates:
        path = base_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"在 {base_dir} 下未找到任一文件: {candidates}")


def episode_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.split("_")[-1]), path.name
    except ValueError:
        return 10**12, path.name


def extract_video_to_frames(
    video_path: Path,
    target_height: int = IMAGE_HEIGHT,
    target_width: int = IMAGE_WIDTH,
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        if (h, w) != (target_height, target_width):
            frame_rgb = cv2.resize(frame_rgb, (target_width, target_height))
        if frame_rgb.ndim == 2:
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames


def get_video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def get_action_from_state_pos(state_pos: np.ndarray) -> np.ndarray:
    if len(state_pos) < 2:
        raise ValueError("episode 至少需要 2 帧才能构造下一帧动作")
    actions = np.zeros_like(state_pos)
    actions[:-1] = state_pos[1:]
    actions[-1] = actions[-2]
    return actions


def load_metadata(episode_dir: Path) -> dict:
    for name in ["metadata.json", "meta.json"]:
        metadata_path = episode_dir / name
        if metadata_path.exists():
            with metadata_path.open("r") as f:
                return json.load(f)
    return {}


def load_state(
    episode_dir: Path,
    *,
    joints_in_degrees: bool,
    gripper_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    left_joint = np.load(episode_dir / "robot_rokae_dual_left_joint.npy").astype(np.float32)
    right_joint = np.load(episode_dir / "robot_rokae_dual_right_joint.npy").astype(np.float32)
    left_gripper = np.load(episode_dir / "gripper_left_width.npy").astype(np.float32)
    right_gripper = np.load(episode_dir / "gripper_right_width.npy").astype(np.float32)
    if left_joint.ndim != 2 or left_joint.shape[1] < ARM_DOF:
        raise ValueError(f"left joint 数据形状异常: {left_joint.shape}, 期望至少 (N, {ARM_DOF})")
    if right_joint.ndim != 2 or right_joint.shape[1] < ARM_DOF:
        raise ValueError(f"right joint 数据形状异常: {right_joint.shape}, 期望至少 (N, {ARM_DOF})")

    raw_lengths = {
        "left_joint": len(left_joint),
        "right_joint": len(right_joint),
        "left_gripper": len(left_gripper),
        "right_gripper": len(right_gripper),
    }
    if len(set(raw_lengths.values())) > 1:
        raise ValueError(f"state 源数据长度不一致: {raw_lengths}")

    left_joint = left_joint[:, :ARM_DOF]
    right_joint = right_joint[:, :ARM_DOF]
    if joints_in_degrees:
        left_joint = np.deg2rad(left_joint)
        right_joint = np.deg2rad(right_joint)

    left_gripper = (left_gripper * gripper_scale).reshape(-1, 1)
    right_gripper = (right_gripper * gripper_scale).reshape(-1, 1)

    state = np.concatenate([left_joint, left_gripper, right_joint, right_gripper], axis=1).astype(np.float32)
    actions = get_action_from_state_pos(state)
    return state, actions


def create_dataset(output_dir: str, fps: int) -> LeRobotDataset:
    image_features = {
        key: {
            "dtype": "image",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channel"],
        }
        for key in VIDEO_FILES
    }
    features = {
        **image_features,
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": list(STATE_NAMES),
        },
        "actions": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": list(STATE_NAMES),
        },
    }
    return LeRobotDataset.create(
        repo_id=output_dir,
        robot_type="rokae_zhixing_dual",
        fps=fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )


def ensure_clean_output(output_dir: str, *, overwrite: bool) -> bool:
    output_path = Path(output_dir)
    if not output_path.exists():
        return True
    if overwrite:
        shutil.rmtree(output_path)
        return True

    user_input = input(f"目录 '{output_path}' 已存在，是否删除该目录及其所有内容？[y/N] ").strip().lower()
    if user_input in ("y", "yes"):
        shutil.rmtree(output_path)
        return True
    print("已取消删除以及数据转换操作")
    return False


def main(
    root_dir: str = "/home/lry/RokaeDual",
    output_dir: str = "/home/lry/temp/rokae_dual_lerobot",
    fps: int = 30,
    task: str = "Rokae dual manipulation task",
    max_episodes: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    joints_in_degrees: bool = False,
    gripper_scale: float = 1.0,
) -> None:
    """Convert /home/lry/RokaeDual style episodes to LeRobot.

    Args:
        root_dir: Directory containing episode_* folders.
        output_dir: Output LeRobot dataset directory/repo id.
        fps: Dataset fps metadata.
        task: Fallback task prompt when metadata has no prompt/task field.
        max_episodes: Optional limit for debugging.
        overwrite: Delete output_dir without prompting when it already exists.
        dry_run: Validate files, shapes, and lengths without writing output.
        joints_in_degrees: Convert joint arrays from degrees to radians when enabled.
        gripper_scale: Scale gripper width values. Default keeps raw values unchanged.
    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise ValueError(f"root_dir 不存在或不是目录: {root_path}")

    episode_dirs = sorted((p for p in root_path.glob("episode_*") if p.is_dir()), key=episode_sort_key)
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]
    if not episode_dirs:
        raise ValueError(f"未在 {root_path} 下找到 episode_* 目录")

    dataset = None
    if not dry_run:
        if not ensure_clean_output(output_dir, overwrite=overwrite):
            return
        dataset = create_dataset(output_dir, fps)

    succ_num = 0
    error_num = 0
    for episode_idx, episode_dir in enumerate(episode_dirs, start=1):
        print(f"处理 episode {episode_idx}/{len(episode_dirs)}: {episode_dir.name}")
        try:
            metadata = load_metadata(episode_dir)
            episode_task = metadata.get("prompt") or metadata.get("task") or metadata.get("recipe_name") or task
            state_pos, actions = load_state(
                episode_dir,
                joints_in_degrees=joints_in_degrees,
                gripper_scale=gripper_scale,
            )
            timestamps = np.load(episode_dir / "timestamps.npy")

            video_lengths = {
                key: get_video_frame_count(episode_dir / filename)
                for key, filename in VIDEO_FILES.items()
            }
            lengths = [len(state_pos), len(actions), len(timestamps), *video_lengths.values()]
            if len(set(lengths)) > 1:
                print(f"警告: 数据长度不一致 {lengths}，跳过此 episode")
                error_num += 1
                continue

            if dry_run:
                print(
                    f"  OK: steps={len(state_pos)}, state_shape={state_pos.shape}, "
                    f"video_lengths={video_lengths}, task={episode_task!r}"
                )
                succ_num += 1
                continue

            assert dataset is not None
            video_frames = {
                key: extract_video_to_frames(episode_dir / filename)
                for key, filename in VIDEO_FILES.items()
            }
            decoded_lengths = [len(state_pos), len(actions), len(timestamps), *(len(frames) for frames in video_frames.values())]
            if len(set(decoded_lengths)) > 1:
                print(f"警告: 解码后数据长度不一致 {decoded_lengths}，跳过此 episode")
                error_num += 1
                continue

            for step_idx in range(len(timestamps)):
                frame_data = {
                    "observation.images.cam_high": video_frames["observation.images.cam_high"][step_idx],
                    "observation.images.cam_left_wrist": video_frames["observation.images.cam_left_wrist"][step_idx],
                    "observation.images.cam_right_wrist": video_frames["observation.images.cam_right_wrist"][step_idx],
                    "observation.state": state_pos[step_idx],
                    "actions": actions[step_idx],
                    "task": episode_task,
                }
                dataset.add_frame(frame_data)
            dataset.save_episode()
            succ_num += 1
            print(f"  {episode_dir} 处理成功，总计成功 {succ_num} 个")
        except Exception as e:
            error_num += 1
            print(f"处理 episode 时出错 {episode_dir}: {e}，总计错误 {error_num} 个")
            continue

    print(f"数据集转换完成: 成功 {succ_num} 个，错误/跳过 {error_num} 个")


if __name__ == "__main__":
    tyro.cli(main)
