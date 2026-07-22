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
from typing import Any

import cv2
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.compute_stats import sample_indices


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
DEFAULT_MAX_LENGTH_DELTA = 3


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


def check_lengths_and_get_target(
    lengths: dict[str, int],
    *,
    max_length_delta: int,
    context: str,
) -> tuple[int, bool]:
    min_length = min(lengths.values())
    max_length = max(lengths.values())
    needs_trim = min_length != max_length
    if max_length - min_length > max_length_delta:
        raise ValueError(f"{context}长度不一致且超过 {max_length_delta} 帧: {lengths}")
    if min_length < 2:
        raise ValueError(f"{context}有效长度过短: {lengths}")
    return min_length, needs_trim


def trim_array(array: np.ndarray, length: int) -> np.ndarray:
    return array[:length]


def trim_frames(frames: list[np.ndarray], length: int) -> list[np.ndarray]:
    return frames[:length]


def save_episode_with_source_videos(
    dataset: LeRobotDataset,
    *,
    episode_dir: Path,
    state_pos: np.ndarray,
    actions: np.ndarray,
    task: str,
) -> None:
    """Save parquet metadata and copy MP4s without decoding frames to temporary PNGs.

    ``LeRobotDataset.add_frame`` writes every video frame as a temporary PNG even
    when the final MP4 already exists.  Supplying the episode buffer directly is
    safe here because ``save_episode`` stores video fields outside Parquet and
    only requires their final MP4 files to exist before it runs.
    """
    episode_index = dataset.meta.total_episodes
    episode_length = len(state_pos)
    episode_buffer = dataset.create_episode_buffer(episode_index=episode_index)
    episode_buffer["size"] = episode_length
    episode_buffer["task"] = [task] * episode_length
    episode_buffer["frame_index"] = list(range(episode_length))
    episode_buffer["timestamp"] = [frame_index / dataset.fps for frame_index in range(episode_length)]
    episode_buffer["observation.state"] = list(state_pos)
    episode_buffer["actions"] = list(actions)

    # LeRobot calculates image normalization statistics from the temporary
    # image paths in the episode buffer.  Preserve correct statistics while
    # only decoding/writing its sampled frames (rather than every frame).
    for key, filename in VIDEO_FILES.items():
        target_path = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(episode_dir / filename, target_path)
        episode_buffer[key] = extract_video_stat_samples(
            episode_dir / filename,
            dataset=dataset,
            episode_index=episode_index,
            video_key=key,
            frame_count=episode_length,
        )

    dataset.episode_buffer = episode_buffer
    dataset.save_episode()


def extract_video_stat_samples(
    video_path: Path,
    *,
    dataset: LeRobotDataset,
    episode_index: int,
    video_key: str,
    frame_count: int,
) -> list[str]:
    """Write only the frames LeRobot samples for per-episode image statistics."""
    sampled_indices = set(sample_indices(frame_count))
    first_sample_path: Path | None = None
    frame_paths: list[str | None] = [None] * frame_count
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    try:
        for frame_index in range(frame_count):
            if not cap.grab():
                raise ValueError(f"读取视频帧失败: {video_path}, frame={frame_index}")
            if frame_index not in sampled_indices:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                raise ValueError(f"读取视频帧失败: {video_path}, frame={frame_index}")
            image_path = dataset._get_image_file_path(episode_index, video_key, frame_index)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(image_path), frame):
                raise ValueError(f"写入统计采样帧失败: {image_path}")
            frame_paths[frame_index] = str(image_path)
            if first_sample_path is None:
                first_sample_path = image_path
    finally:
        cap.release()

    if first_sample_path is None:
        raise ValueError(f"视频中没有可用于统计的帧: {video_path}")
    # save_episode only reads the indices returned by sample_indices(). The
    # remaining entries must merely be valid paths for its episode buffer.
    return [path or str(first_sample_path) for path in frame_paths]


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
    max_length_delta: int = DEFAULT_MAX_LENGTH_DELTA,
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
    target_length, needs_trim = check_lengths_and_get_target(
        raw_lengths,
        max_length_delta=max_length_delta,
        context="state 源数据",
    )
    if needs_trim:
        print(f"  警告: state 源数据长度不一致 {raw_lengths}，裁剪到 {target_length} 帧")

    left_joint = left_joint[:target_length, :ARM_DOF]
    right_joint = right_joint[:target_length, :ARM_DOF]
    if joints_in_degrees:
        left_joint = np.deg2rad(left_joint)
        right_joint = np.deg2rad(right_joint)

    left_gripper = (left_gripper[:target_length] * gripper_scale).reshape(-1, 1)
    right_gripper = (right_gripper[:target_length] * gripper_scale).reshape(-1, 1)

    state = np.concatenate([left_joint, left_gripper, right_joint, right_gripper], axis=1).astype(np.float32)
    actions = get_action_from_state_pos(state)
    return state, actions


def get_episode_task(metadata: dict[str, Any], fallback_task: str) -> str:
    return metadata.get("prompt") or metadata.get("task") or metadata.get("recipe_name") or fallback_task


def collect_episode_lengths(
    episode_dir: Path,
    *,
    state_length: int | None = None,
) -> dict[str, int]:
    lengths: dict[str, int] = {}
    if state_length is not None:
        lengths["state"] = state_length
    lengths["timestamps"] = len(np.load(episode_dir / "timestamps.npy"))
    for key, filename in VIDEO_FILES.items():
        lengths[key] = get_video_frame_count(episode_dir / filename)
    return lengths


def convert_episode(
    episode_dir: Path,
    *,
    dataset: LeRobotDataset | None,
    task: str,
    dry_run: bool,
    joints_in_degrees: bool,
    gripper_scale: float,
    copy_source_videos: bool,
    max_length_delta: int = DEFAULT_MAX_LENGTH_DELTA,
) -> int:
    metadata = load_metadata(episode_dir)
    episode_task = get_episode_task(metadata, task)
    state_pos, actions = load_state(
        episode_dir,
        joints_in_degrees=joints_in_degrees,
        gripper_scale=gripper_scale,
        max_length_delta=max_length_delta,
    )
    timestamps = np.load(episode_dir / "timestamps.npy")

    lengths = collect_episode_lengths(episode_dir, state_length=len(state_pos))
    target_length, needs_trim = check_lengths_and_get_target(
        lengths,
        max_length_delta=max_length_delta,
        context="episode 数据",
    )
    if needs_trim:
        print(f"  警告: episode 数据长度不一致 {lengths}，裁剪到 {target_length} 帧")

    state_pos = trim_array(state_pos, target_length)
    actions = trim_array(actions, target_length)
    timestamps = trim_array(timestamps, target_length)

    if dry_run:
        print(
            f"  OK: steps={target_length}, state_shape={state_pos.shape}, "
            f"lengths={lengths}, task={episode_task!r}"
        )
        return target_length

    if dataset is None:
        raise ValueError("dataset 不能为空，除非 dry_run=True")

    if copy_source_videos:
        save_episode_with_source_videos(
            dataset,
            episode_dir=episode_dir,
            state_pos=state_pos,
            actions=actions,
            task=episode_task,
        )
        return target_length

    # Compatibility path for source videos that need to be re-encoded.
    video_frames = {key: extract_video_to_frames(episode_dir / filename) for key, filename in VIDEO_FILES.items()}
    decoded_lengths = {
        "state": len(state_pos),
        "actions": len(actions),
        "timestamps": len(timestamps),
        **{key: len(frames) for key, frames in video_frames.items()},
    }
    decoded_target_length, decoded_needs_trim = check_lengths_and_get_target(
        decoded_lengths,
        max_length_delta=max_length_delta,
        context="解码后 episode 数据",
    )
    if decoded_needs_trim:
        print(f"  警告: 解码后数据长度不一致 {decoded_lengths}，裁剪到 {decoded_target_length} 帧")
        state_pos = trim_array(state_pos, decoded_target_length)
        actions = trim_array(actions, decoded_target_length)
        video_frames = {key: trim_frames(frames, decoded_target_length) for key, frames in video_frames.items()}
        target_length = decoded_target_length

    for step_idx in range(target_length):
        dataset.add_frame(
            {
                "observation.images.cam_high": video_frames["observation.images.cam_high"][step_idx],
                "observation.images.cam_left_wrist": video_frames["observation.images.cam_left_wrist"][step_idx],
                "observation.images.cam_right_wrist": video_frames["observation.images.cam_right_wrist"][step_idx],
                "observation.state": state_pos[step_idx],
                "actions": actions[step_idx],
                "task": episode_task,
            }
        )
    dataset.save_episode()
    return target_length


def create_dataset(output_dir: str, fps: int, image_writer_threads: int) -> LeRobotDataset:
    video_features = {
        key: {
            # Store camera streams as MP4, not raw image bytes in Parquet.
            "dtype": "video",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channel"],
        }
        for key in VIDEO_FILES
    }
    features = {
        **video_features,
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
        # Keep image arrays in this process. Multiprocessing serializes every
        # frame into an unbounded queue and was the cause of the 24 GiB OOM.
        image_writer_threads=image_writer_threads,
        image_writer_processes=0,
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
    image_writer_threads: int = 4,
    copy_source_videos: bool = True,
    max_length_delta: int = DEFAULT_MAX_LENGTH_DELTA,
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
        image_writer_threads: Number of in-process temporary-image writer threads.
        copy_source_videos: Copy already-compatible source MP4s instead of re-encoding them.
        max_length_delta: Accept length mismatches up to this many frames and trim all modalities to the shortest length.
    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise ValueError(f"root_dir 不存在或不是目录: {root_path}")

    episode_dirs = sorted((p for p in root_path.glob("episode_*") if p.is_dir()), key=episode_sort_key)
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]
    if not episode_dirs:
        raise ValueError(f"未在 {root_path} 下找到 episode_* 目录")
    if image_writer_threads < 1:
        raise ValueError("image_writer_threads 至少为 1")

    dataset = None
    if not dry_run:
        if not ensure_clean_output(output_dir, overwrite=overwrite):
            return
        dataset = create_dataset(output_dir, fps, image_writer_threads)

    succ_num = 0
    error_num = 0
    for episode_idx, episode_dir in enumerate(episode_dirs, start=1):
        print(f"处理 episode {episode_idx}/{len(episode_dirs)}: {episode_dir.name}")
        try:
            steps = convert_episode(
                episode_dir,
                dataset=dataset,
                task=task,
                dry_run=dry_run,
                joints_in_degrees=joints_in_degrees,
                gripper_scale=gripper_scale,
                copy_source_videos=copy_source_videos,
                max_length_delta=max_length_delta,
            )
            succ_num += 1
            print(f"  {episode_dir} 处理成功，steps={steps}，总计成功 {succ_num} 个")
        except Exception as e:
            error_num += 1
            print(f"处理 episode 时出错 {episode_dir}: {e}，总计错误 {error_num} 个")
            continue

    print(f"数据集转换完成: 成功 {succ_num} 个，错误/跳过 {error_num} 个")


if __name__ == "__main__":
    tyro.cli(main)
