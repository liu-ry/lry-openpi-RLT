#!/usr/bin/env python3
"""
Append short length-mismatch Rokae dual raw episodes to an existing LeRobot dataset.

This is intended for episodes skipped by the old strict converter when modalities
were off by only a few frames. Matching episodes are trimmed to the shortest
modality length and appended after the existing converted episodes.

Example:
    .venv/bin/python convert_data/append_rokae_dual_short_mismatch_episodes.py \
        --raw-root /data/vt_umi_dataset/raw_dataset/rokae_stack_paper_cups_mixed_0708 \
        --dataset-dir /path/to/existing/lerobot_dataset

Preview only:
    .venv/bin/python convert_data/append_rokae_dual_short_mismatch_episodes.py \
        --raw-root /data/vt_umi_dataset/raw_dataset/rokae_stack_paper_cups_mixed_0708 \
        --dataset-dir /path/to/existing/lerobot_dataset \
        --dry-run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert2lerobot_rokae_dual_dataset import (  # noqa: E402
    ARM_DOF,
    DEFAULT_MAX_LENGTH_DELTA,
    VIDEO_FILES,
    check_lengths_and_get_target,
    collect_episode_lengths,
    convert_episode,
    episode_sort_key,
)

APPEND_LOG_NAME = "appended_raw_episodes.jsonl"


def raw_state_lengths(episode_dir: Path) -> dict[str, int]:
    left_joint = np.load(episode_dir / "robot_rokae_dual_left_joint.npy", mmap_mode="r")
    right_joint = np.load(episode_dir / "robot_rokae_dual_right_joint.npy", mmap_mode="r")
    left_gripper = np.load(episode_dir / "gripper_left_width.npy", mmap_mode="r")
    right_gripper = np.load(episode_dir / "gripper_right_width.npy", mmap_mode="r")

    if left_joint.ndim != 2 or left_joint.shape[1] < ARM_DOF:
        raise ValueError(f"left joint 数据形状异常: {left_joint.shape}, 期望至少 (N, {ARM_DOF})")
    if right_joint.ndim != 2 or right_joint.shape[1] < ARM_DOF:
        raise ValueError(f"right joint 数据形状异常: {right_joint.shape}, 期望至少 (N, {ARM_DOF})")

    return {
        "left_joint": len(left_joint),
        "right_joint": len(right_joint),
        "left_gripper": len(left_gripper),
        "right_gripper": len(right_gripper),
    }


def mismatch_report(episode_dir: Path, *, max_length_delta: int) -> dict[str, Any] | None:
    state_lengths = raw_state_lengths(episode_dir)
    state_target, state_needs_trim = check_lengths_and_get_target(
        state_lengths,
        max_length_delta=max_length_delta,
        context="state 源数据",
    )
    episode_lengths = collect_episode_lengths(episode_dir, state_length=state_target)
    episode_target, episode_needs_trim = check_lengths_and_get_target(
        episode_lengths,
        max_length_delta=max_length_delta,
        context="episode 数据",
    )

    if not state_needs_trim and not episode_needs_trim:
        return None

    return {
        "state_lengths": state_lengths,
        "episode_lengths": episode_lengths,
        "target_length": episode_target,
        "state_needs_trim": state_needs_trim,
        "episode_needs_trim": episode_needs_trim,
    }


def load_appended_raw_paths(dataset_dir: Path) -> set[str]:
    log_path = dataset_dir / "meta" / APPEND_LOG_NAME
    if not log_path.exists():
        return set()

    appended: set[str] = set()
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            raw_path = record.get("raw_path")
            if raw_path:
                appended.add(str(raw_path))
    return appended


def append_log_record(dataset_dir: Path, record: dict[str, Any]) -> None:
    log_path = dataset_dir / "meta" / APPEND_LOG_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def select_candidate_episodes(
    raw_root: Path,
    *,
    max_length_delta: int,
    skip_raw_paths: set[str],
) -> tuple[list[tuple[Path, dict[str, Any]]], list[tuple[Path, str]]]:
    episode_dirs = sorted((p for p in raw_root.glob("episode_*") if p.is_dir()), key=episode_sort_key)
    candidates: list[tuple[Path, dict[str, Any]]] = []
    skipped_errors: list[tuple[Path, str]] = []

    for episode_dir in episode_dirs:
        raw_path = str(episode_dir.resolve())
        if raw_path in skip_raw_paths:
            continue
        try:
            report = mismatch_report(episode_dir, max_length_delta=max_length_delta)
        except Exception as exc:
            skipped_errors.append((episode_dir, str(exc)))
            continue
        if report is not None:
            candidates.append((episode_dir, report))
    return candidates, skipped_errors


def main(
    raw_root: str,
    dataset_dir: str,
    task: str = "Rokae dual manipulation task",
    max_length_delta: int = DEFAULT_MAX_LENGTH_DELTA,
    dry_run: bool = False,
    joints_in_degrees: bool = False,
    gripper_scale: float = 1.0,
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
    include_already_appended: bool = False,
) -> None:
    """Append raw episodes with <= max_length_delta frame mismatches to an existing LeRobot dataset.

    Args:
        raw_root: Directory containing raw episode_* folders.
        dataset_dir: Existing LeRobot dataset directory to append to.
        task: Fallback task prompt when raw metadata has no prompt/task field.
        max_length_delta: Accept length mismatches up to this many frames and trim to shortest length.
        dry_run: Print selected episodes without modifying the existing dataset.
        joints_in_degrees: Convert joint arrays from degrees to radians when enabled.
        gripper_scale: Scale gripper width values. Default keeps raw values unchanged.
        image_writer_threads: Async image writer thread count used while appending.
        image_writer_processes: Async image writer process count used while appending.
        include_already_appended: Ignore meta/appended_raw_episodes.jsonl and allow appending the same raw paths again.
    """
    raw_root_path = Path(raw_root).expanduser().resolve()
    dataset_path = Path(dataset_dir).expanduser().resolve()
    if not raw_root_path.is_dir():
        raise ValueError(f"raw_root 不存在或不是目录: {raw_root_path}")
    if not dry_run and not dataset_path.is_dir():
        raise ValueError(f"dataset_dir 不存在或不是目录: {dataset_path}")

    skip_raw_paths = set()
    if not include_already_appended and dataset_path.is_dir():
        skip_raw_paths = load_appended_raw_paths(dataset_path)
    candidates, skipped_errors = select_candidate_episodes(
        raw_root_path,
        max_length_delta=max_length_delta,
        skip_raw_paths=skip_raw_paths,
    )

    print(f"扫描 raw episodes: {raw_root_path}")
    print(f"待追加短长度差 episode: {len(candidates)}")
    if skipped_errors:
        print(f"扫描时跳过异常 episode: {len(skipped_errors)}")
        for episode_dir, reason in skipped_errors[:20]:
            print(f"  跳过 {episode_dir}: {reason}")
        if len(skipped_errors) > 20:
            print(f"  ... 还有 {len(skipped_errors) - 20} 个异常未显示")

    for episode_dir, report in candidates:
        print(
            f"  {episode_dir.name}: target={report['target_length']}, "
            f"state_lengths={report['state_lengths']}, episode_lengths={report['episode_lengths']}"
        )

    if dry_run:
        print("dry-run: 未修改 dataset")
        return
    if not candidates:
        print("没有需要追加的 episode")
        return

    dataset = LeRobotDataset(repo_id=str(dataset_path))
    dataset.start_image_writer(num_processes=image_writer_processes, num_threads=image_writer_threads)

    succ_num = 0
    error_num = 0
    for episode_dir, report in candidates:
        before_index = dataset.meta.total_episodes
        print(f"追加 {episode_dir} -> lerobot episode_{before_index:06d}")
        try:
            steps = convert_episode(
                episode_dir,
                dataset=dataset,
                task=task,
                dry_run=False,
                joints_in_degrees=joints_in_degrees,
                gripper_scale=gripper_scale,
                max_length_delta=max_length_delta,
            )
            append_log_record(
                dataset_path,
                {
                    "raw_path": str(episode_dir.resolve()),
                    "raw_episode": episode_dir.name,
                    "lerobot_episode_index": before_index,
                    "steps": steps,
                    **report,
                },
            )
            succ_num += 1
            print(f"  追加成功 steps={steps}，总计成功 {succ_num}")
        except Exception as exc:
            error_num += 1
            dataset.clear_episode_buffer()
            print(f"  追加失败 {episode_dir}: {exc}，总计错误 {error_num}")

    print(f"追加完成: 成功 {succ_num} 个，错误 {error_num} 个")
    print(f"追加记录: {dataset_path / 'meta' / APPEND_LOG_NAME}")


if __name__ == "__main__":
    tyro.cli(main)
