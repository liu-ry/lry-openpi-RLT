"""Compute OpenPI/pi0.5 normalization stats for a dual Rokae LeRobot dataset.

This is the dual-arm counterpart of `scripts/compute_norm_stats.py` for datasets
created by `convert_data/convert2lerobot_rokae_dual_dataset.py`.

It uses the same important convention as the standard OpenPI stats script:
actions are loaded as action sequences with length `action_horizon`. For pi0.5,
`Pi0Config.action_horizon` defaults to 50, so this script defaults to 50.

Default state/action layout:
    [left_j0..left_j6, left_gripper, right_j0..right_j6, right_gripper]

Example:
    .venv/bin/python scripts/compute_norm_stats_rokae_dual.py \
        --dataset-dir /home/lry/temp/rokae_dual_lerobot \
        --output-dir /home/lry/temp/rokae_dual_lerobot/assets
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import tqdm
import tyro
from lerobot.common.datasets import lerobot_dataset

import openpi.shared.normalize as normalize
import openpi.transforms as transforms


STATE_DIM = 16
DEFAULT_ACTION_HORIZON = 50


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def parse_bool_mask(mask: str) -> tuple[bool, ...]:
    values: list[bool] = []
    for item in mask.split(","):
        item = item.strip().lower()
        if item in {"1", "true", "t", "yes", "y"}:
            values.append(True)
        elif item in {"0", "false", "f", "no", "n"}:
            values.append(False)
        else:
            raise ValueError(f"Invalid bool mask item {item!r} in {mask!r}")
    if not values:
        raise ValueError("delta_action_mask cannot be empty")
    return tuple(values)


def create_dataset(dataset_dir: Path, action_horizon: int) -> lerobot_dataset.LeRobotDataset:
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(str(dataset_dir))
    return lerobot_dataset.LeRobotDataset(
        str(dataset_dir),
        delta_timestamps={"actions": [t / dataset_meta.fps for t in range(action_horizon)]},
    )


def apply_transforms(item: dict, transform_fns: Sequence[transforms.DataTransformFn]) -> dict:
    data = {
        "state": np.asarray(item["observation.state"], dtype=np.float32),
        "actions": np.asarray(item["actions"], dtype=np.float32),
    }
    for fn in transform_fns:
        data = fn(data)
    return data


def main(
    dataset_dir: Path,
    output_dir: Path | None = None,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    batch_size: int = 128,
    max_frames: int | None = None,
    use_delta_actions: bool = False,
    delta_action_mask: str = "1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0",
) -> None:
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    dataset = create_dataset(dataset_dir, action_horizon)
    num_frames = len(dataset)
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    transform_fns: list[transforms.DataTransformFn] = [RemoveStrings()]
    if use_delta_actions:
        transform_fns.insert(0, transforms.DeltaActions(parse_bool_mask(delta_action_mask)))

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }

    print(f"Dataset: {dataset_dir}")
    print(f"Frames used: {num_frames}/{len(dataset)}")
    print(f"Action horizon: {action_horizon}")
    print(f"Use delta actions: {use_delta_actions}")

    batch: dict[str, list[np.ndarray]] = {"state": [], "actions": []}
    for idx in tqdm.trange(num_frames, desc="Computing stats"):
        item = apply_transforms(dataset[idx], transform_fns)
        state = np.asarray(item["state"], dtype=np.float32)
        actions = np.asarray(item["actions"], dtype=np.float32)

        if state.shape[-1] != STATE_DIM:
            raise ValueError(f"Expected state dim {STATE_DIM}, got {state.shape} at index {idx}")
        if actions.shape[-1] != STATE_DIM:
            raise ValueError(f"Expected action dim {STATE_DIM}, got {actions.shape} at index {idx}")
        if actions.shape[-2] != action_horizon:
            raise ValueError(f"Expected action horizon {action_horizon}, got {actions.shape} at index {idx}")

        batch["state"].append(state)
        batch["actions"].append(actions)

        if len(batch["state"]) >= batch_size:
            stats["state"].update(np.stack(batch["state"], axis=0))
            stats["actions"].update(np.stack(batch["actions"], axis=0))
            batch = {"state": [], "actions": []}

    if batch["state"]:
        stats["state"].update(np.stack(batch["state"], axis=0))
        stats["actions"].update(np.stack(batch["actions"], axis=0))

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    output_dir = output_dir or (dataset_dir / "assets")
    print(f"Writing stats to: {output_dir}")
    normalize.save(output_dir, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
