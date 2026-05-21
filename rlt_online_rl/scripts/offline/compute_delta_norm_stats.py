"""Compute delta-action quantile norm stats from a LeRobot dataset.

Reads all parquet episode files, computes per-step delta actions
(action[t] - state[t], joints 0-5; joint 6 gripper kept as-is),
then saves q01/q99 for both delta_actions and state to a JSON file
compatible with ActionRepresentationAdapter.

Usage:
    python scripts/offline/compute_delta_norm_stats.py \
        --dataset-dir /home/lry/temp/sync/converted \
        --output-path configs/tasks/dobot_umi/stats/norm_stats_delta.json \
        [--action-dim 7] [--proprio-dim 7] [--q-low 0.01] [--q-high 0.99]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def load_lerobot_episodes(dataset_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return (actions_list, states_list) – one array per episode."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required: pip install pandas pyarrow") from e

    data_dir = dataset_dir / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    all_actions: list[np.ndarray] = []
    all_states: list[np.ndarray] = []

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        if "action" not in df.columns and "actions" not in df.columns:
            raise KeyError(f"Neither 'action' nor 'actions' column found in {pf}")
        action_col = "action" if "action" in df.columns else "actions"
        state_col = "observation.state"
        if state_col not in df.columns:
            raise KeyError(f"Column '{state_col}' not found in {pf}")

        actions = np.stack(df[action_col].to_numpy())  # (T, action_dim)
        states = np.stack(df[state_col].to_numpy())    # (T, proprio_dim)
        all_actions.append(actions.astype(np.float32))
        all_states.append(states.astype(np.float32))
        print(f"  Loaded {pf.name}: {actions.shape[0]} steps")

    return all_actions, all_states


def compute_delta_actions(
    actions: np.ndarray,
    states: np.ndarray,
    *,
    action_dim: int = 7,
) -> np.ndarray:
    """Compute per-step delta: action[:6] - state[:6], action[6] kept absolute."""
    assert actions.shape[0] == states.shape[0], "Mismatch in time axis"
    n_joints = min(6, action_dim - 1)  # joints 0..5 are position, joint 6 is gripper
    delta = actions.copy()
    delta[:, :n_joints] = actions[:, :n_joints] - states[:, :n_joints]
    return delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute delta-action norm stats from LeRobot dataset")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to LeRobot dataset root")
    parser.add_argument(
        "--output-path", type=Path, required=True,
        help="Output JSON path, e.g. configs/tasks/dobot_umi/stats/norm_stats_delta.json"
    )
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--proprio-dim", type=int, default=7)
    parser.add_argument("--q-low", type=float, default=0.01, help="Lower quantile (default: 0.01)")
    parser.add_argument("--q-high", type=float, default=0.99, help="Upper quantile (default: 0.99)")
    args = parser.parse_args()

    print(f"Loading LeRobot dataset from {args.dataset_dir} ...")
    all_actions_eps, all_states_eps = load_lerobot_episodes(args.dataset_dir)

    print("Computing delta actions ...")
    delta_chunks: list[np.ndarray] = []
    state_chunks: list[np.ndarray] = []
    for actions_ep, states_ep in zip(all_actions_eps, all_states_eps):
        a = actions_ep[:, : args.action_dim]
        s = states_ep[:, : args.proprio_dim]
        d = compute_delta_actions(a, s, action_dim=args.action_dim)
        delta_chunks.append(d)
        state_chunks.append(s)

    all_deltas = np.concatenate(delta_chunks, axis=0)   # (N, action_dim)
    all_states = np.concatenate(state_chunks, axis=0)   # (N, proprio_dim)

    print(f"Total steps: {all_deltas.shape[0]}")
    print(f"Delta action stats (before clipping):")
    print(f"  min  : {all_deltas.min(axis=0)}")
    print(f"  max  : {all_deltas.max(axis=0)}")
    print(f"  mean : {all_deltas.mean(axis=0)}")
    print(f"  std  : {all_deltas.std(axis=0)}")

    action_q01 = np.percentile(all_deltas, args.q_low * 100, axis=0).tolist()
    action_q99 = np.percentile(all_deltas, args.q_high * 100, axis=0).tolist()
    state_q01 = np.percentile(all_states, args.q_low * 100, axis=0).tolist()
    state_q99 = np.percentile(all_states, args.q_high * 100, axis=0).tolist()
    action_mean = all_deltas.mean(axis=0).tolist()
    action_std = all_deltas.std(axis=0).tolist()
    state_mean = all_states.mean(axis=0).tolist()
    state_std = all_states.std(axis=0).tolist()

    print(f"\nDelta action q01 : {[f'{v:.5f}' for v in action_q01]}")
    print(f"Delta action q99 : {[f'{v:.5f}' for v in action_q99]}")
    print(f"State       q01  : {[f'{v:.5f}' for v in state_q01]}")
    print(f"State       q99  : {[f'{v:.5f}' for v in state_q99]}")

    stats = {
        "norm_stats": {
            "actions": {
                "mean": action_mean,
                "std": action_std,
                "q01": action_q01,
                "q99": action_q99,
            },
            "state": {
                "mean": state_mean,
                "std": state_std,
                "q01": state_q01,
                "q99": state_q99,
            },
        }
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
