"""Compute action norm stats from successful critical replay transitions.

The replay journal already contains only transitions built from chunks that were
allowed into replay. For Dobot UMI full_task runs this means critical-phase
chunks when the `c` gate was active. This script further filters to episodes
whose final episode reward is positive, then computes the same delta-action
statistics consumed by ActionRepresentationAdapter.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rlt_online_rl.replay import TransitionSource  # noqa: E402


def _iter_pickle_stream(path: Path):
    with path.open("rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def _record_episode_id(record: dict[str, Any]) -> int:
    return int(np.asarray(record["episode_id"]).reshape(()))


def _record_reward_sum(record: dict[str, Any]) -> float:
    return float(np.asarray(record["rewards"], dtype=np.float32).sum())


def _record_success(record: dict[str, Any]) -> int:
    return int(np.asarray(record.get("success", 0)).reshape(()))


def _record_done(record: dict[str, Any]) -> bool:
    return bool(np.asarray(record.get("done", False)).reshape(()))


def _record_source_chunk(record: dict[str, Any], chunk_len: int) -> np.ndarray:
    if "source_chunk" not in record:
        source = int(np.asarray(record.get("source", int(TransitionSource.RL))).reshape(()))
        return np.full((chunk_len,), source, dtype=np.uint8)
    source_chunk = np.asarray(record["source_chunk"], dtype=np.uint8).reshape(-1)
    if source_chunk.shape[0] < chunk_len:
        padded = np.full((chunk_len,), int(source_chunk[-1]) if source_chunk.size else 0, dtype=np.uint8)
        padded[: source_chunk.shape[0]] = source_chunk
        return padded
    return source_chunk[:chunk_len]


def _episode_reward_sums(journal_path: Path) -> dict[int, float]:
    reward_sums: dict[int, float] = defaultdict(float)
    done_seen: dict[int, bool] = defaultdict(bool)
    success_seen: dict[int, bool] = defaultdict(bool)
    for record in _iter_pickle_stream(journal_path):
        episode_id = _record_episode_id(record)
        reward_sums[episode_id] += _record_reward_sum(record)
        done_seen[episode_id] = done_seen[episode_id] or _record_done(record)
        success_seen[episode_id] = success_seen[episode_id] or _record_success(record) > 0
    return dict(reward_sums)


def _select_action_samples(
    record: dict[str, Any],
    *,
    action_dim: int,
    include: str,
    source_filter: set[int] | None,
) -> list[np.ndarray]:
    state0 = np.asarray(record["proprio"], dtype=np.float32).reshape(-1)[:action_dim]

    def to_delta_repr(chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32)[..., :action_dim].copy()
        n_joints = min(6, action_dim)
        chunk[..., :n_joints] -= state0[:n_joints]
        return chunk

    action_chunk = np.asarray(record["action_chunk"], dtype=np.float32)[..., :action_dim]
    ref_chunk = np.asarray(record["ref_chunk"], dtype=np.float32)[..., :action_dim]
    chunks: list[np.ndarray] = []
    if include in {"action", "both"}:
        chunks.append(to_delta_repr(action_chunk))
    if include in {"ref", "both"}:
        chunks.append(to_delta_repr(ref_chunk))
    if source_filter is None:
        return chunks
    source_chunk = _record_source_chunk(record, action_chunk.shape[0])
    mask = np.isin(source_chunk, np.asarray(sorted(source_filter), dtype=np.uint8))
    return [chunk[mask] for chunk in chunks if np.any(mask)]


def _compute_stats(samples: np.ndarray, q_low: float, q_high: float) -> dict[str, list[float]]:
    return {
        "mean": samples.mean(axis=0).tolist(),
        "std": samples.std(axis=0).tolist(),
        "q01": np.percentile(samples, q_low * 100.0, axis=0).tolist(),
        "q99": np.percentile(samples, q_high * 100.0, axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--journal-path",
        type=Path,
        default=ROOT / "runs" / "dobot_umi" / "replay" / "replay_journal.pkl",
        help="Path to replay_journal.pkl.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "configs" / "tasks" / "dobot_umi" / "stats" / "norm_stats_delta_critical.json",
    )
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--proprio-dim", type=int, default=7)
    parser.add_argument("--q-low", type=float, default=0.01)
    parser.add_argument("--q-high", type=float, default=0.99)
    parser.add_argument(
        "--final-reward",
        type=float,
        default=1.0,
        help="Keep episodes whose total replay reward is at least this value.",
    )
    parser.add_argument(
        "--include",
        choices=("action", "ref", "both"),
        default="both",
        help="Which chunks to include in action stats. 'both' matches actor training inputs and BC targets.",
    )
    parser.add_argument(
        "--source",
        choices=("all", "human", "base", "rl", "mixed"),
        default="all",
        help="Optional per-step source filter before computing action stats.",
    )
    args = parser.parse_args()

    if not args.journal_path.exists():
        raise FileNotFoundError(args.journal_path)
    if not (0.0 <= args.q_low < args.q_high <= 1.0):
        raise ValueError(f"Invalid quantiles q_low={args.q_low} q_high={args.q_high}")

    source_filter = None
    if args.source != "all":
        source_filter = {
            "base": int(TransitionSource.BASE),
            "rl": int(TransitionSource.RL),
            "human": int(TransitionSource.HUMAN),
            "mixed": int(TransitionSource.MIXED),
        }[args.source]
        source_filter = {source_filter}

    episode_rewards = _episode_reward_sums(args.journal_path)
    successful_episode_ids = {
        episode_id for episode_id, reward_sum in episode_rewards.items() if reward_sum >= float(args.final_reward)
    }
    if not successful_episode_ids:
        raise ValueError(f"No episodes found with total replay reward >= {args.final_reward}.")

    action_samples: list[np.ndarray] = []
    state_samples: list[np.ndarray] = []
    transition_count = 0
    kept_transition_count = 0
    for record in _iter_pickle_stream(args.journal_path):
        transition_count += 1
        episode_id = _record_episode_id(record)
        if episode_id not in successful_episode_ids:
            continue
        kept_transition_count += 1
        action_samples.extend(
            _select_action_samples(
                record,
                action_dim=args.action_dim,
                include=args.include,
                source_filter=source_filter,
            )
        )
        state_samples.append(np.asarray(record["proprio"], dtype=np.float32).reshape(-1)[: args.proprio_dim])

    action_samples = [sample for sample in action_samples if sample.size]
    if not action_samples or not state_samples:
        raise ValueError("No samples left after filtering.")

    all_actions = np.concatenate(action_samples, axis=0).astype(np.float32, copy=False)
    all_states = np.stack(state_samples, axis=0).astype(np.float32, copy=False)
    if all_actions.shape[1] != args.action_dim:
        raise ValueError(f"Expected action_dim={args.action_dim}, got samples with shape {all_actions.shape}")
    if all_states.shape[1] != args.proprio_dim:
        raise ValueError(f"Expected proprio_dim={args.proprio_dim}, got samples with shape {all_states.shape}")

    stats = {
        "metadata": {
            "source": str(args.journal_path),
            "filter": "successful critical replay transitions",
            "final_reward_min": float(args.final_reward),
            "include": args.include,
            "source_filter": args.source,
            "successful_episode_count": len(successful_episode_ids),
            "total_transition_count": transition_count,
            "kept_transition_count": kept_transition_count,
            "action_sample_count": int(all_actions.shape[0]),
            "state_sample_count": int(all_states.shape[0]),
            "q_low": float(args.q_low),
            "q_high": float(args.q_high),
        },
        "norm_stats": {
            "actions": _compute_stats(all_actions, args.q_low, args.q_high),
            "state": _compute_stats(all_states, args.q_low, args.q_high),
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Loaded transitions: {transition_count}")
    print(f"Successful episodes: {len(successful_episode_ids)} / {len(episode_rewards)}")
    print(f"Kept transitions: {kept_transition_count}")
    print(f"Action samples: {all_actions.shape[0]}")
    print(f"State samples: {all_states.shape[0]}")
    print(f"Action q01: {[f'{v:.6g}' for v in stats['norm_stats']['actions']['q01']]}")
    print(f"Action q99: {[f'{v:.6g}' for v in stats['norm_stats']['actions']['q99']]}")
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
