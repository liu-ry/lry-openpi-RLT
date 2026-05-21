"""Build a replay journal from a LeRobot offline dataset by querying the running
policy server (Machine A) for z_rl tokens and ref_chunks.

This script bridges the gap between a standard VLA-only LeRobot dataset and the
RLT replay journal format. It calls the running RLT policy server to obtain the
RL-Token (z_rl) and the VLA reference action chunk (ref_chunk) for every step,
then packs everything into a replay journal that the offline training scripts can
consume directly.

Typical usage
-------------
    cd rlt_online_rl

    python scripts/offline/build_replay_from_lerobot.py \\
        --dataset-dir /home/lry/temp/sync/converted \\
        --server-url ws://MACHINE_A_IP:8000 \\
        --output-journal runs/agilex_ethernet/replay/replay_journal_from_lerobot.pkl \\
        --rl-config-path configs/tasks/agilex_ethernet/online_rl.yaml \\
        --chunk-len 10 \\
        --stride 2 \\
        --batch-size 8 \\
        --reward 1.0

After building the journal, run offline training normally:

    python scripts/offline/offline_train_from_replay.py \\
        --replay-path runs/agilex_ethernet/replay/replay_journal_from_lerobot.pkl \\
        --steps 5000 \\
        --batch-size 128
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# bootstrap src path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
from PIL import Image

from rlt_online_rl.config import RLTOnlineRLConfig, load_system_config_yaml
from rlt_online_rl.inference import MachineAFeatureClient
from rlt_online_rl.replay import (
    ReplayManager,
    RLTTransition,
    TransitionSource,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _decode_image(value: Any) -> np.ndarray:
    """Decode a parquet image cell (dict with 'bytes') to a uint8 HWC array."""
    if isinstance(value, dict) and "bytes" in value:
        raw = value["bytes"]
    elif isinstance(value, (bytes, bytearray)):
        raw = value
    else:
        raise ValueError(f"Unsupported image cell type: {type(value)}")
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _load_parquet_episode(parquet_path: Path) -> pd.DataFrame:
    return pd.read_parquet(str(parquet_path))


def _build_obs_dict(row: pd.Series, image_keys: list[str]) -> dict[str, Any]:
    """Build the observation dict expected by the RLT policy server."""
    state = np.asarray(row["observation.state"], dtype=np.float32)
    images: dict[str, np.ndarray] = {}
    for key in image_keys:
        col = f"observation.images.{key}"
        if col in row.index and row[col] is not None:
            images[key] = _decode_image(row[col])
    return {"state": state, "images": images}


def _proprio_from_obs(obs: dict[str, Any], proprio_dim: int) -> np.ndarray:
    state = np.asarray(obs["state"], dtype=np.float32)
    return state[:proprio_dim]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build replay journal from a LeRobot dataset by querying the RLT policy server."
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Root directory of the LeRobot dataset (contains meta/, data/).",
    )
    p.add_argument(
        "--server-url",
        type=str,
        default="ws://127.0.0.1:8000",
        help="WebSocket URL of the running RLT policy server (Machine A).",
    )
    p.add_argument(
        "--output-journal",
        type=Path,
        required=True,
        help="Path to write the output replay_journal.pkl.",
    )
    p.add_argument(
        "--rl-config-path",
        type=Path,
        default=None,
        help=(
            "Path to the online_rl.yaml config (used to read z_dim, proprio_dim, etc.). "
            "If omitted, default RLTOnlineRLConfig values are used."
        ),
    )
    p.add_argument(
        "--chunk-len",
        type=int,
        default=10,
        help="Chunk length for building transitions (default: 10).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Stride for sliding-window transition building (default: 2).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of observations to send to the server in each batch (default: 8).",
    )
    p.add_argument(
        "--reward",
        type=float,
        default=1.0,
        help="Reward to assign to every step (default: 1.0 — treat all as successful expert demos).",
    )
    p.add_argument(
        "--image-keys",
        nargs="+",
        default=["cam_top", "cam_wrist", "tactile_left", "tactile_right"],
        help="Image keys to include in observations (default: cam_top cam_wrist tactile_left tactile_right).",
    )
    p.add_argument(
        "--capacity",
        type=int,
        default=200_000,
        help="Replay buffer capacity (default: 200000).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing journal if present.",
    )
    p.add_argument(
        "--connect-timeout",
        type=float,
        default=30.0,
        help="Timeout (seconds) for initial connection to the policy server.",
    )
    p.add_argument(
        "--recv-timeout",
        type=float,
        default=30.0,
        help="Per-request receive timeout (seconds) for the policy server.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the dataset and report episode/step counts without calling the server.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load RL config
    # ------------------------------------------------------------------
    if args.rl_config_path is not None and args.rl_config_path.exists():
        system_cfg = load_system_config_yaml(str(args.rl_config_path))
        rl_config: RLTOnlineRLConfig = system_cfg.rl
        print(f"[config] Loaded RL config from {args.rl_config_path}")
    else:
        rl_config = RLTOnlineRLConfig()
        print("[config] Using default RLTOnlineRLConfig")

    print(
        f"[config] z_dim={rl_config.z_dim}, proprio_dim={rl_config.proprio_dim}, "
        f"chunk_len={rl_config.chunk_len}, action_dim={rl_config.action_dim}"
    )

    # ------------------------------------------------------------------
    # Discover parquet files
    # ------------------------------------------------------------------
    dataset_dir = args.dataset_dir
    data_dir = dataset_dir / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    # Group files by episode_index (filename: episode_XXXXXX.parquet)
    import re
    episode_files: dict[int, Path] = {}
    for pf in parquet_files:
        m = re.match(r"episode_(\d+)\.parquet$", pf.name)
        if m:
            episode_files[int(m.group(1))] = pf

    if not episode_files:
        raise FileNotFoundError(f"No episode_XXXXXX.parquet files found under {data_dir}")

    total_episodes = len(episode_files)
    print(f"[dataset] Found {total_episodes} episodes in {data_dir}")

    # Count steps
    total_steps = 0
    episode_step_counts: dict[int, int] = {}
    for ep_idx, pf in sorted(episode_files.items()):
        df = _load_parquet_episode(pf)
        episode_step_counts[ep_idx] = len(df)
        total_steps += len(df)
    print(f"[dataset] Total steps across all episodes: {total_steps}")

    if args.dry_run:
        for ep_idx, n_steps in sorted(episode_step_counts.items()):
            print(f"  episode {ep_idx:4d}: {n_steps} steps")
        print("[dry-run] Done. Exiting without building journal.")
        return

    # ------------------------------------------------------------------
    # Connect to policy server
    # ------------------------------------------------------------------
    print(f"[server] Connecting to {args.server_url} ...")
    client = MachineAFeatureClient(
        args.server_url,
        connect_timeout_sec=args.connect_timeout,
        recv_timeout_sec=args.recv_timeout,
    )
    print("[server] Connected.")

    # ------------------------------------------------------------------
    # Setup output
    # ------------------------------------------------------------------
    output_path = args.output_journal
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if args.overwrite:
            output_path.unlink()
            print(f"[output] Removed existing journal at {output_path}")
        else:
            print(f"[output] Journal already exists at {output_path}. Use --overwrite to replace.")
            return

    replay_manager = ReplayManager(
        capacity=args.capacity,
        journal_path=str(output_path),
        seed=0,
    )

    # ------------------------------------------------------------------
    # Process each episode
    # ------------------------------------------------------------------
    chunk_len_to_use = args.chunk_len
    stride_to_use = args.stride
    action_dim = rl_config.action_dim
    proprio_dim = rl_config.proprio_dim
    z_dim = rl_config.z_dim
    chunk_len_cfg = chunk_len_to_use

    def _get_z_rl(feat: dict[str, Any]) -> np.ndarray:
        z = np.asarray(feat["z_rl"], dtype=np.float32).reshape(-1)
        # Truncate to z_dim; never zero-pad (mismatched z_dim means wrong config)
        return z[:z_dim]

    def _get_ref_chunk(feat: dict[str, Any]) -> np.ndarray:
        rc = np.asarray(feat["ref_chunk"], dtype=np.float32)
        return rc[:chunk_len_cfg, :action_dim]

    def _get_proprio(feat: dict[str, Any]) -> np.ndarray:
        return np.asarray(feat["proprio"], dtype=np.float32)[:proprio_dim]

    global_step_count = 0

    for ep_idx, pf in sorted(episode_files.items()):
        df = _load_parquet_episode(pf)
        n_steps = len(df)
        print(f"[episode {ep_idx}] Processing {n_steps} steps ...")

        # Determine available image keys for this episode
        available_image_keys = [
            k for k in args.image_keys
            if f"observation.images.{k}" in df.columns
        ]
        if not available_image_keys:
            print(f"  WARNING: No image columns found, available columns: {df.columns.tolist()}")

        # Build all obs dicts for the episode
        all_obs: list[dict[str, Any]] = []
        for i in range(n_steps):
            row = df.iloc[i]
            obs = _build_obs_dict(row, available_image_keys)
            all_obs.append(obs)

        # Query server in batches to get (z_rl, proprio, ref_chunk) for each step
        print(f"  Querying policy server in batches of {args.batch_size} ...")
        all_features: list[dict[str, Any]] = []
        for batch_start in range(0, n_steps, args.batch_size):
            batch_obs = all_obs[batch_start : batch_start + args.batch_size]
            batch_results = client.get_features_batch(batch_obs)
            all_features.extend(batch_results)
            if (batch_start // args.batch_size) % 10 == 0:
                print(f"    step {batch_start}/{n_steps} ...")

        assert len(all_features) == n_steps, (
            f"Expected {n_steps} feature results, got {len(all_features)}"
        )

        # Build RLTTransitions directly using the full ref_chunk from Machine A.
        #
        # WHY NOT use EpisodeStepRecord + build_chunk_transitions_from_episode():
        #   EpisodeStepRecord only stores ref_action (single frame, shape [action_dim]).
        #   build_chunk_transitions_from_episode() reconstructs ref_chunk by stacking
        #   N consecutive ref_action[0] values from N different timesteps — each
        #   independently predicted by the VLA. This loses intra-chunk temporal
        #   coherence: the VLA predicts a chunk as a single consistent trajectory,
        #   and ref_chunk[t:t+10] should come from one VLA call, not 10 separate ones.
        #
        # CORRECT approach: store the full ref_chunk [chunk_len, action_dim] returned
        # by Machine A at timestep t directly as the transition's ref_chunk, and use
        # the next timestep's ref_chunk as next_ref_chunk.
        chunk_len = chunk_len_to_use

        transitions: list[RLTTransition] = []
        for start in range(0, n_steps, stride_to_use):
            feat = all_features[start]
            end = min(start + chunk_len, n_steps)
            last_idx = end - 1
            next_idx = min(start + chunk_len, n_steps - 1)
            next_feat = all_features[next_idx]

            # action_chunk: stack dataset actions from [start, start+chunk_len)
            raw_actions = [
                np.asarray(df.iloc[j]["actions"], dtype=np.float32)[:action_dim]
                for j in range(start, end)
            ]
            # pad to chunk_len if episode ends before chunk is full
            while len(raw_actions) < chunk_len:
                raw_actions.append(raw_actions[-1].copy())
            action_chunk = np.stack(raw_actions, axis=0)  # [chunk_len, action_dim]

            # rewards: sparse — 0 for all steps except the last step of the episode
            raw_rewards = [
                float(args.reward) if (start + k) == n_steps - 1 else 0.0
                for k in range(end - start)
            ]
            # pad remaining slots with 0.0 (these are beyond episode end)
            while len(raw_rewards) < chunk_len:
                raw_rewards.append(0.0)
            rewards = np.array(raw_rewards, dtype=np.float32)

            is_done = bool(last_idx == n_steps - 1)
            source_chunk = np.full((chunk_len,), int(TransitionSource.BASE), dtype=np.uint8)

            transition = RLTTransition(
                z_rl=_get_z_rl(feat),
                proprio=_get_proprio(feat),
                ref_chunk=_get_ref_chunk(feat),          # full VLA chunk at t ✓
                action_chunk=action_chunk,
                rewards=rewards,
                done=is_done,
                next_z_rl=_get_z_rl(next_feat),
                next_proprio=_get_proprio(next_feat),
                next_ref_chunk=_get_ref_chunk(next_feat),  # full VLA chunk at t+chunk_len ✓
                source=int(TransitionSource.BASE),
                source_chunk=source_chunk,
                collection_phase="warmup",
                success=1,
                intervention_flag=False,
                episode_id=ep_idx,
                step_id=start,
            )
            transitions.append(transition)

        replay_manager.add_transitions(transitions)
        global_step_count += n_steps
        print(
            f"  Done. Built {len(transitions)} transitions "
            f"(chunk_len={chunk_len_to_use}, stride={stride_to_use})."
        )

    # ------------------------------------------------------------------
    # Close connection
    # ------------------------------------------------------------------
    client.close()
    print("[server] Connection closed.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    stats = replay_manager.stats()
    print("\n[replay] Journal summary:")
    print(f"  {stats}")
    print(f"\n[output] Replay journal written to: {output_path}")
    print("\nNext steps:")
    print("  # Offline train from the journal:")
    print(f"  python scripts/offline/offline_train_from_replay.py \\")
    print(f"      --replay-path {output_path} \\")
    print(f"      --steps 5000 \\")
    print(f"      --batch-size 128 \\")
    print(f"      --source all \\")
    print(f"      --phase all")


if __name__ == "__main__":
    main()
