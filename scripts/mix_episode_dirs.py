#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


DEFAULT_SOURCES = [
    Path("/home/lry/RokaeDual"),
    Path("/home/lry/RokaeDual-five"),
]
DEFAULT_OUTPUT = Path("/home/lry/mixed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mix episode_* directories from multiple source directories, shuffle them, "
            "and copy them into a new directory with consecutive episode names."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        dest="sources",
        help=(
            "Source directory containing episode_* subdirectories. Can be passed multiple "
            "times. Defaults to /home/lry/RokaeDual and /home/lry/RokaeDual-five."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for mixed episodes. Default: /home/lry/mixed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Set this to reproduce the same shuffled order.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First output episode index. Default: 0.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Zero-pad output episode numbers to this width. Example: --width 6 -> episode_000000.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned copy order without creating or copying files.",
    )
    return parser.parse_args()


def _episode_sort_key(path: Path) -> tuple[int, int | str]:
    suffix = path.name.removeprefix("episode_")
    try:
        return (0, int(suffix))
    except ValueError:
        return (1, suffix)


def _find_episodes(source_dirs: list[Path]) -> list[Path]:
    episodes: list[Path] = []
    for source_dir in source_dirs:
        source_dir = source_dir.expanduser().resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
        episodes.extend(sorted(source_dir.glob("episode_*"), key=_episode_sort_key))

    episodes = [path for path in episodes if path.is_dir()]
    if not episodes:
        sources = ", ".join(str(path) for path in source_dirs)
        raise RuntimeError(f"No episode_* directories found in: {sources}")
    return episodes


def _episode_name(index: int, width: int) -> str:
    if width > 0:
        return f"episode_{index:0{width}d}"
    return f"episode_{index}"


def _prepare_output(output_dir: Path, clear_output: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if output_dir.exists():
        if not clear_output:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Pass --clear-output to replace it, or choose a different --output."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def _write_manifest(output_dir: Path, manifest: list[dict[str, object]]) -> None:
    manifest_path = output_dir / "mix_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> int:
    args = _parse_args()
    source_dirs = args.sources or DEFAULT_SOURCES
    output_dir = args.output.expanduser().resolve()

    episodes = _find_episodes(source_dirs)
    rng = random.Random(args.seed)
    rng.shuffle(episodes)

    print(f"Found {len(episodes)} episodes from {len(source_dirs)} source directories.")
    print(f"Output: {output_dir}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")

    _prepare_output(output_dir, clear_output=args.clear_output, dry_run=args.dry_run)

    manifest: list[dict[str, object]] = []
    for offset, src in enumerate(episodes):
        output_index = args.start_index + offset
        dst_name = _episode_name(output_index, args.width)
        dst = output_dir / dst_name
        record = {
            "new_episode": dst_name,
            "new_index": output_index,
            "source_episode": src.name,
            "source_path": str(src),
            "source_root": str(src.parent),
        }
        manifest.append(record)

        if args.dry_run:
            print(f"{dst_name} <- {src}")
            continue
        shutil.copytree(src, dst)

    if not args.dry_run:
        _write_manifest(output_dir, manifest)
        print(f"Copied {len(manifest)} episodes.")
        print(f"Manifest: {output_dir / 'mix_manifest.json'}")
    else:
        print("Dry run only; no files were copied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
