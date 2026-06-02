#!/usr/bin/env python3
import argparse
import collections
import dataclasses
import functools
import io
import json
import os
import pickle
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from PIL import Image


ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "replay_viewer.html"


def find_repo_root(start):
    for path in (start, *start.parents):
        if (path / "rlt_online_rl" / "src").exists() and (path / "src").exists():
            return path
    raise RuntimeError(f"Could not locate repo root from {start}")


def validate_data_root(path):
    root = Path(path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    journal = root / "replay_journal.pkl"
    episodes = root / "episodes"
    if not journal.exists():
        raise FileNotFoundError(f"Missing replay_journal.pkl under {root}")
    if not episodes.exists() or not episodes.is_dir():
        raise FileNotFoundError(f"Missing episodes directory under {root}")
    return root


def default_data_root(script_root, repo_root):
    candidates = []
    env_dir = os.environ.get("REPLAY_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            Path.cwd(),
            script_root,
            repo_root / "rlt_online_rl" / "runs" / "dobot_umi" / "replay",
        ]
    )
    for candidate in candidates:
        try:
            return validate_data_root(candidate)
        except Exception:
            continue
    return Path.cwd().resolve()


REPO_ROOT = find_repo_root(ROOT)
for p in (REPO_ROOT / "rlt_online_rl" / "src", REPO_ROOT / "src", REPO_ROOT):
    sys.path.insert(0, str(p))

app = Flask(__name__)
DATA_LOCK = threading.Lock()
DATA_ROOT = default_data_root(ROOT, REPO_ROOT)
IMPORT_ROOT = Path(tempfile.gettempdir()) / "visualize_pkl_imports"
IMPORT_ROOT.mkdir(parents=True, exist_ok=True)
IMPORT_RETENTION_SECONDS = 24 * 60 * 60


def cleanup_import_dirs(*, keep_root: Path | None = None) -> None:
    now = time.time()
    for child in IMPORT_ROOT.glob("replay_*"):
        try:
            if keep_root is not None and child.resolve() == keep_root.resolve():
                continue
            if not child.is_dir():
                continue
            age = now - child.stat().st_mtime
            if age > IMPORT_RETENTION_SECONDS:
                shutil.rmtree(child, ignore_errors=True)
        except Exception:
            continue


def set_data_root(path):
    global DATA_ROOT
    keep_root = None
    try:
        keep_root = DATA_ROOT.resolve()
    except Exception:
        keep_root = None
    cleanup_import_dirs(keep_root=keep_root)
    with DATA_LOCK:
        DATA_ROOT = validate_data_root(path)
        load_journal.cache_clear()
        load_journal_episode_groups.cache_clear()
        load_episode.cache_clear()
        get_episode_image_bytes.cache_clear()
        return DATA_ROOT


def get_data_root():
    with DATA_LOCK:
        return DATA_ROOT


def natural_episode_key(path):
    parts = path.stem.split("_")
    try:
        return (int(parts[1]), int(parts[2]))
    except Exception:
        return (10**9, path.name)


def iter_pickle_stream(path):
    with path.open("rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


@functools.lru_cache(maxsize=1)
def load_journal():
    return list(iter_pickle_stream(get_data_root() / "replay_journal.pkl"))


@functools.lru_cache(maxsize=2)
def load_episode(name):
    path = safe_episode_path(name)
    with path.open("rb") as f:
        return pickle.load(f)


def safe_episode_path(name):
    base = (get_data_root() / "episodes").resolve()
    path = (base / name).resolve()
    if not str(path).startswith(str(base)) or path.suffix != ".pkl" or not path.exists():
        raise FileNotFoundError(name)
    return path


def object_fields(obj):
    if dataclasses.is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return None


def scalar_value(x):
    if isinstance(x, np.ndarray) and x.shape == ():
        return scalar_value(x.item())
    if isinstance(x, np.generic):
        return scalar_value(x.item())
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    return None


def array_stats(arr):
    stats = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
    }
    if arr.size and np.issubdtype(arr.dtype, np.number):
        finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
        if finite.size:
            stats.update(
                {
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                }
            )
    return stats


def array_preview(arr, limit=3000):
    if arr.size <= limit:
        return arr.tolist()
    flat = arr.reshape(-1)
    head = flat[:limit].tolist()
    return {"truncated": True, "shown": limit, "total": int(arr.size), "flat_head": head}


def serialize(obj, depth=0, max_depth=6, include_arrays=True):
    if depth > max_depth:
        return {"type": type(obj).__name__, "value": "..."}
    if isinstance(obj, np.ndarray):
        out = {"kind": "ndarray", **array_stats(obj)}
        if include_arrays:
            out["values"] = array_preview(obj)
        return out
    if isinstance(obj, np.generic):
        return serialize(obj.item(), depth, max_depth, include_arrays)
    if isinstance(obj, dict):
        return {
            "kind": "dict",
            "len": len(obj),
            "items": {str(k): serialize(v, depth + 1, max_depth, include_arrays) for k, v in obj.items()},
        }
    if isinstance(obj, (list, tuple)):
        return {
            "kind": type(obj).__name__,
            "len": len(obj),
            "items": [serialize(v, depth + 1, max_depth, include_arrays) for v in list(obj)],
        }
    fields = object_fields(obj)
    if fields is not None:
        return {
            "kind": type(obj).__module__ + "." + type(obj).__name__,
            "fields": {k: serialize(v, depth + 1, max_depth, include_arrays) for k, v in fields.items()},
        }
    val = scalar_value(obj)
    if val is not None:
        return {"kind": type(obj).__name__, "value": val}
    return {"kind": type(obj).__name__, "value": repr(obj)}


def short_structure(obj, depth=0, max_depth=5):
    if isinstance(obj, np.ndarray):
        return {"kind": "ndarray", **array_stats(obj)}
    if depth >= max_depth:
        return {"kind": type(obj).__name__}
    if isinstance(obj, dict):
        return {
            "kind": "dict",
            "len": len(obj),
            "items": {str(k): short_structure(v, depth + 1, max_depth) for k, v in list(obj.items())[:40]},
        }
    if isinstance(obj, (list, tuple)):
        return {
            "kind": type(obj).__name__,
            "len": len(obj),
            "sample": [short_structure(v, depth + 1, max_depth) for v in list(obj)[:2]],
        }
    fields = object_fields(obj)
    if fields is not None:
        return {
            "kind": type(obj).__module__ + "." + type(obj).__name__,
            "fields": {k: short_structure(v, depth + 1, max_depth) for k, v in fields.items()},
        }
    val = scalar_value(obj)
    if val is not None:
        return {"kind": type(obj).__name__, "value": val}
    return {"kind": type(obj).__name__, "value": repr(obj)[:160]}


def make_curve(name, points, *, default=False, kind="line"):
    return {
        "name": name,
        "points": points,
        "count": sum(1 for point in points if not point.get("break")),
        "default": bool(default),
        "kind": kind,
    }


SOURCE_LABELS = {
    0: "BASE",
    1: "RL",
    2: "HUMAN",
    3: "MIXED",
}


JOURNAL_FIELD_DESCRIPTIONS = {
    "z_rl": "当前时刻的 RL token / latent 表征，通常作为策略输入，shape=(2048,)。",
    "proprio": "当前时刻的本体状态向量，当前数据里是 7 维关节/位姿相关状态。",
    "ref_chunk": "当前时刻的参考动作 chunk，shape=(50, 7)。表示从当前步开始的 50 步参考轨迹。",
    "action_chunk": "当前时刻实际行为 chunk，shape=(50, 7)。在线阶段可能来自 RL、人类干预或混合源。",
    "rewards": "与 action_chunk 对齐的 50 步 reward 序列。",
    "done": "这条 transition 是否已经到达 episode 终止边界。",
    "next_z_rl": "next state 对应的 RL token / latent 表征。",
    "next_proprio": "next state 对应的本体状态向量。",
    "next_ref_chunk": "next state 对应的参考动作 chunk，通常等价于下一锚点看到的 50 步参考轨迹。",
    "source": "这条 transition 的主来源。0=BASE, 1=RL, 2=HUMAN, 3=MIXED。",
    "source_chunk": "chunk 内每一步动作来源，长度 50。用它可以看一个 chunk 内是否混入人工接管。",
    "collection_phase_id": "collection_phase 的数值编码。0=unknown, 1=warmup, 2=online。",
    "success": "是否成功，通常终止附近才会变成 1。",
    "intervention_flag": "当前 transition 是否发生过人工干预。",
    "episode_id": "该 transition 所属 episode 编号。",
    "step_id": "该 transition 的起始 step 编号。当前数据大多按 chunk stride 前进。",
    "collection_phase": "采集阶段字符串，如 warmup / online。",
}


def curve_from_vector(name, arr):
    values = np.asarray(arr, dtype=np.float32).reshape(-1)
    return {
        "name": name,
        "points": [{"x": int(i), "y": float(v)} for i, v in enumerate(values)],
        "count": int(values.shape[0]),
    }


def journal_transition_curves(record):
    curves = []
    for key in ("ref_chunk", "action_chunk", "next_ref_chunk"):
        arr = np.asarray(record.get(key))
        if arr.ndim == 2 and arr.shape[1] <= 16:
            for dim in range(arr.shape[1]):
                curves.append(curve_from_vector(f"{key}[:, {dim}]", arr[:, dim]))
    for key in ("rewards", "source_chunk", "proprio", "next_proprio"):
        if key in record:
            arr = np.asarray(record[key])
            if arr.ndim == 1 and arr.size <= 64:
                curves.append(curve_from_vector(key, arr))
    return curves


def scalar_int(value):
    return int(np.asarray(value).item())


def scalar_bool(value):
    return bool(np.asarray(value).item())


@functools.lru_cache(maxsize=1)
def load_journal_episode_groups():
    groups = collections.defaultdict(list)
    for journal_idx, record in enumerate(load_journal()):
        episode_id = scalar_int(record.get("episode_id", 0))
        groups[episode_id].append({"journal_idx": journal_idx, "record": record})
    for rows in groups.values():
        rows.sort(key=lambda item: scalar_int(item["record"].get("step_id", 0)))
    return dict(groups)


def journal_episode_timeline_len(rows):
    if not rows:
        return 0
    return max(
        scalar_int(item["record"].get("step_id", 0)) + int(np.asarray(item["record"]["ref_chunk"]).shape[0])
        for item in rows
    )


def segmented_chunk_curve(rows, field, dim):
    points = []
    for item in rows:
        record = item["record"]
        start = scalar_int(record.get("step_id", 0))
        chunk_values = np.asarray(record[field], dtype=np.float32)[:, dim]
        points.extend({"x": int(start + offset), "y": float(value)} for offset, value in enumerate(chunk_values))
        points.append({"break": True})
    return make_curve(f"{field}[:, {dim}]", points, default=False, kind="raw_segment")


def segmented_horizon_curve(rows, field):
    points = []
    for item in rows:
        record = item["record"]
        start = scalar_int(record.get("step_id", 0))
        chunk_values = np.asarray(record[field], dtype=np.float32).reshape(-1)
        points.extend({"x": int(start + offset), "y": float(value)} for offset, value in enumerate(chunk_values))
        points.append({"break": True})
    return make_curve(field, points, default=False, kind="raw_segment")


def aligned_step_curve(rows, field, element_idx=None):
    points = []
    for item in rows:
        record = item["record"]
        x = scalar_int(record.get("step_id", 0))
        arr = np.asarray(record[field])
        if arr.shape == ():
            y = float(np.asarray(arr).item())
        else:
            y = float(np.asarray(arr, dtype=np.float32).reshape(-1)[element_idx])
        points.append({"x": x, "y": y})
    name = field if element_idx is None else f"{field}[{element_idx}]"
    return make_curve(name, points, default=False, kind="anchor")


def build_journal_episode_curves(rows):
    curves = []
    if not rows:
        return curves
    sample = rows[0]["record"]
    for field in ("ref_chunk", "action_chunk", "next_ref_chunk"):
        arr = np.asarray(sample[field])
        if arr.ndim == 2 and arr.shape[1] <= 16:
            for dim in range(arr.shape[1]):
                curves.append(segmented_chunk_curve(rows, field, dim))
    for field in ("rewards", "source_chunk"):
        arr = np.asarray(sample[field])
        if arr.ndim == 1:
            curves.append(segmented_horizon_curve(rows, field))
    for field in ("proprio", "next_proprio"):
        arr = np.asarray(sample[field])
        if arr.ndim == 1 and arr.size <= 16:
            for dim in range(arr.size):
                curves.append(aligned_step_curve(rows, field, dim))
    for field in ("source", "success", "intervention_flag", "done"):
        curves.append(aligned_step_curve(rows, field))
    return curves


def vector_step_curves(prefix, rows, attr, *, default_prefixes=()):
    curves = []
    if not rows:
        return curves
    first = np.asarray(rows[0], dtype=np.float32)
    if first.ndim != 1 or first.size > 32:
        return curves
    stacked = np.asarray(rows, dtype=np.float32)
    for dim in range(stacked.shape[1]):
        points = [{"x": int(i), "y": float(v)} for i, v in enumerate(stacked[:, dim])]
        name = f"{prefix}.{attr}[{dim}]"
        curves.append(make_curve(name, points, default=any(name.startswith(p) for p in default_prefixes), kind="continuous"))
    return curves


def scalar_series_curve(name, values, *, default=False):
    points = [{"x": int(i), "y": float(v)} for i, v in enumerate(values)]
    return make_curve(name, points, default=default, kind="continuous")


def build_episode_curves(observations, steps):
    curves = []
    obs_states = [obs.get("state") for obs in observations if isinstance(obs, dict) and "state" in obs]
    if obs_states:
        curves.extend(vector_step_curves("obs", obs_states, "state", default_prefixes=()))
    step_actions = [step.action for step in steps if hasattr(step, "action")]
    step_refs = [step.ref_action for step in steps if hasattr(step, "ref_action")]
    if step_actions:
        curves.extend(vector_step_curves("step", step_actions, "action", default_prefixes=("step.action",)))
    if step_refs:
        curves.extend(vector_step_curves("step", step_refs, "ref_action", default_prefixes=("step.ref_action",)))
    if steps:
        curves.append(scalar_series_curve("step.reward", [float(step.reward) for step in steps], default=False))
        curves.append(scalar_series_curve("step.source", [float(step.source) for step in steps], default=False))
        curves.append(scalar_series_curve("step.done", [float(step.done) for step in steps], default=False))
        curves.append(scalar_series_curve("step.intervention_flag", [float(step.intervention_flag) for step in steps], default=False))
    return curves


def journal_episode_summary(episode_id, rows):
    starts = [scalar_int(item["record"].get("step_id", 0)) for item in rows]
    phases = sorted({str(item["record"].get("collection_phase", "unknown")) for item in rows})
    sources = sorted({SOURCE_LABELS.get(scalar_int(item["record"].get("source", 0)), str(scalar_int(item["record"].get("source", 0)))) for item in rows})
    return {
        "episode_id": episode_id,
        "transition_count": len(rows),
        "timeline_len": journal_episode_timeline_len(rows),
        "step_min": min(starts) if starts else 0,
        "step_max": max(starts) if starts else 0,
        "phases": phases,
        "sources": sources,
    }


def journal_transition_meta(record):
    source = int(np.asarray(record.get("source", 0)).item())
    phase_id = int(np.asarray(record.get("collection_phase_id", 0)).item())
    return {
        "source_label": SOURCE_LABELS.get(source, str(source)),
        "collection_phase_id": phase_id,
        "field_descriptions": JOURNAL_FIELD_DESCRIPTIONS,
    }


def episode_to_dict(ep):
    return object_fields(ep) or {}


@app.get("/")
def index():
    return send_file(HTML_PATH)


@app.get("/api/current-root")
def current_root():
    return jsonify({"root": str(get_data_root())})


@app.post("/api/set-data-root")
def set_data_root_api():
    payload = request.get_json(silent=True) or {}
    root = payload.get("path", "")
    try:
        active = set_data_root(root)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "root": str(active)})


@app.post("/api/import-picked-root")
def import_picked_root_api():
    replay_journal = request.files.get("replay_journal")
    episode_files = request.files.getlist("episode_files")
    if replay_journal is None:
        return jsonify({"ok": False, "error": "Missing replay_journal.pkl from selected directory"}), 400
    if not episode_files:
        return jsonify({"ok": False, "error": "Missing episodes/*.pkl from selected directory"}), 400

    import_dir = IMPORT_ROOT / f"replay_{int(time.time() * 1000)}"
    episodes_dir = import_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    try:
        replay_journal.save(import_dir / "replay_journal.pkl")
        for episode_file in episode_files:
            name = Path(episode_file.filename or "").name
            if not name.endswith(".pkl"):
                continue
            episode_file.save(episodes_dir / name)
        active = set_data_root(import_dir)
    except Exception:
        shutil.rmtree(import_dir, ignore_errors=True)
        raise
    return jsonify({"ok": True, "root": str(active), "imported_episode_count": len(list(episodes_dir.glob('*.pkl')))})


@app.get("/api/summary")
def summary():
    data_root = get_data_root()
    eps = []
    for p in sorted((data_root / "episodes").glob("*.pkl"), key=natural_episode_key):
        eps.append({"name": p.name, "size_mb": round(p.stat().st_size / 1024 / 1024, 2), "mtime": p.stat().st_mtime})
    return jsonify(
        {
            "root": str(data_root),
            "journal_size_mb": round((data_root / "replay_journal.pkl").stat().st_size / 1024 / 1024, 2),
            "journal_count": len(load_journal()),
            "journal_episode_count": len(load_journal_episode_groups()),
            "episodes": eps,
        }
    )


@app.get("/api/journal/overview")
def journal_overview():
    items = load_journal()
    grouped = load_journal_episode_groups()
    return jsonify(
        {
            "count": len(items),
            "structure": short_structure(items[0] if items else None),
            "journal_episodes": [journal_episode_summary(ep, rows) for ep, rows in sorted(grouped.items())],
            "field_descriptions": JOURNAL_FIELD_DESCRIPTIONS,
        }
    )


@app.get("/api/journal/episode/<int:episode_id>/overview")
def journal_episode_overview(episode_id):
    grouped = load_journal_episode_groups()
    rows = grouped.get(episode_id, [])
    if not rows:
        return jsonify({"error": f"episode_id {episode_id} not found in replay_journal.pkl"}), 404
    return jsonify(
        {
            "episode_id": episode_id,
            "timeline_len": journal_episode_timeline_len(rows),
            "transition_count": len(rows),
            "summary": journal_episode_summary(episode_id, rows),
            "curves": build_journal_episode_curves(rows),
            "first_transition": scalar_int(rows[0]["journal_idx"]),
            "last_transition": scalar_int(rows[-1]["journal_idx"]),
        }
    )


@app.get("/api/journal/episode/<int:episode_id>/frame/<int:x>")
def journal_episode_frame(episode_id, x):
    grouped = load_journal_episode_groups()
    rows = grouped.get(episode_id, [])
    if not rows:
        return jsonify({"error": f"episode_id {episode_id} not found in replay_journal.pkl"}), 404
    timeline_len = journal_episode_timeline_len(rows)
    x = max(0, min(x, max(0, timeline_len - 1)))
    active = []
    nearest = None
    nearest_dist = None
    for item in rows:
        record = item["record"]
        start = scalar_int(record.get("step_id", 0))
        chunk_len = int(np.asarray(record["ref_chunk"]).shape[0])
        stop = start + chunk_len
        dist = 0 if start <= x < stop else min(abs(x - start), abs(x - (stop - 1)))
        if nearest is None or dist < nearest_dist:
            nearest = item
            nearest_dist = dist
        if start <= x < stop:
            active.append(item)
    record = (nearest or rows[0])["record"]
    return jsonify(
        {
            "episode_id": episode_id,
            "x": x,
            "count": timeline_len,
            "active_transition_count": len(active),
            "active_transition_indices": [scalar_int(item["journal_idx"]) for item in active[:12]],
            "nearest_transition_index": scalar_int((nearest or rows[0])["journal_idx"]),
            "nearest_transition_step_id": scalar_int(record.get("step_id", 0)),
            "data": serialize(record),
            "meta": journal_transition_meta(record),
        }
    )


@app.get("/api/episode/<name>/overview")
def episode_overview(name):
    ep = load_episode(name)
    data = episode_to_dict(ep)
    observations = data.get("observations", [])
    steps = data.get("steps", [])
    chunks = data.get("chunks", [])
    return jsonify(
        {
            "name": name,
            "count": len(observations),
            "steps": len(steps),
            "chunks": len(chunks),
            "summary": serialize(data.get("summary", {}), include_arrays=False),
            "structure": short_structure(ep),
            "curves": build_episode_curves(observations, steps),
        }
    )


@app.get("/api/episode/<name>/frame/<int:idx>")
def episode_frame(name, idx):
    ep = load_episode(name)
    data = episode_to_dict(ep)
    observations = data.get("observations", [])
    steps = data.get("steps", [])
    chunks = data.get("chunks", [])
    idx = max(0, min(idx, len(observations) - 1))
    obs = observations[idx] if observations else {}
    obs_out = {}
    cams = []
    if isinstance(obs, dict):
        for k, v in obs.items():
            if k == "images" and isinstance(v, dict):
                cams = list(v.keys())
                obs_out[k] = {"kind": "images", "cameras": cams}
            else:
                obs_out[k] = serialize(v)
    step = None
    if idx < len(steps):
        step = serialize(steps[idx])
    active_chunks = []
    for c in chunks:
        fields = object_fields(c) or {}
        if fields.get("step_start", -1) <= idx < fields.get("step_stop", -1):
            active_chunks.append(serialize(c))
    return jsonify({"idx": idx, "count": len(observations), "cameras": cams, "observation": obs_out, "step": step, "chunks": active_chunks})


@functools.lru_cache(maxsize=256)
def get_episode_image_bytes(name, idx, camera):
    ep = load_episode(name)
    observations = episode_to_dict(ep).get("observations", [])
    if not observations:
        raise FileNotFoundError(f"No observations in {name}")
    idx = max(0, min(int(idx), len(observations) - 1))
    obs = observations[idx]
    arr = obs["images"][camera]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


@app.get("/api/episode/<name>/image/<int:idx>/<camera>.png")
def episode_image(name, idx, camera):
    return Response(get_episode_image_bytes(name, idx, camera), mimetype="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/search")
def search():
    mode = request.args.get("mode", "journal")
    query = request.args.get("q", "").lower()
    idx = int(request.args.get("idx", "0"))
    if mode == "journal":
        text = json.dumps(serialize(load_journal()[idx], include_arrays=False), ensure_ascii=False).lower()
    else:
        name = request.args["name"]
        text = json.dumps(episode_frame(name, idx).json, ensure_ascii=False).lower()
    return jsonify({"match": query in text if query else True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_ROOT))
    args = parser.parse_args()
    set_data_root(args.data_dir)
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="127.0.0.1", port=port, threaded=True)
