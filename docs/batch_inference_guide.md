# Machine A Server Batch Inference Guide

## Overview

Server (`serve_rlt_policy.py`) supports both single and batch inference. Machine B uses fixed micro-batches for dense replay (`step_trace_stride > 0`), which keeps JAX on pre-warmed shapes while reducing WebSocket round-trips. Plain chunk replay (`step_trace_stride = 0`) keeps on-demand single fetches to avoid padding small tail requests.

## For Machine B (RL Client Side)

### You don't need to change runtime code

`inference.py` has been updated. `_build_replay_transitions` automatically uses the batch path for dense replay only. The tuning knob is:

```yaml
env_driver:
  replay_feature_batch_size: 16
```

Just confirm two things:

### 1. Confirm Server Version

Make sure server is started with the latest `serve_rlt_policy.py`. After connecting, metadata should contain:

```python
{'has_rl_token': True, 'supports_batch': True, ...}
```

### 2. Behavior Change

**Before (sequential):**

```
episode ends → _build_replay_transitions
  → each window requests Machine A one by one (N round-trips)
  → total time: N × 70ms
```

**After (dense replay batch, automatic):**

```
episode ends → _build_replay_transitions
  → if step_trace_stride > 0, internally calls _prefetch_features_batch
  → collects all uncached observations
  → sends fixed micro-batches to Machine A, default size=16
  → total time: ceil(N / 16) batch requests
```

For `step_trace_stride = 0`, replay finalize skips batch prefetch and fetches only missing anchors on demand with single `get_features()` calls.

### 3. What Is NOT Affected

- ✅ Real-time rollout (`get_features` single request per chunk) — unchanged
- ✅ actor_service — unchanged
- ✅ learner_service — unchanged
- ✅ Replay format — unchanged
- ✅ Keyboard control — unchanged
- ✅ If server doesn't support batch, automatically falls back to sequential

### 4. New Log Output

With `step_trace_stride > 0`, after each episode ends you will see additional log lines:

```
Prefetching 37 features via batch request (cached=8 micro_batch_size=16)
Batch prefetch chunk 1 size=16 done in 1430.2ms
Batch prefetch chunk 2 size=16 done in 1421.7ms
Batch prefetch chunk 3 size=5 done in 734.5ms
Batch prefetch done: 37 features in 3586.4ms (96.9ms/sample, requests=3)
```

## Protocol Specification

### Single Request (unchanged)

```
Client sends:
{
    "images": {
        "base_0_rgb": ndarray (224, 224, 3) uint8,
        "left_wrist_0_rgb": ndarray (224, 224, 3) uint8,
        "right_wrist_0_rgb": ndarray (224, 224, 3) uint8
    },
    "state": ndarray (7,) float32,
    "prompt": "task description"
}

Server returns:
{
    "z_rl": ndarray (2048,) float32,
    "proprio": ndarray (7,) float32,
    "ref_chunk": ndarray (50, 7) float32,
    "policy_timing": {"infer_ms": float},
    "_raw_actions": ndarray (50, 7) float32,
    "_raw_rl_token": ndarray (1, 2048) float32
}
```

### Batch Request (new)

```
Client sends:
{
    "batch": [
        {"images": {...}, "state": (7,), "prompt": "..."},
        {"images": {...}, "state": (7,), "prompt": "..."},
        ...
    ]
}

Server returns:
{
    "batch_results": [
        {"z_rl": (2048,), "proprio": (7,), "ref_chunk": (50,7), ...},
        {"z_rl": (2048,), "proprio": (7,), "ref_chunk": (50,7), ...},
        ...
    ],
    "batch_size": int,           # actual number of observations
    "padded_size": int,          # padded to pre-compiled size
    "total_infer_ms": float,     # total inference time
    "per_sample_infer_ms": float # amortized per sample
}
```

## Server Side Details

### Starting the Server

```bash
cd /path/to/mt-fvla
python scripts/serve_rlt_policy.py \
    --config rlt_pi05_agilexbag_image_delta_joint \
    --checkpoint-dir checkpoints/.../params \
    --port 8000
```

On startup, the server:
1. Loads model and checkpoint
2. Warms up JIT compilation for pre-compiled batch sizes (1, 2, 4, 6, 8, 10, 12, 16)
3. Starts WebSocket server

### Batch Size Padding

To avoid JIT recompilation for unseen batch sizes, the server pads to the nearest pre-compiled size:

| Requested | Padded To | JIT Compiled |
|-----------|-----------|-------------|
| 1         | 1         | ✅ (warmup) |
| 2         | 2         | ✅ (warmup) |
| 3         | 4         | ✅ (warmup) |
| 4         | 4         | ✅ (warmup) |
| 5-6       | 6         | ✅ (warmup) |
| 7-8       | 8         | ✅ (warmup) |
| 9-10      | 10        | ✅ (warmup) |
| 11-12     | 12        | ✅ (warmup) |
| 13-16     | 16        | ✅ (warmup) |
| 17+       | as-is     | ⚠️ First call triggers JIT (~30s) |

Padding is transparent to the client — server returns exactly the number of results requested (padding results are discarded).

Machine B avoids `17+` requests during dense replay finalize by splitting missing anchors into fixed micro-batches, default `16`.

### Pre-compiled Batch Sizes

Default: `[1, 2, 4, 6, 8, 10, 12, 16]`

`serve_rlt_policy.py` uses `RLTPolicy.COMPILED_BATCH_SIZES` for both padding and startup warmup, so the list has one source of truth:

```python
COMPILED_BATCH_SIZES = [1, 2, 4, 6, 8, 10, 12, 16]
```

Larger batch sizes require more GPU memory and do not necessarily reduce total latency. On the local RTX 4090 benchmark for `rlt_pi05_agilexbag_image_delta_joint`, cached forward times were approximately:

- batch_size=1: 113ms
- batch_size=8: 733ms
- batch_size=16: 1431ms
- batch_size=32: 3120ms

Batch size 52 was the measured local limit, while 53 failed with OOM. For online replay finalize, prefer `16` first and drop to `8` if the Machine A GPU is under pressure.

### WebSocket Configuration

Ping timeout is disabled to prevent disconnection during JIT compilation:

```python
# websocket_policy_server.py
ping_interval=None,
ping_timeout=None,
```

## Performance

### Benchmark (local RTX 4090)

| Mode | N samples | Total Time | Per Sample |
|------|-----------|-----------|-----------|
| Sequential | 16 | ~1806ms | ~113ms |
| Batch (N=8) | 8 | ~733ms | ~92ms |
| Batch (N=16) | 16 | ~1431ms | ~89ms |
| Batch (N=32) | 32 | ~3120ms | ~97ms |

### Replay Build Scenario

Typical dense episode: 100 steps, stride=2, chunk_len=10 → ~45 windows → about 25 uncached observation anchors after cache reuse.

| Mode | Time |
|------|------|
| Sequential | 25 × 113ms ≈ 2825ms, plus RPC overhead |
| Fixed micro-batch=16 | 16 + 9 anchors ≈ 1431ms + 733-1431ms |
| **Expected speedup** | **modest, usually ~15-30% depending on network overhead** |

The main benefit is stability in dense replay: replay finalize no longer sends dynamic batch sizes such as 23, 37, or 41 to JAX, so it avoids repeated first-call JIT pauses for episode-dependent shapes. Chunk replay (`step_trace_stride=0`) usually has only a few missing anchors, so it stays on-demand single to avoid padding overhead.

## Files Changed

| File | Change |
|------|--------|
| `scripts/serve_rlt_policy.py` | Added `_infer_batch`, batch padding, JIT warmup |
| `src/openpi/serving/websocket_policy_server.py` | Disabled ping timeout |
| `rlt_online_rl/src/rlt_online_rl/inference.py` | Added `get_features_batch`, micro-batched `_prefetch_features_batch` |
| `rlt_online_rl/src/rlt_online_rl/config.py` | Added `env_driver.replay_feature_batch_size` |

## Testing

```bash
# Real server test
python scripts/test_batch_padding.py --host localhost --port 8000

# Batch size sweep
python scripts/test_rlt_batch.py --host localhost --port 8000 --batch-size 16
```
