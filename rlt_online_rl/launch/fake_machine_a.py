#!/usr/bin/env python3
"""
cd /path/to/rlt_online_rl
conda activate rlt_online_rl310
python launch/fake_machine_a.py
"""

from __future__ import annotations

import http
import http.server
import threading

import numpy as np
from openpi_client import msgpack_numpy
from websockets.datastructures import Headers
from websockets.http11 import Response
from websockets.sync.server import serve

HOST = "127.0.0.1"
PORT = 8000
Z_DIM = 2048
PROPRIO_DIM = 7
CHUNK_LEN = 50   # 与 online_rl.yaml 的 chunk_len 保持一致
ACTION_DIM = 7

# Small absolute-action nudge that does not return to the start pose,
# so reset behavior is easy to observe on the robot.
MOVE_OFFSET = np.array([0.02, 0.02, 0.02, 0.02, -0.02, -0.02, 0.0], dtype=np.float32)
MOVE_PROFILE = np.linspace(0.2, 1.0, CHUNK_LEN, dtype=np.float32)

packer = msgpack_numpy.Packer()


def build_payload(observation: dict) -> dict:
    state = np.asarray(observation.get("state", []), dtype=np.float32).reshape(-1)

    z_rl = np.zeros((Z_DIM,), dtype=np.float32)

    proprio = np.zeros((PROPRIO_DIM,), dtype=np.float32)
    if state.size > 0:
        n = min(PROPRIO_DIM, state.size)
        proprio[:n] = state[:n]

    base_action = np.zeros((ACTION_DIM,), dtype=np.float32)
    if state.size > 0:
        n = min(ACTION_DIM, state.size)
        base_action[:n] = state[:n]

    ref_chunk = np.repeat(base_action[None, :], CHUNK_LEN, axis=0)
    ref_chunk += MOVE_PROFILE[:, None] * MOVE_OFFSET[None, :]

    return {
        "z_rl": z_rl,
        "proprio": proprio,
        "ref_chunk": ref_chunk,
    }


def handler(ws) -> None:
    ws.send(packer.pack({"server": "fake-machine-a", "mode": "nudge-no-return"}))
    while True:
        try:
            raw = ws.recv()
        except Exception:
            return
        observation = msgpack_numpy.unpackb(raw)
        ws.send(packer.pack(build_payload(observation)))


class _HealthzHandler(http.server.BaseHTTPRequestHandler):
    """最小 HTTP 服务器：仅响应 GET /healthz → 200 OK。

    MachineAFeatureClient 在建立 WebSocket 前会先 HTTP GET /healthz，
    fake_machine_a 的 WebSocket 服务器不处理普通 HTTP，会返回 426。
    这个小服务器在同一 HOST 的不同端口（PORT+1）上监听，并在
    WebSocket 服务器端口上由 process_request 拦截 /healthz 请求。

    实际上，最简单的方案是：让 WebSocket serve() 的 process_request
    拦截 /healthz 并返回 200，无需额外端口。
    """

    def log_message(self, fmt, *args):
        pass  # 静默

    def do_GET(self):
        if self.path == "/healthz":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()


def _start_healthz_server() -> None:
    """在 PORT+1 上启动 HTTP /healthz 服务（后台线程）。"""
    server = http.server.HTTPServer((HOST, PORT + 1), _HealthzHandler)
    server.serve_forever()


def _process_request(connection, request):
    """拦截 /healthz 健康检查请求，直接返回 HTTP 200，不进行 WebSocket 握手。

    websockets 14+ 要求返回 Response 对象（而非旧版的 (status, headers, body) 元组）。
    """
    if request.path == "/healthz":
        return Response(
            status_code=200,
            reason_phrase="OK",
            headers=Headers([("Content-Type", "text/plain"), ("Content-Length", "3")]),
            body=b"ok\n",
        )
    return None


def main() -> None:
    print(f"[fake_a] serving ws://{HOST}:{PORT}  healthz=http://{HOST}:{PORT}/healthz", flush=True)
    with serve(
        handler,
        HOST,
        PORT,
        max_size=None,
        compression=None,
        process_request=_process_request,
    ) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
