"""
server.py — FastAPI WebSocket server.

Each connected iOS client gets its own agent instance.
Multiple clients can connect simultaneously (e.g. for testing).

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Connect from iOS:
    ws://<your-machine-ip>:8000/ws
"""

from __future__ import annotations

import base64
import io
import json
import time
from typing import Dict

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image

from agent import IMAGE_H, IMAGE_W, RLAgent

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="RL Agent Server")

# One agent per connected client (keyed by WebSocket id)
_agents: Dict[int, RLAgent] = {}


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client_id = id(ws)
    agent = RLAgent()
    _agents[client_id] = agent
    print(f"[server] client {client_id} connected")

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            msg_type = msg.get("type")

            # ── Incoming frame ────────────────────────────────────────────────
            if msg_type == "frame":
                frame = decode_frame(msg["data"])
                if frame is None:
                    continue

                action_idx, log_prob, action_label = agent.select_action(frame)

                response = {
                    "type": "action",
                    "action": action_label,          # human-readable label for iOS
                    "value": [float(action_idx)],    # raw index if iOS needs it
                    "step": agent.stats["steps"],
                }
                await ws.send_text(json.dumps(response))

            # ── Human reward ──────────────────────────────────────────────────
            elif msg_type == "reward":
                value = float(msg.get("value", 0.0))
                agent.record_reward(value)
                print(f"[server] reward {value:+.1f} received | "
                      f"stats: {agent.stats}")

            else:
                print(f"[server] unknown message type: {msg_type}")

    except WebSocketDisconnect:
        print(f"[server] client {client_id} disconnected")
    except Exception as e:
        print(f"[server] error for client {client_id}: {e}")
    finally:
        _agents.pop(client_id, None)


# ── HTTP health check ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "connected_clients": len(_agents),
        "timestamp": time.time(),
    }


@app.get("/stats")
def stats():
    return {
        client_id: agent.stats
        for client_id, agent in _agents.items()
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def decode_frame(b64_string: str) -> np.ndarray | None:
    """
    Decode a base64 JPEG string from the iOS client into a
    uint8 numpy array of shape (IMAGE_H, IMAGE_W, 3).
    """
    try:
        raw_bytes = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        # Ensure the image matches the agent's expected input size.
        # The iOS app already resizes to 128x128, but this guards against
        # any mismatch.
        if image.size != (IMAGE_W, IMAGE_H):
            image = image.resize((IMAGE_W, IMAGE_H), Image.BILINEAR)

        return np.array(image, dtype=np.uint8)
    except Exception as e:
        print(f"[server] frame decode error: {e}")
        return None
