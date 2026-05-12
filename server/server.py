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
from policy import PolicyAgent
import vqvae


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="RL Agent Server")

# One generator
gen   = vqvae.VQVAEGenerator(num_codes=64)

# One agent per connected client (keyed by WebSocket id)
_agents: Dict[int, PolicyAgent] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────
def decode_frame(b64_string: str) -> np.ndarray | None:
    """
    Decode a base64 JPEG string from the iOS client into a
    uint8 numpy array of shape (IMAGE_H, IMAGE_W, 1).
    """
    try:
        raw_bytes = base64.b64decode(b64_string)
        # Convert to "L" for 8-bit grayscale (Luminance)
        image = Image.open(io.BytesIO(raw_bytes)).convert("L")

        if image.size != (IMAGE_W, IMAGE_H):
            image = image.resize((IMAGE_W, IMAGE_H), Image.BILINEAR)

        # Convert to numpy array
        frame = np.array(image, dtype=np.uint8)
        
        # RL models usually expect (H, W, C). Add the channel dimension back.
        return frame[:, :, np.newaxis]
        
    except Exception as e:
        print(f"[server] frame decode error: {e}")
        return None
        
def encode_frame(frame: np.ndarray) -> str:
    """
    Converts a numpy array (H, W, 3) back into a base64 JPEG string 
    for the iOS client to display.
    """
    try:
        image = Image.fromarray(frame)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70) # Quality 70 to save bandwidth
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[server] frame encode error: {e}")
        return ""

def generate_test_pattern(width: int = 128, height: int = 128) -> str:
    """
    Generates a 4-quadrant random color test pattern.
    Useful for checking image orientation and scaling on iOS.
    """
    try:
        # Create 4 random colors
        colors = np.random.randint(0, 256, (4, 3), dtype=np.uint8)
        
        # Create a blank canvas
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Fill quadrants
        h_mid, w_mid = height // 2, width // 2
        canvas[0:h_mid, 0:w_mid] = colors[0]     # Top-left
        canvas[0:h_mid, w_mid:] = colors[1]      # Top-right
        canvas[h_mid:, 0:w_mid] = colors[2]      # Bottom-left
        canvas[h_mid:, w_mid:] = colors[3]       # Bottom-right
        
        # Standard encoding pipeline
        image = Image.fromarray(canvas)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[server] test pattern error: {e}")
        return ""

# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client_id = id(ws)
    # agent = RLAgent()
    agent = PolicyAgent(num_codes=64)

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


                # 2. Prepare the Image Action
                # We send the frame the agent 'saw' or a modified visualization
                # encoded_action_image = encode_frame(frame) # TODO: Not the action - just a pass-through
                #encoded_action_image = generate_test_pattern()
                # action_grid = gen.random_action_grid()       # TODO: replace with agent's output

                action_grid  = agent.select_action(frame)          # (32, 32) from the policy net
                action_image = gen.decode_actions(action_grid)     # (128, 128, 3) uint8
                encoded_action_image = encode_frame(action_image)

                # 3. Construct response matching iOS 'ActionMessage' struct
                response = {
                    "type": "frame",            # Matches the struct 'type' filter
                    "action": encoded_action_image, # This is the base64 string
                    "action_label": "",
                    "step": agent.stats["steps"],
                }                
                await ws.send_text(json.dumps(response))

                # 4. Update policy
                agent.update()                                  # update every step

            # ── Human reward ──────────────────────────────────────────────────
            elif msg_type == "reward": # TODO: This needs to be an internal signal with intermittent user input.
                reward_val = float(msg.get("value", 0)) 
                agent.record_reward(reward_val)                
                print(f"[server] reward {reward_val:+.1f} received | "
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