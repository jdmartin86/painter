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

from policy import PolicyAgent
import vqvae_mnist as vqvae

IMAGE_H = 128
IMAGE_W = 128
IMAGE_C = 1


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="RL Agent Server")

# Initialize the generator wrapper with your subset size (e.g., 64)
# Internally, it loads your trained 7x7 grid VQ-VAE model architecture
gen = vqvae.VQVAEGenerator(checkpoint_path="vqvae_mnsit.pth", num_codes=64)

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
    Converts a numpy array back into a base64 JPEG string.
    Automatically handles dimension squeezing, handles grayscale formatting,
    and upscales clean 28x28 MNIST frames back up to 128x128 for iOS visualization.
    """
    try:
        # Drop channel dimension if the array comes out as (28, 28, 1)
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame.squeeze(axis=-1)
            
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Detect image layout mode (L for grayscale, RGB for color patterns)
        mode = "L" if frame.ndim == 2 else "RGB"
        image = Image.fromarray(frame, mode=mode)
        
        # Upscale the VQ-VAE's 28x28 output to the 128x128 expected by iOS.
        # NEAREST preserves crisp, clear individual retro pixels instead of blurry interpolations.
        if image.size != (IMAGE_W, IMAGE_H):
            image = image.resize((IMAGE_W, IMAGE_H), Image.NEAREST)
            
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
    
    # Dynamically reads gen.num_codes (64) and gen.GRID_SIZE (7) from your new generator setup
    agent = PolicyAgent(num_codes=gen.num_codes, grid_size=gen.GRID_SIZE)
    _agents[client_id] = agent
    
    # Track the "running" reward value between frames
    pending_reward = 0.0
    print(f"[server] client {client_id} connected (Grid Size: {gen.GRID_SIZE}x{gen.GRID_SIZE})")

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            # ── Human reward (Buffered) ──────────────────────────────────────
            if msg_type == "reward":
                val = float(msg.get("value", 0))
                pending_reward += val
                print(f"[server] buffered human feedback: {val:+.1f} | total pending: {pending_reward:+.1f}")
                continue # Wait for frame to process and apply

            # ── Incoming frame (The "Step") ──────────────────────────────────
            elif msg_type == "frame":
                frame = decode_frame(msg["data"]) 
                if frame is None:
                    continue

                # 1. Apply the reward (defaults to 0.0 if reward branch wasn't hit)
                current_step_reward = pending_reward
                agent.record_reward(current_step_reward)                

                # 2. Select action based on the current frame
                action_grid  = agent.select_action(frame)
                action_image = gen.decode_actions(action_grid)
                encoded_action_image = encode_frame(action_image)

                # 3. Construct response matching iOS 'ActionMessage' struct
                response = {
                    "type": "frame",
                    "action": encoded_action_image,
                    "action_label": f"R: {current_step_reward:+.1f}",
                    "step": agent.stats["steps"],
                }                
                await ws.send_text(json.dumps(response))

                # 4. Update policy
                agent.update()

                # 5. Status Prints
                print(f"[server] step {agent.stats['steps']} | reward {current_step_reward:+.1f} applied | "
                      f"stats: {agent.stats}")
                
                # 6. Reset buffer for next frame
                pending_reward = 0.0
                
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