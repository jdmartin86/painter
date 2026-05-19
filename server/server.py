from __future__ import annotations

import io
import json
import time
import struct
from typing import Dict

import numpy as np
import cv2  
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from policy import PolicyAgent
import vqvae_mnist as vqvae

IMAGE_H = 128
IMAGE_W = 128

# ── App Initialization ────────────────────────────────────────────────────────

app = FastAPI(title="RL Agent Server")

# Initialize VQ-VAE model architecture
gen = vqvae.VQVAEGenerator(checkpoint_path="vqvae_mnsit.pth", num_codes=64)

_agents: Dict[int, PolicyAgent] = {}
_grids: Dict[int, np.ndarray] = {}  


# ── High Speed Frame Processors (No Base64, No PIL) ───────────────────────────

def fast_decode_frame(raw_bytes: bytes) -> np.ndarray | None:
    """
    Directly converts raw JPEG bytes into a uint8 numpy array using OpenCV.
    Bypasses base64 string decoding and PIL image allocation entirely.
    """
    try:
        # Convert raw byte array buffer to numpy layout
        nparr = np.frombuffer(raw_bytes, np.uint8)
        # Decode straight to Grayscale matrix
        frame = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if frame is None:
            return None

        # Resize efficiently if dimensions mismatch
        if frame.shape[1] != IMAGE_W or frame.shape[0] != IMAGE_H:
            frame = cv2.resize(frame, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_LINEAR)

        # Add tracking channel dimension (H, W, 1)
        return frame[:, :, np.newaxis]
    except Exception as e:
        print(f"[server] fast decode error: {e}")
        return None


def fast_encode_frame(frame: np.ndarray) -> bytes:
    """
    Compresses a numpy array into raw JPEG binary bytes instantly using OpenCV.
    Converts Grayscale matrices into standard 3-channel JPEGs for iOS compatibility.
    """
    try:
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame.squeeze(axis=-1)
            
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Upscale clean 28x28 VQ-VAE grid cleanly to 128x128
        if frame.shape[1] != IMAGE_W or frame.shape[0] != IMAGE_H:
            frame = cv2.resize(frame, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_NEAREST)
            
        # ── THE FIX: Force Grayscale matrix into standard 3-channel image mapping ──
        # This duplicates the mono channel across B, G, and R, creating a legal JPEG profile
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        # Encode straight into byte buffer format
        success, encoded_img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if success:
            return encoded_img.tobytes()
    except Exception as e:
        print(f"[server] fast encode error: {e}")
    return b""

# ── WebSocket Binary Endpoint ─────────────────────────────────────────────────

# ── WebSocket Binary Endpoint ─────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client_id = id(ws)
    
    agent = PolicyAgent(num_codes=gen.num_codes, grid_size=gen.GRID_SIZE)
    _agents[client_id] = agent
    _grids[client_id] = np.zeros((gen.GRID_SIZE, gen.GRID_SIZE), dtype=np.uint8)
    
    pending_reward = 0.0
    print(f"[server] Client {client_id} connected seamlessly. Ready for binary stream.")

    try:
        while True:
            # 1. Fetch the message type using the underlying framework reader
            websocket_msg = await ws.receive()
            
            # Catch Clean Disconnect Messages Natively
            if websocket_msg.get("type") == "websocket.disconnect":
                print(f"[server] client {client_id} sent disconnect flag.")
                break

            # ── BRANCH A: Handle Human Async Reward Tap (Text Message) ───────
            current_step_reward = 0.
            if "text" in websocket_msg:
                text_data = websocket_msg["text"]
                try:
                    msg = json.loads(text_data)
                    if msg.get("type") == "reward":
                        current_step_reward = float(msg.get("value", 0))
                        agent.record_reward(current_step_reward)                
                        print(f"[server] buffered reward: {current_step_reward:+.1f} | total pending: {pending_reward:+.1f}")
                except Exception as json_err:
                    print(f"[server] json parse error: {json_err}")
                continue

            # ── BRANCH B: Handle Raw Swift Camera Frames (Binary Message) ────
            elif "bytes" in websocket_msg:
                raw_bytes = websocket_msg["bytes"]
                if not raw_bytes or len(raw_bytes) == 0:
                    continue
                
                # Decode the raw JPEG bytes directly into a numpy matrix
                frame = fast_decode_frame(raw_bytes)
                if frame is None:
                    print("[server] Warning: Frame processing returned empty array layout.")
                    continue

                # 2. Select model actions
                chosen_code, active_position = agent.select_action(frame)
                
                r = active_position // gen.GRID_SIZE
                c = active_position % gen.GRID_SIZE
                _grids[client_id][r, c] = chosen_code
                
                # 3. Process image decoding paths via shared backend
                action_image = gen.decode_actions(_grids[client_id])
                jpeg_bytes = fast_encode_frame(action_image)
                
                if not jpeg_bytes:
                    print("[server] Warning: Generated image compression returned empty byte string.")
                    continue

                # 4. Construct Custom Ultra-Light Binary Protocol Header
                metadata_header = struct.pack("!IBBf", agent.stats["steps"], r, c, current_step_reward)
                binary_payload = metadata_header + jpeg_bytes
                
                # Push back down the socket explicitly as raw data frames to your Swift listener
                await ws.send_bytes(binary_payload)

                # 5. Run the optimizer update step
                agent.update()

                # 6. Housekeeping and reset frame buffer state
                print(f"[server] step {agent.stats['steps']} | pos ({r},{c}) | reward {current_step_reward:+.1f} | payload sent: {len(binary_payload)} bytes")
                pending_reward = 0.0

    except WebSocketDisconnect:
        print(f"[server] Client {client_id} disconnected via ASGI exception context.")
    except Exception as e:
        print(f"[server] unexpected runtime error for client {client_id}: {e}")
    finally:
        _agents.pop(client_id, None)
        _grids.pop(client_id, None)
        print(f"[server] cleaned up state resources for client {client_id}")

# ── Health Metrics ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "connected_clients": len(_agents), "timestamp": time.time()}

@app.get("/stats")
def stats():
    return {client_id: agent.stats for client_id, agent in _agents.items()}