from __future__ import annotations

import io
import json
import time
import struct
from typing import Dict

import numpy as np
import cv2  
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Import the architectural components directly
from agent import StreamAC
from detect_face import compute_reward
import vqvae_mnist as vqvae

IMAGE_H = 128
IMAGE_W = 128

# ── App Initialization ────────────────────────────────────────────────────────

app = FastAPI(title="RL Agent Server")

# Initialize VQ-VAE model architecture
gen = vqvae.VQVAEGenerator(checkpoint_path="vqvae_mnsit.pth", num_codes=64)

# Dedicated client multi-tenant tracking states
_agents: Dict[int, StreamAC] = {}
_grids: Dict[int, np.ndarray] = {}  
_histories: Dict[int, dict] = {}

# ── High Speed Frame Processors ───────────────────────────

def decode_frame(raw_bytes: bytes) -> np.ndarray | None:
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

def encode_frame(frame: np.ndarray) -> bytes:
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

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client_id = id(ws)
    
    # Instantiate state mapping matching your requested architecture pattern
    agent = StreamAC(
        num_codes=gen.num_codes, 
        grid_size=gen.GRID_SIZE,
        entropy_coeff=0.01
    )
    _agents[client_id] = agent
    _grids[client_id] = np.zeros((gen.GRID_SIZE, gen.GRID_SIZE), dtype=np.uint8)
    _histories[client_id] = {"obs": None, "action": None}
    
    print(f"[server] Client {client_id} connected seamlessly. Ready for binary stream.")

    try:
        while True:
            websocket_msg = await ws.receive()
            
            if websocket_msg.get("type") == "websocket.disconnect":
                print(f"[server] client {client_id} sent disconnect flag.")
                break

            # Local shortcuts to avoid nested dictionary mutations inside loop paths
            history = _histories[client_id]
            grid = _grids[client_id]

            # ── BRANCH A: Handle Human Async Reward Tap (Text Message) ───────
            reward = 0.
            if "text" in websocket_msg:
                text_data = websocket_msg["text"]
                try:
                    msg = json.loads(text_data)
                    if msg.get("type") == "reward":
                        reward = float(msg.get("value", 0))
                        agent.stats["last_reward"] = reward            
                        print(f"[server] viewer manual reward injection: {reward:+.1f}")
                except Exception as json_err:
                    print(f"[server] json parse error: {json_err}")
                continue

            # ── BRANCH B: Handle Raw Swift Camera Frames (Binary Message) ────
            elif "bytes" in websocket_msg:
                raw_bytes = websocket_msg["bytes"]
                if not raw_bytes or len(raw_bytes) == 0:
                    continue
                
                raw_frame = decode_frame(raw_bytes)
                if raw_frame is None:
                    print("[server] Warning: Frame processing returned empty array layout.")
                    continue
                
                # Reshape matrix safely to expected shape (1, 1, H, W)
                torch_input = torch.tensor(raw_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                next_obs = raw_frame

                # Run reward assessment step layout
                reward = compute_reward(next_obs)

                # Select grid actions by executing forward inference
                with torch.no_grad():
                    action_tensor = agent.sample_action(torch_input)
                    action = action_tensor.cpu().numpy().astype(np.uint8)
                                                
                # Run optimization update pass step if a history frame exists
                if history["obs"] is not None:
                    try:
                        # Convert history frames from (128, 128, 1) to explicit 4D tensors (1, 1, 128, 128)
                        s_tensor = torch.tensor(history["obs"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                        s_prime_tensor = torch.tensor(next_obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

                        # Flatten action history from (7, 7) to (1, 49) to match Categorical distribution tracking batch shape
                        flattened_action = history["action"].reshape(1, -1)

                        agent.update_params(
                            s=s_tensor,
                            a=flattened_action,
                            r=reward,
                            s_prime=s_prime_tensor
                        )                    
                    except Exception as train_err:
                        print(f"[server] Parameter optimization step error skipped: {train_err}")

                # Rotate tracking buffers for the next step sequence
                history["obs"] = next_obs
                history["action"] = action
                
                # Process image decoding paths via shared generative backend
                action_image = gen.decode_actions(action)
                jpeg_bytes = encode_frame(action_image)
                
                if not jpeg_bytes:
                    print("[server] Warning: Generated image compression returned empty byte string.")
                    continue

                # Construct Custom Ultra-Light Binary Protocol Header
                # Packs: steps (uint32), padding/placeholders for r/c (uint8), and reward (float)
                metadata_header = struct.pack("!IBBf", agent.stats["steps"], 0, 0, float(reward))
                binary_payload = metadata_header + jpeg_bytes
                
                # Push back down the socket explicitly as raw data frames to your Swift listener
                await ws.send_bytes(binary_payload)

                # Housekeeping updates
                agent.stats["steps"] += 1
                agent.stats["last_reward"] = reward
                print(f"[server] step {agent.stats['steps']} | reward {reward:+.1f} |")

    except WebSocketDisconnect:
        print(f"[server] Client {client_id} disconnected via ASGI exception context.")
    except Exception as e:
        import traceback
        print(f"[server] unexpected runtime error for client {client_id}: {e}")
        traceback.print_exc()
    finally:
        _agents.pop(client_id, None)
        _grids.pop(client_id, None)
        _histories.pop(client_id, None)
        print(f"[server] cleaned up state resources for client {client_id}")

# ── Health Metrics ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "connected_clients": len(_agents), "timestamp": time.time()}

@app.get("/stats")
def stats():
    return {client_id: agent.stats for client_id, agent in _agents.items()}