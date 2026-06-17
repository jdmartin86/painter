from __future__ import annotations

import io
import json
import time
import struct
import asyncio
from typing import Dict

import numpy as np
import cv2  
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Import the architectural components directly
from agent import StreamAC
from detect_face import compute_reward
import vqvae_mnist as vqvae

IMAGE_H = 128
IMAGE_W = 128

# ── App Initialization ────────────────────────────────────────────────────────

app = FastAPI(title="RL Agent Server")

# 1. Allow a separately hosted frontend client to securely read telemetry data
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any standalone dev client (like VS Code Live Server)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize VQ-VAE model architecture
gen = vqvae.VQVAEGenerator(checkpoint_path="vqvae_mnsit.pth", num_codes=64)

# Dedicated client multi-tenant tracking states
_agents: Dict[int, StreamAC] = {}
_grids: Dict[int, np.ndarray] = {}  
_histories: Dict[int, dict] = {}
_locks: Dict[int, asyncio.Lock] = {}  # FIX: per-client locks to serialize forward/backward

# ── Background Worker Function ────────────────────────────────────────────────

def run_background_update(
    agent: StreamAC,
    s_tensor: torch.Tensor,
    flattened_action: np.ndarray,
    reward: float,
    s_prime_tensor: torch.Tensor,
    client_id: int,
):
    """
    Executes the neural network update pass safely in a separate execution flow.
    Tensors are cloned+detached before use so the main thread's concurrent
    forward pass cannot mutate them mid-gradient (fixes the inplace autograd error).
    Increments the update tracking metrics upon successful completion.
    """
    try:
        # FIX: clone + detach so autograd sees a stable version of each tensor
        # regardless of what the main loop does to the originals after dispatch.
        s = s_tensor.clone().detach()
        s_prime = s_prime_tensor.clone().detach()

        agent.update_params(
            s=s,
            a=flattened_action,
            r=reward,
            s_prime=s_prime
        )
        # FIX: "updates" key is now guaranteed to exist (seeded at construction)
        agent.stats["updates"] += 1
    except Exception as train_err:
        print(f"[server] Background parameter optimization step error for client {client_id}: {train_err}")

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
            
        # Force Grayscale matrix into standard 3-channel image mapping
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

    # FIX: seed all expected stats keys so += never hits a missing key
    agent.stats.setdefault("updates", 0)
    agent.stats.setdefault("steps", 0)
    agent.stats.setdefault("last_reward", 0.0)

    _agents[client_id] = agent
    _grids[client_id] = np.zeros((gen.GRID_SIZE, gen.GRID_SIZE), dtype=np.uint8)
    _histories[client_id] = {"obs": None, "action": None}
    _locks[client_id] = asyncio.Lock()  # FIX: one lock per client

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
            lock = _locks[client_id]

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

                # FIX: acquire the lock before forward pass so it cannot overlap
                # with a concurrent backward pass running in the background thread.
                async with lock:
                    with torch.no_grad():
                        # action_tensor = agent.sample_action(torch_input)
                        action_tensor = agent.epsilon_greedy_action(torch_input)
                        # action_tensor = agent.greedy_action(torch_input)
                action = action_tensor.cpu().numpy().astype(np.uint8)
                                            
                # Run optimization update pass step if a history frame exists
                if history["obs"] is not None:
                    # Convert history frames from (128, 128, 1) to explicit 4D tensors (1, 1, 128, 128)
                    s_tensor = (
                        torch.tensor(history["obs"], dtype=torch.float32)
                        .permute(2, 0, 1).unsqueeze(0)
                        .div_(255.0)
                    )
                    s_prime_tensor = (
                        torch.tensor(next_obs, dtype=torch.float32)
                        .permute(2, 0, 1).unsqueeze(0)
                        .div_(255.0)
                    )

                    # Flatten action history from (7, 7) to (1, 49) to match Categorical distribution tracking batch shape
                    flattened_action = history["action"].reshape(1, -1)

                    # FIX: wrap the blocking backward pass so it acquires the same
                    # lock as the forward pass — forward and backward are now mutually exclusive.
                    async def _locked_update(
                        _agent=agent,
                        _s=s_tensor,
                        _a=flattened_action,
                        _r=reward,
                        _sp=s_prime_tensor,
                        _cid=client_id,
                        _lock=lock,
                    ):
                        if _lock.locked():
                            return
                        async with _lock:
                            await asyncio.to_thread(
                                run_background_update,
                                _agent, _s, _a, _r, _sp, _cid,
                            )

                    asyncio.create_task(_locked_update())
                    
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
                print(f"[server] frame {agent.stats['steps']} | update {agent.stats['updates']} | reward {reward:+.1f}")

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
        _locks.pop(client_id, None)  # FIX: clean up the lock too
        print(f"[server] cleaned up state resources for client {client_id}")

# ── Health Metrics ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "connected_clients": len(_agents), "timestamp": time.time()}

@app.get("/stats")
def stats():
    return {client_id: agent.stats for client_id, agent in _agents.items()}

# ── Live-Streaming Image Generators (On-Demand Only) ──────────────────────────

async def get_input_stream(client_id: int):
    """Streams the raw incoming camera environment frames directly from history."""
    while True:
        if client_id in _histories:
            history = _histories[client_id]
            raw_frame = history.get("obs")
            if raw_frame is not None:
                encoded_bytes = encode_frame(raw_frame)
                if encoded_bytes:
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + encoded_bytes + b'\r\n')
        else:
            break
        await asyncio.sleep(0.04)

async def get_output_stream(client_id: int):
    """
    On-Demand Generator: Grabs the current 'action' token state, passes it through 
    the VQ-VAE decoder, and serves the sharp reconstructed visual output frame.
    """
    while True:
        if client_id in _histories:
            history = _histories[client_id]
            action = history.get("action")  # Pulls the active (7, 7) token block layout
            
            if action is not None:
                # Use your existing VQ-VAE decoder setup to construct the visual matrix
                action_image = gen.decode_actions(action)
                
                # Convert the decoded array structure into raw web-ready JPEGs
                encoded_bytes = encode_frame(action_image)
                if encoded_bytes:
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + encoded_bytes + b'\r\n')
        else:
            break
        await asyncio.sleep(0.04)

@app.get("/video_feed/input/{client_id}")
async def video_feed_input(client_id: int):
    return StreamingResponse(get_input_stream(client_id), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed/output/{client_id}")
async def video_feed_output(client_id: int):
    return StreamingResponse(get_output_stream(client_id), media_type="multipart/x-mixed-replace; boundary=frame")