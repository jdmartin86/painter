"""
VQVAEGenerator — wraps the pretrained OpenAI DALL-E dVAE decoder.

The agent selects from a small subset of the full 8192-token codebook,
keeping the action space tractable for RL while the decoder runs unchanged.

Install:
    pip install torch pillow "dall-e" "attrs==21.4.0"
"""

import os
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import dall_e  # noqa — required so pickle can find dall_e.decoder.Decoder


def _best_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class VQVAEGenerator:
    DECODER_URL = "https://cdn.openai.com/dall-e/decoder.pkl"
    NUM_TOKENS  = 8192
    GRID_SIZE   = 32

    def __init__(self, checkpoint_path="decoder.pkl", device=None, num_codes=64):
        """
        Args:
            num_codes: size of the codebook subset the agent acts over.
                       Agent outputs values in [0, num_codes); this class
                       maps them to real codebook indices before decoding.
        """
        self.device = torch.device(device or _best_device())
        self.num_codes = num_codes
        print(f"Using device: {self.device}, codebook subset: {num_codes}/{self.NUM_TOKENS}")

        self._download(checkpoint_path)
        print("Loading decoder...")
        with open(checkpoint_path, "rb") as f:
            self.dec = torch.load(f, map_location=self.device, weights_only=False)
        self.dec.eval()

        # Fixed random subset of codebook indices, shape (num_codes,)
        self.codebook_subset = torch.randperm(self.NUM_TOKENS)[:num_codes].to(self.device)
        print("Ready.")

    def _download(self, path):
        if not os.path.exists(path):
            print("Downloading decoder.pkl (~400 MB)...")
            urllib.request.urlretrieve(self.DECODER_URL, path)

    @staticmethod
    def _unmap_pixels(x, eps=0.1):
        return torch.clamp((x - eps) / (1 - 2 * eps), 0, 1)

    def decode_actions(self, action_grid) -> np.ndarray:
        """
        Decode an agent action grid into an image.

        Args:
            action_grid: array-like of shape (GRID_SIZE, GRID_SIZE) with
                         integer values in [0, num_codes). This is what the
                         RL agent outputs directly.

        Returns:
            np.ndarray of shape (128, 128, 3), dtype uint8.
        """
        action_grid = torch.as_tensor(action_grid, dtype=torch.long, device=self.device)
        assert action_grid.shape == (self.GRID_SIZE, self.GRID_SIZE), \
            f"Expected ({self.GRID_SIZE}, {self.GRID_SIZE}), got {action_grid.shape}"
        assert action_grid.max() < self.num_codes, \
            f"Action value {action_grid.max()} out of range [0, {self.num_codes})"

        # Map agent actions → real codebook indices
        indices = self.codebook_subset[action_grid].unsqueeze(0)  # (1, 32, 32)

        with torch.no_grad():
            z = F.one_hot(indices, num_classes=self.NUM_TOKENS) \
                 .permute(0, 3, 1, 2).float()
            x_stats = self.dec(z)
            x = self._unmap_pixels(torch.sigmoid(x_stats[:, :3]))

        img = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = np.array(Image.fromarray(img).resize((128, 128), Image.LANCZOS))
        return img

    def random_action_grid(self) -> np.ndarray:
        """
        Sample a random action grid — useful for testing and for the agent's
        initial random policy.

        Returns:
            np.ndarray of shape (GRID_SIZE, GRID_SIZE), dtype int64,
            values in [0, num_codes).
        """
        return np.random.randint(0, self.num_codes, (self.GRID_SIZE, self.GRID_SIZE))