"""
VQVAEGenerator — wraps the pretrained OpenAI DALL-E dVAE decoder.

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

    def __init__(self, checkpoint_path="decoder.pkl", device=None):
        self.device = torch.device(device or _best_device())
        print(f"Using device: {self.device}")

        self._download(checkpoint_path)
        print("Loading decoder...")
        with open(checkpoint_path, "rb") as f:
            self.dec = torch.load(f, map_location=self.device, weights_only=False)
        self.dec.eval()
        print("Ready.")

    def _download(self, path):
        if not os.path.exists(path):
            print("Downloading decoder.pkl (~400 MB)...")
            urllib.request.urlretrieve(self.DECODER_URL, path)

    @staticmethod
    def _unmap_pixels(x, eps=0.1):
        return torch.clamp((x - eps) / (1 - 2 * eps), 0, 1)

    def generate_image(self, output_size=128) -> np.ndarray:
        """
        Generate an image from random codebook tokens.
        Returns np.ndarray of shape (output_size, output_size, 3), dtype uint8.
        """
        with torch.no_grad():
            indices = torch.randint(
                0, self.NUM_TOKENS,
                (1, self.GRID_SIZE, self.GRID_SIZE),
                device=self.device,
            )
            z = F.one_hot(indices, num_classes=self.NUM_TOKENS) \
                 .permute(0, 3, 1, 2).float()
            x_stats = self.dec(z)
            x = self._unmap_pixels(torch.sigmoid(x_stats[:, :3]))

        img = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        if output_size != 256:
            img = np.array(
                Image.fromarray(img).resize((output_size, output_size), Image.LANCZOS)
            )
        return img