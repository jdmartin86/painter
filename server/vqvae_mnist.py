"""
VQVAEGenerator — wraps a pretrained MNIST VQ-VAE decoder.

The agent selects from a small subset of the full codebook,
keeping the action space tractable for RL while the decoder runs unchanged.

Install:
    pip install torch pillow torchvision
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Optional

def _best_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Model definition (Matches your trained VectorQuantizerEMA and VQVAE architecture exactly) ──

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.embedding_dim, 1.0 / self.embedding_dim)
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', self.embedding.weight.data.clone()) 
        
        self.decay = decay
        self.epsilon = epsilon

    def forward(self, inputs):
        # BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        distances = torch.cdist(flat_input, self.embedding.weight, p=2)
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # EMA codebook updates skipped during evaluation context (self.training is False)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices


class ResBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(False),
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(False),
            nn.Conv2d(hidden_dim, in_dim, kernel_size=1, bias=False)
        )
        
    def forward(self, x):
        return x + self.block(x)


class VQVAE(nn.Module):
    def __init__(self, num_hiddens, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()

        # Encoder: 1×28×28 → D×7×7
        self.encoder = nn.Sequential(
            nn.Conv2d(1, num_hiddens, 4, stride=2, padding=1),                    # 64×14×14
            nn.ReLU(),
            ResBlock(num_hiddens, num_residual_hiddens),
            nn.Conv2d(num_hiddens, num_hiddens, 4, stride=2, padding=1),          # 64×7×7
            nn.ReLU(),
            ResBlock(num_hiddens, num_residual_hiddens),
            nn.Conv2d(num_hiddens, embedding_dim, 1),                              # D×7×7
            nn.BatchNorm2d(embedding_dim)
        )

        self.vq = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost)

        # Decoder: D×7×7 → 1×28×28
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, num_hiddens, 1),                              # 64×7×7
            ResBlock(num_hiddens, num_residual_hiddens),
            nn.ConvTranspose2d(num_hiddens, num_hiddens, 4, stride=2, padding=1),  # 64×14×14
            nn.ReLU(),
            ResBlock(num_hiddens, num_residual_hiddens),
            nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, 4, stride=2, padding=1),  # 32×28×28
            nn.ReLU(),
            nn.Conv2d(num_hiddens // 2, 1, 3, padding=1),                         # 1×28×28
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        vq_loss, quantized, encoding_indices = self.vq(z)
        x_recon = self.decoder(quantized)
        return vq_loss, x_recon, encoding_indices


# ── Generator wrapper ─────────────────────────────────────────────────────────

class VQVAEGenerator:
    # Your custom MNIST encoder structure compresses to a clean 7x7 spatial layout
    GRID_SIZE = 7

    def __init__(
        self,
        checkpoint_path: str = "vqvae_mnsit.pth",
        num_hiddens: int = 64,
        num_residual_hiddens: int = 32,
        num_embeddings: int = 128,
        embedding_dim: int = 4,
        commitment_cost: float = 0.25,
        device: Optional[str] = None,
        num_codes: int = 64,
    ):
        """
        Args:
            checkpoint_path:     Path to a saved VQVAE state-dict or checkpoint payload.
            num_hiddens / …:     Configured parameters matching training script.
            num_codes:           Size of codebook subset the agent produces predictions across.
        """
        self.device    = torch.device(device or _best_device())
        self.num_codes = num_codes
        self.NUM_TOKENS = num_embeddings
        print(
            f"Using device: {self.device}, "
            f"codebook subset: {num_codes}/{num_embeddings}"
        )

        # Build model and load weights
        self.model = VQVAE(
            num_hiddens=num_hiddens,
            num_residual_hiddens=num_residual_hiddens,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        ).to(self.device)

        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # Handle checkpoint files containing tracking keys cleanly
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
            
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Extract internal components for quick lookups
        self.vq  = self.model.vq
        self.dec = self.model.decoder

        # Fixed random subset map of codebook choices, shape (num_codes,)
        self.codebook_subset = torch.randperm(num_embeddings)[:num_codes].to(self.device)
        print("Ready.")

    def decode_actions(self, action_grid) -> np.ndarray:
        """
        Decode an agent action grid into an image.

        Args:
            action_grid: array-like of shape (7, 7) with integer values in [0, num_codes).

        Returns:
            np.ndarray of shape (28, 28), dtype uint8 (grayscale).
        """
        action_grid = torch.as_tensor(action_grid, dtype=torch.long, device=self.device)
        assert action_grid.shape == (self.GRID_SIZE, self.GRID_SIZE), (
            f"Expected ({self.GRID_SIZE}, {self.GRID_SIZE}), got {action_grid.shape}"
        )
        assert action_grid.max() < self.num_codes, (
            f"Action value {action_grid.max()} out of range [0, {self.num_codes})"
        )

        # Map agent actions to actual codebook indices
        indices = self.codebook_subset[action_grid]           # (7, 7)

        with torch.no_grad():
            flat = indices.view(-1)                            # (49,)
            quantized = self.vq.embedding(flat)                # (49, embedding_dim)
            quantized = (
                quantized
                .view(1, self.GRID_SIZE, self.GRID_SIZE, -1)   # (1, 7, 7, embedding_dim)
                .permute(0, 3, 1, 2)                           # (1, embedding_dim, 7, 7)
                .contiguous()
            )
            x_recon = self.dec(quantized)                      # (1, 1, 28, 28)

        img = x_recon.squeeze().cpu().numpy()                  # (28, 28), values in [0,1]
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return img

    def random_action_grid(self) -> np.ndarray:
        """
        Sample a random action grid for initial exploration steps.

        Returns:
            np.ndarray of shape (7, 7), dtype int64, values in [0, num_codes).
        """
        return np.random.randint(0, self.num_codes, (self.GRID_SIZE, self.GRID_SIZE))