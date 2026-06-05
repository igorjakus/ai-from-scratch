import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: Tensor) -> Tensor:
        assert x.dim() == 2, f"Expected 2D tensor (batch, input_dim), got shape {tuple(x.shape)}"
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        assert z.dim() == 2, f"Expected 2D tensor (batch, latent_dim), got shape {tuple(z.shape)}"
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encode(x)
        logits = self.decode(z)
        assert logits.shape == x.shape
        return logits, z

    @torch.inference_mode()
    def sample(self, num_samples: int, device: torch.device) -> Tensor:
        # AE has no regularized latent space, so sampling
        # won't produce meaningful outputs (unlike VAE)
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return torch.sigmoid(self.decode(z))


def ae_loss(x: Tensor, logits: Tensor) -> Tensor:
    assert x.shape == logits.shape, f"Shape mismatch: x={tuple(x.shape)}, logits={tuple(logits.shape)}"
    # pure reconstruction loss - no regularization on the latent space
    return F.binary_cross_entropy_with_logits(logits, x, reduction='sum')
