import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # get mean and logvar of the latent distribution
        mu, logvar = self.encode(x)
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        # sample from the latent distribution
        z = self.reparameterize(mu, logvar)

        # decode the sampled latent vector
        logits = self.decode(z)

        return logits, mu, logvar

    @torch.inference_mode()
    def sample(self, num_samples: int, device: torch.device) -> Tensor:
        z = torch.randn(num_samples, self.fc_mu.out_features).to(device)
        logits = self.decode(z)
        x = torch.sigmoid(logits)
        return x


def vae_loss(x: Tensor, logits: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    # we have two forces:
    # 1. the reconstruction loss forces the model to reconstruct the input accurately
    recon_loss = F.binary_cross_entropy_with_logits(logits, x, reduction='sum')

    # 2. the KL divergence loss forces the latent distribution to be close to N(0, 1)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss
