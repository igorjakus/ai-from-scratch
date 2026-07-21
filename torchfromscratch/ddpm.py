"""Denoising Diffusion Probabilistic Models (DDPM) implementation in PyTorch."""

import torch
import torch.nn as nn

from torch import Tensor

class LinearNoiseScheduler:
    def __init__(self, T: int = 1000, beta_min: float = 1e-4, beta_max: float = 0.02):
        self.T = T
        self.betas = torch.linspace(beta_min, beta_max, T)  # (T,)
        self.alphas = 1.0 - self.betas  # (T,)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0) # (T,)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)  # (T,)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)  # (T,)

    def add_noise(self, x: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)  # (B, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)  # (B, 1)
        noisy_x = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
        return noisy_x

    def remove_noise(self, noisy_x: Tensor, noise_pred: Tensor, t: Tensor) -> Tensor:
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)  # (B, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)  # (B, 1)
        x = (noisy_x - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar
        return x

    def remove_noise_2(self, noisy_x: Tensor, noise_pred: Tensor, t: Tensor) -> Tensor:
        alpha = self.alphas[t].view(-1, 1)                                      # α_t
        beta = self.betas[t].view(-1, 1)                                        # β_t
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].view(-1, 1) # √(1 - ᾱ_t)

        # predicted mean of x_{t-1}
        mean = (1.0 / torch.sqrt(alpha)) * (noisy_x - beta / sqrt_one_minus_alpha_bar * noise_pred)

        # add noise for t > 0, no noise for t = 0 (final step)
        sigma = torch.sqrt(beta)
        z = torch.randn_like(noisy_x)
        z[t.view(-1) == 0] = 0.0

        return mean + sigma * z


class NoisePredictor(nn.Module):
    def __init__(self, in_channels: int, time_steps: int, time_emb_dim: int):
        super().__init__()
        self.in_channels = in_channels

        self.time_embedding = nn.Embedding(time_steps, time_emb_dim)

        self.sequential = nn.Sequential(
            nn.Linear(in_channels + time_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, in_channels)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        assert x.shape[0] == t.shape[0], "Batch size of x and t must match"
        assert t.dim() == 1, "t must be a 1D tensor of time steps"
        assert x.dim() == 2 and x.shape[1] == self.in_channels, f"x must be of shape (batch_size, {self.in_channels})"

        time_emb = self.time_embedding(t)
        x_time = torch.cat([x, time_emb], dim=-1)
        noise_pred = self.sequential(x_time)
        return noise_pred


class Sampler:
    def __init__(self, model: nn.Module, noise_scheduler: LinearNoiseScheduler):
        self.model = model
        self.noise_scheduler = noise_scheduler
    
    def sample(self, num_samples: int, device: torch.device):
        x = torch.randn(num_samples, self.model.in_channels, device=device)

        for t in reversed(range(self.noise_scheduler.T)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise_pred = self.model(x, t_tensor)
            x = self.noise_scheduler.remove_noise(x, noise_pred, t_tensor)

        return x
