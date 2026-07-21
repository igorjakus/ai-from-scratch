import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import get_inputs, flatten_images


class Generator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim

        self.generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.generator(x)
    
    @torch.inference_mode()
    def sample(self, num_samples: int) -> Tensor:
        device = self.generator[0].weight.device
        z = torch.randn(num_samples, self.input_dim, device=device)
        samples = self.forward(z)
        return samples


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim

        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        logits = self.discriminator(x)
        assert logits.shape == (x.size(0), 1)
        return logits



def discriminator_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    real_targets = torch.ones_like(real_logits)
    fake_targets = torch.zeros_like(fake_logits)
    real_loss = F.binary_cross_entropy_with_logits(real_logits, real_targets)
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
    return real_loss + fake_loss
    
def generator_loss(fake_logits: Tensor) -> Tensor:
    targets = torch.ones_like(fake_logits)
    return F.binary_cross_entropy_with_logits(fake_logits, targets)


def gan_train(
    generator: Generator,
    discriminator: Discriminator,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
) -> list[float]:
    g_optimizer = torch.optim.Adam(generator.parameters(),     lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    epoch_losses = []
    for epoch in range(epochs):
        total_g_loss = 0.0
        total_d_loss = 0.0
        total_samples = 0

        for batch in tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}'):
            x_real = get_inputs(batch).to(device)
            x_real = flatten_images(x_real)
            batch_size = x_real.size(0)

            # Train Discriminator
            z = torch.randn(batch_size, generator.input_dim, device=device)
            x_fake = generator(z).detach()
            x_fake = x_fake.detach()  # we don't need generator in the graph
                                      # better performance when we avoid backprop to G

            real_logits = discriminator(x_real)
            fake_logits = discriminator(x_fake)

            d_loss = discriminator_loss(real_logits, fake_logits)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            z = torch.randn(batch_size, generator.input_dim, device=device)
            x_fake = generator(z)

            fake_logits = discriminator(x_fake)
            g_loss = generator_loss(fake_logits)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            total_g_loss += g_loss.item() * batch_size
            total_d_loss += d_loss.item() * batch_size
            total_samples += batch_size

        avg_g_loss = total_g_loss / total_samples
        avg_d_loss = total_d_loss / total_samples
        epoch_losses.append((avg_g_loss, avg_d_loss))
        print(f'Epoch {epoch+1}: Avg G Loss={avg_g_loss:.4f}, Avg D Loss={avg_d_loss:.4f}')

    return epoch_losses
