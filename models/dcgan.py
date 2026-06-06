import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import get_inputs


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, out_channels: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        # (B, latent_dim, 1, 1) -> (B, out_channels, 32, 32)
        self.generator = nn.Sequential(
            # B, latent_dim, 1, 1 -> B, hidden_dim*8, 4, 4
            nn.ConvTranspose2d(latent_dim, hidden_dim * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(),
            
            # B, hidden_dim*8, 4, 4 -> B, hidden_dim*4, 8, 8
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),

            # B, hidden_dim*4, 8, 8 -> B, hidden_dim*2, 16, 16
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),

            # B, hidden_dim*2, 16, 16 -> B, out_channels, 32, 32
            nn.ConvTranspose2d(hidden_dim * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, z: Tensor) -> Tensor:
        B, latent_dim = z.shape
        z = z.view(B, latent_dim, 1, 1)  # B, C, H, W
        x = self.generator(z)
        assert x.shape == (B, self.out_channels, 32, 32)
        return x

    @torch.inference_mode()
    def sample(self, num_samples: int) -> Tensor:
        device = self.generator[0].weight.device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.forward(z)
        assert samples.shape == (num_samples, self.out_channels, 32, 32)
        return samples


class DCGANDiscriminator(nn.Module):
    def __init__(self, hidden_dim: int, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # (B, in_channels, 32, 32) -> (B, 1, 1, 1)
        self.discriminator = nn.Sequential(
            # B, in_channels, 32, 32 -> B, hidden_dim*2, 16, 16
            nn.Conv2d(in_channels, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # B, hidden_dim*2, 16, 16 -> B, hidden_dim*4, 8, 8
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # B, hidden_dim*4, 8, 8 -> B, hidden_dim*8, 4, 4
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # B, hidden_dim*8, 4, 4 -> B, 1
            nn.Conv2d(hidden_dim * 8, 1, kernel_size=4, stride=1, padding=0)
        )
        self.apply(weights_init)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.discriminator(x)
        assert logits.shape == (x.size(0), 1, 1, 1)
        logits = logits.view(x.size(0), 1)  # B, 1
        return logits



def discriminator_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    real_targets = torch.ones_like(real_logits) * 0.9  # label smoothing
    fake_targets = torch.zeros_like(fake_logits) + 0.1
    real_loss = F.binary_cross_entropy_with_logits(real_logits, real_targets)
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
    return real_loss + fake_loss
    
def generator_loss(fake_logits: Tensor) -> Tensor:
    targets = torch.ones_like(fake_logits)
    return F.binary_cross_entropy_with_logits(fake_logits, targets)


def dcgan_train(
    generator: DCGANGenerator,
    discriminator: DCGANDiscriminator,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
) -> list[tuple[float, float]]:
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(),     lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    epoch_losses = []
    for epoch in range(epochs):
        total_g_loss = 0.0
        total_d_loss = 0.0
        total_samples = 0

        for batch in tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}'):
            x_real = get_inputs(batch).to(device) * 2 - 1
            batch_size = x_real.size(0)

            # Train Discriminator
            z = torch.randn(batch_size, generator.latent_dim, device=device)
            x_fake = generator(z)
            x_fake = x_fake.detach()  # we don't need generator in the graph
                                      # better performance when we avoid backprop to G

            real_logits = discriminator(x_real)
            fake_logits = discriminator(x_fake)

            d_loss = discriminator_loss(real_logits, fake_logits)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            z = torch.randn(batch_size, generator.latent_dim, device=device)
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
