import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import get_inputs, flatten_images


class Generator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim

        self.generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        prob_real = self.discriminator(x)
        assert prob_real.shape == (x.size(0), 1)
        return prob_real



def discriminator_loss(prob_real: Tensor, prob_fake: Tensor) -> Tensor:
    return -torch.mean(torch.log(prob_real + 1e-8) + torch.log(1 - prob_fake + 1e-8))\
    
def generator_loss(prob_fake: Tensor) -> Tensor:
    return -torch.mean(torch.log(prob_fake + 1e-8))


def gan_train(
    generator: Generator,
    discriminator: Discriminator,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
) -> list[float]:
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

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

            prob_real = discriminator(x_real)
            prob_fake = discriminator(x_fake)

            d_loss = discriminator_loss(prob_real, prob_fake)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            z = torch.randn(batch_size, generator.input_dim, device=device)
            x_fake = generator(z)

            prob_fake = discriminator(x_fake)
            g_loss = generator_loss(prob_fake)

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
