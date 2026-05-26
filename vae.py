import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim import Adam


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * input_dim)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        assert mu.shape == logvar.shape
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        # reparametrization trick
        eps = torch.randn_like(mu)
        std = torch.exp(logvar / 2)
        z = eps * std + mu

        mu2, logvar2 = self.decode(z)
        logvar2 = torch.clamp(logvar2, min=-10, max=10)
        assert mu2.shape == logvar2.shape == x.shape

        return mu2, logvar2, mu, logvar

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder(x)
        mu, logvar = torch.chunk(enc, 2, dim=-1)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dec = self.decoder(z)
        mu, logvar = torch.chunk(dec, 2, dim=-1)
        return mu, logvar


def vae_loss(x: torch.Tensor, mu2: torch.Tensor, _logvar2: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # reconstruction loss
    recon_loss = (mu2 - x).pow(2).sum()
    # mu2 is the most probable reconstruction of x so we take it as the sample to compare with x

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # we want to minimize the KL divergence between the latent distribution and the standard normal distribution

    # so we have two forces:
    # 1. the reconstruction loss forces the model to reconstruct the input accurately
    # 2. the KL divergence loss forces the latent distribution to be close to the standard normal distribution, which encourages the model to learn a smooth latent space
    return recon_loss + kl_loss


def get_inputs(batch: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def flatten_images(x: torch.Tensor) -> torch.Tensor:
    if x.dim() > 2:
        return x.view(x.size(0), -1)
    return x


def train(
    model: VAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 1,
) -> float:
    model.train()
    total_loss = 0
    total_batches = 0

    for _ in tqdm(range(epochs), desc='Epochs'):
        for batch in loader:
            x = get_inputs(batch)
            x = x.to(device)
            x = flatten_images(x)

            mu2, logvar2, mu, logvar = model(x)

            loss = vae_loss(x, mu2, logvar2, mu, logvar)
            total_loss += loss.item()
            total_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return total_loss / total_batches


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def visualize(model: VAE, loader: DataLoader, device: torch.device):
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = get_inputs(batch)
            x = x.to(device)
            x = flatten_images(x)
            mu2, logvar2, _, _ = model(x)
            # visualize the first 10 reconstructions
            for i in range(10):
                original = x[i].cpu().numpy().reshape(28, 28)
                reconstruction = mu2[i].cpu().numpy().reshape(28, 28)

                _fig, axes = plt.subplots(1, 2)
                axes[0].imshow(original, cmap='gray')
                axes[0].set_title('Original')
                axes[0].axis('off')

                axes[1].imshow(reconstruction, cmap='gray')
                axes[1].set_title('Reconstruction')
                axes[1].axis('off')

                plt.show()
            break  # only visualize the first batch


def main():
    DEVICE = pick_device()
    print(f'Using device: {DEVICE}')

    BATCH_SIZE = 128
    HIDDEN_DIM = 400
    LATENT_DIM = 20
    LEARNING_RATE = 1e-3
    EPOCHS = 100

    dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    INPUT_DIM = dataset[0][0].numel()
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
    model.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, train_loader, optimizer, DEVICE, epochs=EPOCHS)
    visualize(model, train_loader, DEVICE)


if __name__ == '__main__':
    main()