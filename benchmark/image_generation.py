"""
Benchmark: Image Generation

Compares:
- Reconstruction quality (how well each model reconstructs inputs)
- Generation quality (how well each model generates new samples from random z)

NOTE: Unlike other scripts in this repo, this one was written by AI.
"""

import torch
import torchvision
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import Adam

from models.vae import VAE, vae_loss
from models.ae import AE, ae_loss
from utils import pick_device, train, get_inputs, flatten_images


# ── Hyperparameters ────────────────────────────────────────────────────────────

BATCH_SIZE = 128
HIDDEN_DIM = 400
LATENT_DIM = 20
LEARNING_RATE = 1e-3
EPOCHS = 10
NUM_SAMPLES = 10  # images to show in visualizations


# ── Data ───────────────────────────────────────────────────────────────────────

def load_mnist(batch_size: int) -> tuple[DataLoader, int]:
    dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim = dataset[0][0].numel()
    return loader, input_dim


# ── Visualization ──────────────────────────────────────────────────────────────

def show_reconstructions(models: dict, loader: DataLoader, device: torch.device):
    """Show original images alongside reconstructions from each model."""
    batch = next(iter(loader))
    x = get_inputs(batch).to(device)
    x = flatten_images(x)[:NUM_SAMPLES]

    n_models = len(models)
    fig, axes = plt.subplots(n_models + 1, NUM_SAMPLES, figsize=(NUM_SAMPLES * 1.5, (n_models + 1) * 1.5))
    fig.suptitle('Reconstructions', fontsize=14)

    # Row 0: originals
    for i in range(NUM_SAMPLES):
        axes[0, i].imshow(x[i].cpu().view(28, 28), cmap='gray')
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel('Original', rotation=90, labelpad=40, fontsize=10)

    for row, (name, model) in enumerate(models.items(), start=1):
        model.eval()
        with torch.no_grad():
            outputs = model(x)
            logits = outputs[0]
            recons = torch.sigmoid(logits)

        for i in range(NUM_SAMPLES):
            axes[row, i].imshow(recons[i].cpu().view(28, 28), cmap='gray')
            axes[row, i].axis('off')
        axes[row, 0].set_ylabel(name, rotation=90, labelpad=40, fontsize=10)

    plt.tight_layout()
    plt.show()


def show_samples(models: dict, device: torch.device):
    """Show samples generated from random latent vectors."""
    n_models = len(models)
    fig, axes = plt.subplots(n_models, NUM_SAMPLES, figsize=(NUM_SAMPLES * 1.5, n_models * 1.5))
    fig.suptitle('Generated Samples (from random z)', fontsize=14)

    for row, (name, model) in enumerate(models.items()):
        samples = model.sample(NUM_SAMPLES, device)
        for i in range(NUM_SAMPLES):
            ax = axes[row, i] if n_models > 1 else axes[i]
            ax.imshow(samples[i].cpu().view(28, 28), cmap='gray')
            ax.axis('off')
        first_ax = axes[row, 0] if n_models > 1 else axes[0]
        first_ax.set_ylabel(name, rotation=90, labelpad=40, fontsize=10)

    plt.tight_layout()
    plt.show()


def show_loss_curves(loss_curves: dict[str, list[float]]):
    """Plot training loss over epochs for each model."""
    plt.figure(figsize=(8, 4))
    for name, losses in loss_curves.items():
        plt.plot(losses, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (avg per batch)')
    plt.title('Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = pick_device()
    print(f'Using device: {device}')

    loader, input_dim = load_mnist(BATCH_SIZE)

    # AE loss_fn ignores the z returned by forward()
    ae_loss_fn = lambda x, logits, _z: ae_loss(x, logits)

    models_cfg = {
        'VAE': (VAE(input_dim, HIDDEN_DIM, LATENT_DIM), vae_loss),
        'AE':  (AE(input_dim, HIDDEN_DIM, LATENT_DIM),  ae_loss_fn),
    }

    loss_curves = {}
    trained_models = {}

    for name, (model, loss_fn) in models_cfg.items():
        print(f'\nTraining {name}...')
        model.to(device)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        losses = train(model, loader, optimizer, device, loss_fn, epochs=EPOCHS)
        loss_curves[name] = losses
        trained_models[name] = model
        print(f'{name} final loss: {losses[-1]:.2f}')

    show_loss_curves(loss_curves)
    show_reconstructions(trained_models, loader, device)
    show_samples(trained_models, device)


if __name__ == '__main__':
    main()
