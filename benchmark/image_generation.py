"""
Benchmark: Image Generation

Compares:
- Reconstruction quality (how well each model reconstructs inputs)
- Generation quality (how well each model generates new samples from random z)

NOTE: Unlike other scripts in this repo, this one was written by AI.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.ae import AE, ae_loss
from models.dcgan import DCGANDiscriminator, DCGANGenerator, dcgan_train
from models.flow_matching import FlowMatching, fm_train
from models.gan import Discriminator, Generator, gan_train
from models.vae import VAE, vae_loss
from utils import count_params, flatten_images, get_inputs, pick_device, train


BATCH_SIZE = 128
HIDDEN_DIM = 400
DCGAN_FEATURES = 32
LATENT_DIM = 40
LEARNING_RATE = 1e-3
NUM_SAMPLES = 5
IMAGE_SIZE = 32
OUTPUT_DIR = Path('benchmark/results')


@dataclass(frozen=True)
class ModelSpec:
    name: str
    enabled: bool
    epochs: int
    build: Callable[[int], Any]
    train: Callable[[Any, DataLoader, torch.device, int], tuple[Any, list]]
    sample: Callable[[Any, int, torch.device], torch.Tensor]
    reconstruct: Callable[[Any, torch.Tensor, torch.device], torch.Tensor] | None
    evaluate: Callable[[Any, DataLoader, torch.device], float] | None
    params: Callable[[Any], int]


@dataclass(frozen=True)
class TrainedModel:
    artifact: Any
    spec: ModelSpec


def load_mnist(batch_size: int) -> tuple[DataLoader, DataLoader, int]:
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        torchvision.transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    val_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    loader_kwargs = dict(batch_size=batch_size, pin_memory=True, num_workers=4, persistent_workers=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    input_dim = train_dataset[0][0].numel()
    return train_loader, val_loader, input_dim


def as_image(sample: torch.Tensor) -> torch.Tensor:
    sample = sample.detach().cpu()
    if sample.min() < 0:
        sample = sample * 0.5 + 0.5
    if sample.dim() == 1:
        return sample.view(IMAGE_SIZE, IMAGE_SIZE)
    if sample.dim() == 2:
        return sample.view(IMAGE_SIZE, IMAGE_SIZE)
    if sample.dim() == 3 and sample.size(0) == 1:
        return sample.squeeze(0)
    if sample.dim() == 3:
        return sample.permute(1, 2, 0)
    raise ValueError(f'Unsupported sample shape: {tuple(sample.shape)}')


@torch.inference_mode()
def flow_matching_reconstruct(model: FlowMatching, x_1: torch.Tensor, steps: int = 10) -> torch.Tensor:
    device = x_1.device
    batch_size = x_1.size(0)

    x = x_1.clone()
    dt = -1.0 / steps
    t = torch.ones(batch_size, device=device)
    for _ in range(steps):
        x = x + model(x, t) * dt
        t = t + dt

    dt = 1.0 / steps
    t = torch.zeros(batch_size, device=device)
    for _ in range(steps):
        x = x + model(x, t) * dt
        t = t + dt

    return torch.clamp(x, 0.0, 1.0)


def train_standard_model(model: torch.nn.Module, loader: DataLoader, device: torch.device, epochs: int, loss_fn) -> tuple[Any, list]:
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    if isinstance(model, FlowMatching):
        losses = fm_train(model, loader=loader, optimizer=optimizer, device=device, epochs=epochs)
    else:
        losses = train(model, loader=loader, optimizer=optimizer, device=device, loss_fn=loss_fn, epochs=epochs)
    return model, losses


def train_gan_bundle(bundle: tuple[Generator, Discriminator], loader: DataLoader, device: torch.device, epochs: int) -> tuple[Any, list]:
    generator, discriminator = bundle
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    losses = gan_train(generator, discriminator, loader, device, epochs=epochs)
    return (generator, discriminator), losses


def train_dcgan_bundle(bundle: tuple[DCGANGenerator, DCGANDiscriminator], loader: DataLoader, device: torch.device, epochs: int) -> tuple[Any, list]:
    generator, discriminator = bundle
    losses = dcgan_train(generator, discriminator, loader, device, epochs=epochs)
    return bundle, losses


def sample_flat_model(model: torch.nn.Module, num_samples: int, device: torch.device) -> torch.Tensor:
    return model.sample(num_samples, device)


def sample_flow_matching_model(model: FlowMatching, num_samples: int, device: torch.device) -> torch.Tensor:
    return model.sample(num_samples, steps=500)


def sample_weighted_generator(bundle: tuple[Any, Any], num_samples: int, device: torch.device) -> torch.Tensor:
    return bundle[0].sample(num_samples)


def reconstruct_logits(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    logits = model(x)[0]
    return torch.sigmoid(logits)


def reconstruct_flow_matching(model: FlowMatching, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return flow_matching_reconstruct(model, x, steps=100)


@torch.inference_mode()
def evaluate_reconstruction(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    is_fm = isinstance(model, FlowMatching)

    for batch in loader:
        x = flatten_images(get_inputs(batch).to(device))
        if is_fm:
            x_recon = flow_matching_reconstruct(model, x, steps=10)
            loss_val = F.binary_cross_entropy(torch.clamp(x_recon, 1e-7, 1.0 - 1e-7), x, reduction='sum').item()
        else:
            logits = model(x)[0]
            loss_val = F.binary_cross_entropy_with_logits(logits, x, reduction='sum').item()
        total_loss += loss_val
        total_samples += x.size(0)

    return total_loss / total_samples


def show_reconstructions(models: dict[str, TrainedModel], loader: DataLoader, device: torch.device) -> None:
    reconstructable = {name: model for name, model in models.items() if model.spec.reconstruct is not None}
    if not reconstructable:
        return

    batch = next(iter(loader))
    x = flatten_images(get_inputs(batch).to(device))[:NUM_SAMPLES]

    n_models = len(reconstructable)
    row_labels = ['Original'] + list(reconstructable.keys())
    fig, axes = plt.subplots(n_models + 1, NUM_SAMPLES, figsize=(NUM_SAMPLES * 1.5 + 1, (n_models + 1) * 1.5))
    fig.suptitle('Reconstructions', fontsize=14, y=1.01)

    for i in range(NUM_SAMPLES):
        axes[0, i].imshow(as_image(x[i]), cmap='gray')
        axes[0, i].axis('off')

    for row, (name, model) in enumerate(reconstructable.items(), start=1):
        with torch.no_grad():
            recons = model.spec.reconstruct(model.artifact, x, device)

        for i in range(NUM_SAMPLES):
            axes[row, i].imshow(as_image(recons[i]), cmap='gray')
            axes[row, i].axis('off')

    n_rows = n_models + 1
    for row, label in enumerate(row_labels):
        y = 1 - (row + 0.5) / n_rows
        fig.text(0.01, y, label, va='center', ha='left', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / 'reconstructions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved reconstructions -> {OUTPUT_DIR}/reconstructions.png')


def show_samples(models: dict[str, TrainedModel], device: torch.device) -> None:
    if not models:
        return

    n_models = len(models)
    fig, axes = plt.subplots(n_models, NUM_SAMPLES, figsize=(NUM_SAMPLES * 1.5 + 1, n_models * 1.5))
    fig.suptitle('Generated Samples (from random z)', fontsize=14, y=1.01)

    for row, (name, model) in enumerate(models.items()):
        samples = model.spec.sample(model.artifact, NUM_SAMPLES, device)
        for i in range(NUM_SAMPLES):
            ax = axes[row, i] if n_models > 1 else axes[i]
            ax.imshow(as_image(samples[i]), cmap='gray')
            ax.axis('off')

    for row, name in enumerate(models.keys()):
        y = 1 - (row + 0.5) / n_models
        fig.text(0.01, y, name, va='center', ha='left', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / 'samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved samples       -> {OUTPUT_DIR}/samples.png')


def show_loss_curves(loss_curves: dict[str, list]) -> None:
    plt.figure(figsize=(8, 4))
    for name, losses in loss_curves.items():
        if losses and isinstance(losses[0], tuple):
            g_losses = [g for g, _ in losses]
            d_losses = [d for _, d in losses]
            plt.plot(g_losses, label=f'{name} (G)')
            plt.plot(d_losses, label=f'{name} (D)', linestyle='--')
        else:
            plt.plot(losses, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (avg per batch)')
    plt.title('Training Loss')
    plt.legend()
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / 'loss_curves.png', dpi=150)
    plt.close()
    print(f'Saved loss curves   -> {OUTPUT_DIR}/loss_curves.png')


def print_summary(results: dict[str, dict]) -> None:
    cols = ['Model', 'Val Recon Loss', '# Params', 'Train Time']
    widths = [max(len(c), max(len(str(r.get(c, ''))) for r in results.values())) + 2 for c in cols]
    widths[0] = max(widths[0], max(len(k) for k in results) + 2)

    def row(cells):
        return '│' + '│'.join(str(c).center(w) for c, w in zip(cells, widths)) + '│'

    sep = lambda l, m, r: l + m.join('─' * w for w in widths) + r

    print()
    print(sep('┌', '┬', '┐'))
    print(row(cols))
    print(sep('├', '┼', '┤'))
    for name, r in results.items():
        print(row([name, r['Val Recon Loss'], r['# Params'], r['Train Time']]))
    print(sep('└', '┴', '┘'))


def format_final_loss(losses: list) -> str:
    if losses and isinstance(losses[-1], tuple):
        g_final, d_final = losses[-1]
        return f'G={g_final:.4f}, D={d_final:.4f}'
    return f'{losses[-1]:.2f}'


def build_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name='DCGAN',
            enabled=True,
            epochs=10,
            build=lambda _input_dim: (DCGANGenerator(LATENT_DIM, DCGAN_FEATURES, 1), DCGANDiscriminator(DCGAN_FEATURES, 1)),
            train=train_dcgan_bundle,
            sample=sample_weighted_generator,
            reconstruct=None,
            evaluate=None,
            params=lambda bundle: count_params(bundle[0]) + count_params(bundle[1]),
        ),
        ModelSpec(
            name='GAN',
            enabled=True,
            epochs=10,
            build=lambda input_dim: (Generator(input_dim, HIDDEN_DIM), Discriminator(input_dim, HIDDEN_DIM)),
            train=train_gan_bundle,
            sample=sample_weighted_generator,
            reconstruct=None,
            evaluate=None,
            params=lambda bundle: count_params(bundle[0]) + count_params(bundle[1]),
        ),
        ModelSpec(
            name='FlowMatching',
            enabled=True,
            epochs=25,
            build=lambda input_dim: FlowMatching(input_dim, HIDDEN_DIM),
            train=lambda model, loader, device, epochs: train_standard_model(model, loader, device, epochs, loss_fn=None),
            sample=sample_flow_matching_model,
            reconstruct=reconstruct_flow_matching,
            evaluate=evaluate_reconstruction,
            params=count_params,
        ),
        ModelSpec(
            name='VAE',
            enabled=True,
            epochs=25,
            build=lambda input_dim: VAE(input_dim, HIDDEN_DIM, LATENT_DIM),
            train=lambda model, loader, device, epochs: train_standard_model(model, loader, device, epochs, loss_fn=vae_loss),
            sample=sample_flat_model,
            reconstruct=reconstruct_logits,
            evaluate=evaluate_reconstruction,
            params=count_params,
        ),
        ModelSpec(
            name='AE',
            enabled=False,
            epochs=50,
            build=lambda input_dim: AE(input_dim, HIDDEN_DIM, LATENT_DIM),
            train=lambda model, loader, device, epochs: train_standard_model(model, loader, device, epochs, loss_fn=lambda x, logits, *_: ae_loss(x, logits)),
            sample=sample_flat_model,
            reconstruct=reconstruct_logits,
            evaluate=evaluate_reconstruction,
            params=count_params,
        ),
    ]


def main() -> None:
    device = pick_device()
    print(f'Using device: {device}')

    train_loader, val_loader, input_dim = load_mnist(BATCH_SIZE)

    loss_curves: dict[str, list] = {}
    trained_models: dict[str, TrainedModel] = {}
    results: dict[str, dict] = {}

    for spec in (s for s in build_specs() if s.enabled):
        print(f'\nTraining {spec.name}...')
        artifact = spec.build(input_dim)

        t0 = time.perf_counter()
        artifact, losses = spec.train(artifact, train_loader, device, spec.epochs)
        elapsed = time.perf_counter() - t0

        trained_models[spec.name] = TrainedModel(artifact=artifact, spec=spec)
        loss_curves[spec.name] = losses

        print(f'{spec.name} final train loss: {format_final_loss(losses)}  ({elapsed:.1f}s)')

        if spec.evaluate is None:
            val_recon_loss = 'N/A'
        else:
            val_recon_loss = f'{spec.evaluate(artifact, val_loader, device):.2f}'

        results[spec.name] = {
            'Val Recon Loss': val_recon_loss,
            '# Params': f'{spec.params(artifact):,}',
            'Train Time': f'{elapsed:.1f}s',
        }

    show_loss_curves(loss_curves)
    show_reconstructions(trained_models, train_loader, device)
    show_samples(trained_models, device)
    print_summary(results)


if __name__ == '__main__':
    main()
