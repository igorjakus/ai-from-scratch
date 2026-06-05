"""
Benchmark: Image Generation

Compares:
- Reconstruction quality (how well each model reconstructs inputs)
- Generation quality (how well each model generates new samples from random z)

NOTE: Unlike other scripts in this repo, this one was written by AI.
"""

import time
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path

from torch.utils.data import DataLoader
from torch.optim import Adam

from models.vae import VAE, vae_loss
from models.ae import AE, ae_loss
from models.flow_matching import FlowMatching, fm_train
from models.gan import Generator, Discriminator, gan_train
from utils import pick_device, train, count_params, get_inputs, flatten_images


# ── Hyperparameters ────────────────────────────────────────────────────────────

BATCH_SIZE = 128
HIDDEN_DIM = 400
LATENT_DIM = 40
LEARNING_RATE = 1e-3
EPOCHS = 50
NUM_SAMPLES = 5  # images to show in visualizations
OUTPUT_DIR = Path('benchmark/results')


# ── Data ───────────────────────────────────────────────────────────────────────

def load_mnist(batch_size: int) -> tuple[DataLoader, DataLoader, int]:
    transform = torchvision.transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True,  transform=transform, download=True)
    val_dataset   = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    loader_kwargs = dict(batch_size=batch_size, pin_memory=True, num_workers=4, persistent_workers=True)
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)

    input_dim = train_dataset[0][0].numel()
    return train_loader, val_loader, input_dim


# ── Flow Matching Reconstruction Helper ─────────────────────────────────────────

@torch.inference_mode()
def flow_matching_reconstruct(model: FlowMatching, x_1: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """Reconstruct inputs by running FlowMatching ODE backward to noise, then forward to data."""
    device = x_1.device
    B = x_1.size(0)
    
    # Go backward from t=1 to t=0
    x = x_1.clone()
    dt = -1.0 / steps
    t = torch.ones(B, device=device)
    for _ in range(steps):
        v = model(x, t)
        x = x + v * dt
        t = t + dt
    
    # Now x is noise at t=0. Go forward from t=0 to t=1
    dt = 1.0 / steps
    t = torch.zeros(B, device=device)
    for _ in range(steps):
        v = model(x, t)
        x = x + v * dt
        t = t + dt
        
    return torch.clamp(x, 0.0, 1.0)


# ── Visualization ──────────────────────────────────────────────────────────────

def show_reconstructions(models: dict, loader: DataLoader, device: torch.device):
    """Show original images alongside reconstructions from each model.
    Generator (GAN) is skipped — it has no encoder and cannot reconstruct inputs.
    """
    # GAN Generator has no encoder, skip it
    reconstructable = {k: v for k, v in models.items() if not isinstance(v, Generator)}
    if not reconstructable:
        return

    batch = next(iter(loader))
    x = get_inputs(batch).to(device)
    x = flatten_images(x)[:NUM_SAMPLES]

    n_models = len(reconstructable)
    row_labels = ['Original'] + list(reconstructable.keys())
    fig, axes = plt.subplots(n_models + 1, NUM_SAMPLES, figsize=(NUM_SAMPLES * 1.5 + 1, (n_models + 1) * 1.5))
    fig.suptitle('Reconstructions', fontsize=14, y=1.01)

    # Row 0: originals
    for i in range(NUM_SAMPLES):
        axes[0, i].imshow(x[i].cpu().view(28, 28), cmap='gray')
        axes[0, i].axis('off')

    for row, (name, model) in enumerate(reconstructable.items(), start=1):
        model.eval()
        with torch.no_grad():
            if isinstance(model, FlowMatching):
                recons = flow_matching_reconstruct(model, x, steps=100)
            else:
                outputs = model(x)
                logits = outputs[0]
                recons = torch.sigmoid(logits)

        for i in range(NUM_SAMPLES):
            axes[row, i].imshow(recons[i].cpu().view(28, 28), cmap='gray')
            axes[row, i].axis('off')

    # Add row labels via fig.text (axis('off') hides set_ylabel)
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


def show_samples(models: dict, device: torch.device):
    """Show samples generated from random latent vectors."""
    n_models = len(models)
    fig, axes = plt.subplots(n_models, NUM_SAMPLES, figsize=(NUM_SAMPLES * 1.5 + 1, n_models * 1.5))
    fig.suptitle('Generated Samples (from random z)', fontsize=14, y=1.01)

    for row, (name, model) in enumerate(models.items()):
        if isinstance(model, Generator):
            samples = model.sample(NUM_SAMPLES)          # derives device from weights
        elif isinstance(model, FlowMatching):
            samples = model.sample(NUM_SAMPLES, steps=500)
        else:
            samples = model.sample(NUM_SAMPLES, device)
        for i in range(NUM_SAMPLES):
            ax = axes[row, i] if n_models > 1 else axes[i]
            ax.imshow(samples[i].cpu().view(28, 28), cmap='gray')
            ax.axis('off')

    # Add row labels via fig.text (axis('off') hides set_ylabel)
    for row, name in enumerate(models.keys()):
        y = 1 - (row + 0.5) / n_models
        fig.text(0.01, y, name, va='center', ha='left', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / 'samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved samples       -> {OUTPUT_DIR}/samples.png')


def show_loss_curves(loss_curves: dict[str, list]):
    """Plot training loss over epochs for each model.
    GAN stores (g_loss, d_loss) tuples — plotted as two separate lines.
    """
    plt.figure(figsize=(8, 4))
    for name, losses in loss_curves.items():
        if losses and isinstance(losses[0], tuple):  # GAN: list[(g_loss, d_loss)]
            g_losses = [g for g, d in losses]
            d_losses = [d for g, d in losses]
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


# ── Evaluation & Summary ───────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Reconstruction BCE loss per sample on the validation set (no KL term)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    is_fm = isinstance(model, FlowMatching)
    for batch in loader:
        x = get_inputs(batch).to(device)
        x = flatten_images(x)
        if is_fm:
            x_recon = flow_matching_reconstruct(model, x, steps=10)
            loss_val = F.binary_cross_entropy(torch.clamp(x_recon, 1e-7, 1.0 - 1e-7), x, reduction='sum').item()
        else:
            logits = model(x)[0]  # first output is always logits
            loss_val = F.binary_cross_entropy_with_logits(logits, x, reduction='sum').item()
        total_loss += loss_val
        total_samples += x.size(0)
    return total_loss / total_samples


def print_summary(results: dict[str, dict]):
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = pick_device()
    print(f'Using device: {device}')

    train_loader, val_loader, input_dim = load_mnist(BATCH_SIZE)

    # AE loss_fn ignores the z returned by forward()
    ae_loss_fn = lambda x, logits, _z: ae_loss(x, logits)

    models_cfg = {
        'VAE': (VAE(input_dim, HIDDEN_DIM, LATENT_DIM), vae_loss),
        'AE':  (AE(input_dim, HIDDEN_DIM, LATENT_DIM),  ae_loss_fn),
        'FlowMatching': (FlowMatching(input_dim, HIDDEN_DIM), None),
    }

    loss_curves = {}
    trained_models = {}
    results = {}

    for name, (model, loss_fn) in models_cfg.items():
        print(f'\nTraining {name}...')
        model.to(device)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

        t0 = time.perf_counter()
        if isinstance(model, FlowMatching):
            losses = fm_train(model, loader=train_loader, optimizer=optimizer, device=device, epochs=EPOCHS)
        else:
            losses = train(model, loader=train_loader, optimizer=optimizer, device=device, loss_fn=loss_fn, epochs=EPOCHS)
        elapsed = time.perf_counter() - t0

        loss_curves[name] = losses
        trained_models[name] = model
        print(f'{name} final train loss: {losses[-1]:.2f}  ({elapsed:.1f}s)')

        val_loss = evaluate(model, val_loader, device)
        results[name] = {
            'Val Recon Loss': f'{val_loss:.2f}',
            '# Params': f'{count_params(model):,}',
            'Train Time': f'{elapsed:.1f}s',
        }

    # ── GAN (two models, optimizers created inside gan_train) ──────────────────
    print('\nTraining GAN...')
    generator = Generator(input_dim, HIDDEN_DIM)
    discriminator = Discriminator(input_dim, HIDDEN_DIM)
    generator.to(device)
    discriminator.to(device)

    t0 = time.perf_counter()
    gan_losses = gan_train(generator, discriminator, train_loader, device, epochs=EPOCHS)
    elapsed = time.perf_counter() - t0

    g_final, d_final = gan_losses[-1]
    print(f'GAN final train loss: G={g_final:.4f}, D={d_final:.4f}  ({elapsed:.1f}s)')

    loss_curves['GAN'] = gan_losses       # list[(g_loss, d_loss)]
    trained_models['GAN'] = generator    # only generator needed for sampling
    results['GAN'] = {
        'Val Recon Loss': 'N/A',         # GAN has no encoder, can't reconstruct
        '# Params': f'{count_params(generator) + count_params(discriminator):,}',
        'Train Time': f'{elapsed:.1f}s',
    }

    show_loss_curves(loss_curves)
    show_reconstructions(trained_models, train_loader, device)
    show_samples(trained_models, device)
    print_summary(results)


if __name__ == '__main__':
    main()
