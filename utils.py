import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_inputs(batch: Tensor | list[Tensor] | tuple[Tensor, ...]) -> Tensor:
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def flatten_images(x: Tensor) -> Tensor:
    if x.dim() > 2:
        return x.view(x.size(0), -1)
    return x


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn,
    epochs: int = 1,
) -> list[float]:
    """Generic training loop. loss_fn(x, *model(x)) must return a scalar."""
    model.train()
    epoch_losses = []

    for _ in tqdm(range(epochs), desc='Epochs'):
        total_loss = 0.0
        for batch in loader:
            x = get_inputs(batch).to(device)
            x = flatten_images(x)

            outputs = model(x)
            loss = loss_fn(x, *outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_losses.append(total_loss / len(loader))

    return epoch_losses
