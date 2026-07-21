import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import get_inputs, flatten_images


class FlowMatching(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim

        self.sequential = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x_t: Tensor, t: Tensor):
        B, N = x_t.shape
        t = t.view(B, 1)
        v = self.sequential(torch.cat((x_t, t), dim=1))
        return v

    @torch.inference_mode()
    def sample(self, num_samples: int, steps: int) -> Tensor:
        device = self.sequential[0].weight.device
        x = torch.randn(num_samples, self.input_dim, device=device)
        dt = torch.tensor(1.0 / steps, device=device)
        t = torch.zeros(num_samples, device=device)

        # integrate using Euler's method
        # x_1 = x_0 + \int_0^1 v(t) dts
        for _ in range(steps):
            v = self.forward(x, t)
            x = x + v * dt
            t = t + dt

        return torch.clamp(x, 0.0, 1.0)


def cfm_loss(v_pred: Tensor, x_0: Tensor, x_1: Tensor) -> Tensor:
    """
    Conditional Flow Matching loss

    Let x_0 be noise and x_1 be data. We sample a random time t and interpolate:
        x(t) = (1 - t) * x_0 + t * x_1

    The true velocity for a linear path from x_0 to x_1 is constant:
        v(t) = dx/dt = -x_0 + x_1 = x_1 - x_0 

    We minimize MSE between the predicted and true velocity.
    """
    v_target = x_1 - x_0
    return F.mse_loss(v_pred, v_target, reduction='mean')


def fm_train(
    model: FlowMatching,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 1,
) -> list[float]:
    model.train()
    epoch_losses = []

    for _ in tqdm(range(epochs), desc='Epochs'):
        total_loss = 0.0
        total_samples = 0

        for batch in loader:
            x_1 = get_inputs(batch).to(device)
            x_1 = flatten_images(x_1)
            B = x_1.size(0)

            # sample noise and a random time for each element in the batch
            x_0 = torch.randn_like(x_1)
            t = torch.rand(B, device=device)

            # linear interpolation: straight path from noise to data
            x_t = (1 - t.view(B, 1)) * x_0 + t.view(B, 1) * x_1

            v_pred = model(x_t, t)
            loss = cfm_loss(v_pred, x_0, x_1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

        epoch_losses.append(total_loss / total_samples)

    return epoch_losses
