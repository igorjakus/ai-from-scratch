import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.training = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return x * torch.rand_like(x) * (1 / (1 - self.p))
        return x * (1 - self.p)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"

    def eval(self) -> None:
        self.training = False
        return self
    
    def train(self, mode: bool = True) -> 'Dropout':
        self.training = mode
        return self