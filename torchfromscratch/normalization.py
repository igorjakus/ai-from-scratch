import torch

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.var(dim=-1, unbiased=False) + self.eps
        norm = x * (var + self.eps).rsqrt()  # rsqrt = 1/ sqrt(var)
        return norm * self.scale
    
    def __repr__(self) -> str:
        return f"RMSNorm(scale={self.scale.shape}, eps={self.eps})"


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) + self.eps
        norm = (x - mean) * (var + self.eps).rsqrt()  # rsqrt = 1/ sqrt(var)
        return norm * self.scale + self.bias
    
    def __repr__(self) -> str:
        return f"LayerNorm(scale={self.scale.shape}, bias={self.bias.shape}, eps={self.eps})"


class BatchNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
        self.eps = eps
        self.training = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True, unbiased=False) + self.eps
            self.running_mean = self.running_mean * 0.99 + mean * 0.01
            self.running_var = self.running_var * 0.99 + var * 0.01
        else:
            mean = self.running_mean
            var = self.running_var
    
        norm = (x - mean) * (var + self.eps).rsqrt()  # rsqrt = 1/ sqrt(var)
        return norm * self.scale + self.bias

    def __repr__(self) -> str:
        return f"BatchNorm(scale={self.scale.shape}, bias={self.bias.shape}, running_mean={self.running_mean.shape}, running_var={self.running_var.shape}, eps={self.eps})"

    def eval(self) -> None:
        self.training = False
        return self
    
    def train(self, mode: bool = True) -> 'BatchNorm':
        self.training = mode
        return self

    