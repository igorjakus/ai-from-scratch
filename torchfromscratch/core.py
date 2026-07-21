class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True)
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})"


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

    def __repr__(self) -> str:
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"


class Sequential(nn.Module):
    def __init__(self, *modules: nn.Module) -> None:
        super().__init__()
        self.modules = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.modules:
            x = module(x)
        return x

    def __repr__(self) -> str:
        return f"Sequential({', '.join(repr(module) for module in self.modules)})"

    def eval(self) -> None:
        for module in self.modules:
            module.eval()
        return self
    
    def train(self, mode: bool = True) -> 'Sequential':
        for module in self.modules:
            module.train(mode)
        return self