import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Float, Array, Int, jaxtyped, PRNGKeyArray
from beartype import beartype as typechecker


from attention import MultiHeadAttention


class RMSNorm(eqx.Module):
    weight: Float[Array, "features"]
    eps: float = eqx.field(static=True)

    def __init__(self, input_size: int, eps: float = 1e-6):
        self.weight = jnp.ones(input_size)
        self.eps = eps

    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[Array, "features"]) -> Float[Array, "features"]:
        rms = jnp.sqrt(jnp.mean(x ** 2) + self.eps)
        return (x / rms) * self.weight


class Embedding(eqx.Module):
    weight: Float[Array, "num_embeddings embedding_dim"]

    def __init__(self, 
        num_embeddings: int, 
        embedding_dim: int,
        *, 
        key: PRNGKeyArray, 
        scale: float = 0.02,
    ) -> None:
        self.weight = jax.random.normal(key, (num_embeddings, embedding_dim)) * scale

    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Int[Array, ""]) -> Float[Array, "embedding_dim"]:
        return self.weight[x]


class RoPE(eqx.Module):
    head_dim: int = eqx.field(static=True)
    theta: float = eqx.field(static=True)

    def __init__(self, head_dim: int, theta: float = 1e4):
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")

        self.head_dim = head_dim
        self.theta = theta

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        x: Float[Array, "seq_len head_dim"],
        positions: Int[Array, "seq_len"],    
    ) -> Float[Array, "seq_len head_dim"]:
        i = jnp.arange(0, self.head_dim, 2)
        freqs = self.theta ** (-i / self.head_dim)  # head_dim // 2
        # freqs = theta^0, theta^(-2/hd),  theta^(-4/hd), ..., ~theta^(-1)

        angles = positions[:, None] * freqs[None, :]   # (seq_len, head_dim // 2)
        cos = jnp.repeat(jnp.cos(angles), 2, axis=-1)  # (seq_len, head_dim)
        sin = jnp.repeat(jnp.sin(angles), 2, axis=-1)  # (seq_len, head_dim)

        x1, x2 = x[..., 0::2], x[..., 1::2]
        rotated = jnp.stack([-x2, x1], axis=-1).reshape(x.shape)

        return x * cos + rotated * sin



class FeedForward(eqx.Module):
    w1: Array[Float, "embedding_dim hidden_dim"]
    w2: Array[Float, "hidden_dim embedding_dim"]
    activation: Callable[[Float[Array, "hidden_dim"]], Float[Array, "hidden_dim"]]

    def __init__(self,
        embedding_dim: int, 
        hidden_dim: int, 
        key: PRNGKeyArray, 
        activation: Callable[[Float[Array, "hidden_dim"]], Float[Array, "hidden_dim"]]
    ):
        self.w1 = jax.random.normal(key, (embedding_dim, hidden_dim))
        self.w2 = jax.random.normal(key, (hidden_dim, embedding_dim))
        self.activation = activation

    def __call__(self, x: Float[Array, "batch embedding_dim"]) -> Float[Array, "batch embedding_dim"]:
        x = self.w1(x)
        x = self.activation(x)
        x = self.w2(x)
        return x