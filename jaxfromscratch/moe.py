import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped
from beartype import beartype as typechecker
from typing import Callable, Tuple


class FeedForward(eqx.Module):
    up_proj: eqx.nn.Linear
    down_proj: eqx.nn.Linear
    activation: Callable[[Float[Array, "hidden_dim"]], Float[Array, "hidden_dim"]] = eqx.field(static=True)

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        *,
        key: PRNGKeyArray,
        activation: Callable[[Float[Array, "hidden_dim"]], Float[Array, "hidden_dim"]]
    ):
        key1, key2 = jax.random.split(key)
        self.up_proj = eqx.nn.Linear(model_dim, hidden_dim, key=key1)
        self.down_proj = eqx.nn.Linear(hidden_dim, model_dim, key=key2)
        self.activation = activation

    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[Array, "model_dim"]) -> Float[Array, "model_dim"]:
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return x


class Router(eqx.Module):
    proj: eqx.nn.Linear
    num_experts: int = eqx.field(static=True)
    model_dim: int = eqx.field(static=True)
    top_k: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    def __init__(self, num_experts: int, model_dim: int, *, top_k: int, alpha: float, key: PRNGKeyArray):
        self.proj = eqx.nn.Linear(model_dim, num_experts, key=key)
        self.num_experts = num_experts
        self.model_dim = model_dim
        self.top_k = top_k
        self.alpha = alpha

    def __call__(self, x: Float[Array, "model_dim"]) -> Tuple[Float[Array, "num_experts"], Float[Array, ""]]:
        scores = self.proj(x)                                # num_experts
        values, indices = jax.lax.top_k(scores, self.top_k)  # top_k, top_k
        masked_scores = jnp.full_like(scores, -jnp.inf).at[indices].set(values)  # num_experts
        weights = jax.nn.softmax(masked_scores)  # num_experts

        balanced_weights = jnp.full_like(weights, 1 / self.num_experts)
        aux_loss = -jnp.sum(balanced_weights * jnp.log(weights + 1e-9)) * self.alpha

        return weights, aux_loss



class MoE(eqx.Module):
    router: Router 
    experts: FeedForward # vmap of FeedForward
    num_experts: int = eqx.field(static=True)
    top_k: int = eqx.field(static=True)

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        *,
        key: PRNGKeyArray,
        ffn_activation: Callable[[Float[Array, "model_dim"]], Float[Array, "hidden_dim"]],
        router_alpha: float
    ):
        if top_k > num_experts:
            raise ValueError(f"top_k must be <= num_experts, got {top_k} and {num_experts}")

        router_key, experts_key = jax.random.split(key, 2)
        expert_keys = jax.random.split(experts_key, num_experts)

        self.router = Router(num_experts, model_dim, top_k=top_k, alpha=router_alpha, key=router_key)
        self.experts = eqx.filter_vmap(lambda k: FeedForward(model_dim, hidden_dim, key=k, activation=ffn_activation))(expert_keys)
        self.num_experts = num_experts
        self.top_k = top_k

    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[Array, "model_dim"]) -> Tuple[Float[Array, "model_dim"], Float[Array, ""]]:
        weights, router_aux_loss = self.router(x)

        y = jax.vmap(lambda expert: expert(x))(self.experts)  # num_experts, model_dim
        z = jnp.einsum("e,ed->d", weights, y)  # model_dim

        return z, router_aux_loss

