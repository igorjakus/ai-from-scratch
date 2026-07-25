from beartype import beartype as typechecker
from jaxtyping import Array, Float, Int, PRNGKeyArray, jaxtyped
from jax import random
from jax import numpy as jnp


@jaxtyped(typechecker=typechecker)
def jax_categorical(
    key: PRNGKeyArray,
    logits: Float[Array, "... vocab"],
) -> Int[Array, "..."]:
    """Sample from a categorical using JAX's built-in.

    Reference / sanity-check baseline. JAX implements `random.categorical`
    internally via the Gumbel-max trick, so this should match `gumbel_max_categorical`
    in both output distribution and runtime.
    """
    return jax.random.categorical(key, logits, axis=-1)


@jaxtyped(typechecker=typechecker)
def gumbel_max_categorical(
    key: PRNGKeyArray,
    logits: Float[Array, "... vocab"],
) -> Int[Array, "..."]:
    """Sample via the Gumbel-max trick.

    Algorithm:
        for each category k: g_k ~ Gumbel(0, 1)  (independent draws)
        return argmax_k (logits_k + g_k)

    Equivalently: sample u ~ Uniform(0, 1), then g = -log(-log(u)).
    The argmax over (logits + g) is an exact sample from softmax(logits).

    Reference: https://lips.cs.princeton.edu/the-gumbel-max-trick/
    """
    vocab_size = logits.shape[-1]
    u = jax.random.uniform(key, shape=(vocab_size,), minval=1e-8, maxval=1-1e-8)
    g = -jax.log(-jax.log(u))
    return jax.argmax(logits + g, axis=-1)


@jaxtyped(typechecker=typechecker)
def cumsum_categorical(
    key: PRNGKeyArray,
    logits: Float[Array, "... vocab"],
) -> Int[Array, "..."]:
    """Sample via softmax -> cumsum -> binary search (baseline).

    Algorithm:
        probs = softmax(logits)            # V exponentials + a reduction
        cdf   = cumsum(probs)              # O(V) reduction
        u     = uniform(0, 1)              # one sample per leading axis
        return searchsorted(cdf, u)        # O(log V) binary search per row

    The "classical" approach. Expected to be slower than Gumbel-max at large V
    because of the softmax + cumsum, but conceptually simpler.
    """
    vocab_size = logits.shape[-1]
    probs = jax.nn.softmax(logits, axis=-1)
    cdf = jnp.cumsum(probs, axis=-1)
    u = random.uniform(key, vocab_size)
    return jnp.searchsorted(cdf, u, side='left')

