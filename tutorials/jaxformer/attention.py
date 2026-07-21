import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Array, Float, jaxtyped, PRNGKeyArray
from beartype import beartype as typechecker


class MultiHeadAttention(eqx.Module):
    qkv_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, model_dim: int, num_heads: int, *, key: PRNGKeyArray):
        key1, key2 = jax.random.split(key, 2)

        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim must be divisible by num_heads, but got {model_dim} and {num_heads}")

        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv_proj = eqx.nn.Linear(model_dim, 3 * model_dim, key=key1, use_bias=False)
        self.out_proj = eqx.nn.Linear(model_dim, model_dim, key=key2, use_bias=False)

    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[Array, "seq_len model_dim"]) -> Float[Array, "seq_len model_dim"]:
        seq_len, model_dim = x.shape
    
        qkv = jax.vmap(self.qkv_proj)(x)  # seq_len, 3 * model_dim
        q, k, v = jnp.split(qkv, 3, axis=-1)  #  3x (seq_len, model_dim)
        q = q.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)  # num_heads, seq_len, head_dim
        k = k.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)  # num_heads, seq_len, head_dim
        v = v.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)  # num_heads, seq_len, head_dim

        attn_scores = jnp.einsum("h i d, h j d -> h i j", q, k)  # num_heads, seq_len, seq_len
        attn_scores /= jnp.sqrt(self.head_dim)                   # num_heads, seq_len, seq_len
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))  # seq_len, seq_len
        attn_scores = jnp.where(causal_mask, attn_scores, -jnp.inf)  # num_heads, seq_len, seq_len
        attn_scores = jax.nn.softmax(attn_scores, axis=-1)       # num_heads, seq_len, seq_len

        out = jnp.einsum("h i j, h j d -> h i d", attn_scores, v)  # num_heads, seq_len, head_dim
        out = out.transpose(1, 0, 2).reshape(seq_len, model_dim)  # seq_len, model_dim
        out = jax.vmap(self.out_proj)(out)  # seq_len, model_dim
    
        return out


class GroupedQueryAttention(eqx.Module):
    """Multi-Head Attention but shares value and keys matrices across many heads.
    We have num_heads // group_size different groups."""
    q_proj: eqx.nn.Linear
    kv_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    kv_groups: int = eqx.field(static=True)
    kv_group_size: int = eqx.field(static=True)

    def __init__(self, model_dim: int, num_heads: int, kv_groups: int, *, key: PRNGKeyArray):
        key1, key2, key3 = jax.random.split(key, 3)

        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim must be divisible by num_heads, but got {model_dim} and {num_heads}")

        if num_heads % kv_groups != 0:
            raise ValueError(f"num_heads must be divisible by kv_groups, but got {num_heads} and {kv_groups}")

        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.kv_groups = kv_groups
        self.kv_group_size = num_heads // kv_groups

        self.q_proj = eqx.nn.Linear(model_dim, model_dim, key=key1, use_bias=False)
        self.kv_proj = eqx.nn.Linear(model_dim, 2 * self.kv_groups * self.head_dim, key=key2, use_bias=False)
        self.out_proj = eqx.nn.Linear(model_dim, model_dim, key=key3, use_bias=False)
    
    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[Array, "seq_len model_dim"]) -> Float[Array, "seq_len model_dim"]:
        seq_len, model_dim = x.shape

        q = jax.vmap(self.q_proj)(x)  # seq_len model_dim
        q = q.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)  # num_heads seq_len head_dim

        kv = jax.vmap(self.kv_proj)(x)  # seq_len,  2 * kv_groups * head_dim
        kv = kv.reshape(seq_len, self.kv_groups, 2 * self.head_dim).transpose(1, 0, 2)  # kv_groups, seq_len, 2*head_dim
        kv = jnp.repeat(kv, self.kv_group_size, axis=0)  # num_heads seq_len 2*head_dim
        k, v = jnp.split(kv, 2, axis=-1)  # 2x (num_heads seq_len head_dim)

        attn_scores = jnp.einsum("h i d, h j d -> h i j", q, k)
        attn_scores /= jnp.sqrt(self.head_dim)
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        attn_scores = jnp.where(causal_mask, attn_scores, -jnp.inf)  # num_heads seq_len seq_len
        attn_scores = jax.nn.softmax(attn_scores, axis=-1)

        out = jnp.einsum("h i j, h j d -> h i d", attn_scores, v)  # num_heads, seq_len, head_dim
        out = out.transpose(1, 0, 2).reshape(seq_len, model_dim)
        out = jax.vmap(self.out_proj)(out)

        return out



class MultiQueryAttention(eqx.Module):
    """Multi-Head Attention but shares value and keys matrices across ALL heads.
    The same as Grouped-Query Attention with 1 large group, group_size = num_heads"""
    q_proj: eqx.nn.Linear
    kv_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, model_dim: int, num_heads: int, *, key: PRNGKeyArray):
        key1, key2, key3 = jax.random.split(key, 3)

        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim must be divisible by num_heads, but got {model_dim} and {num_heads}")

        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.q_proj = eqx.nn.Linear(model_dim, model_dim, key=key1, use_bias=False)
        self.kv_proj = eqx.nn.Linear(model_dim, 2 * self.head_dim, key=key2, use_bias=False)
        self.out_proj = eqx.nn.Linear(model_dim, model_dim, key=key3, use_bias=False)
    
    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[Array, "seq_len model_dim"]) -> Float[Array, "seq_len model_dim"]:
        seq_len, model_dim = x.shape

        q = jax.vmap(self.q_proj)(x)  # seq_len model_dim
        q = q.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)  # num_heads seq_len head_dim

        kv = jax.vmap(self.kv_proj)(x)    # seq_len 2*head_dim
        k, v = jnp.split(kv, 2, axis=-1)  # 2x (seq_len head_dim) 

        attn_scores = jnp.einsum("h i d, j d -> h i j", q, k)
        attn_scores /= jnp.sqrt(self.head_dim)
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))  # seq_len, seq_len
        attn_scores = jnp.where(causal_mask, attn_scores, -jnp.inf)
        attn_scores = jax.nn.softmax(attn_scores, axis=-1)

        out = jnp.einsum("h i j, j d -> h i d", attn_scores, v)  # num_heads seq_len head_dim
        out = out.transpose(1, 0, 2).reshape(seq_len, model_dim) # seq_len model_dim
        out = jax.vmap(self.out_proj)(out)                            # seq_len model_dim
        
        return out


class MultiHeadLatentAttention(eqx.Module):
    """Multi Head Attention but we put tiny autoencoder in front of the K/V projections
    Simplified version without RoPE and query compression!
    
    MHA:   x ─┬─► W_K ─► K
              └─► W_V ─► V

    MLA:   x ──► W_DKV ─► c_KV ─┬─► W_UK ─► K
                                └─► W_UV ─► V
    """
    q_proj: eqx.nn.Linear
    kv_down_proj: eqx.nn.Linear
    k_up_proj: eqx.nn.Linear
    v_up_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, model_dim: int, num_heads: int, latent_dim: int, *, key: PRNGKeyArray):
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)

        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim must be divisible by num_heads, but got {model_dim} and {num_heads}")

        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = eqx.nn.Linear(model_dim, model_dim, key=key1, use_bias=False)
        self.kv_down_proj = eqx.nn.Linear(model_dim, latent_dim, key=key2, use_bias=False)
        self.k_up_proj = eqx.nn.Linear(latent_dim, model_dim, key=key3, use_bias=False)
        self.v_up_proj = eqx.nn.Linear(latent_dim, model_dim, key=key4, use_bias=False)
        self.out_proj = eqx.nn.Linear(model_dim, model_dim, key=key5, use_bias=False)

    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[Array, "seq_len model_dim"]) -> Float[Array, "seq_len model_dim"]:
        seq_len, model_dim = x.shape

        q = jax.vmap(self.q_proj)(x)  # seq_len model_dim

        # get k/v latent representations
        kv_latent = jax.vmap(self.kv_down_proj)(x)  # seq_len latent_dim
        
        # move k/v from latent space to model space
        k = jax.vmap(self.k_up_proj)(kv_latent)  # seq_len model_dim
        v = jax.vmap(self.v_up_proj)(kv_latent)  # seq_len model_dim

        # reshape q, k, v to num_heads seq_len head_dim
        q = q.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)  # num_heads seq_len head_dim
        k = k.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)  # num_heads seq_len head_dim
        v = v.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)  # num_heads seq_len head_dim

        # compute attention scores
        attn_scores = jnp.einsum("h i d, h j d -> h i j", q, k)
        attn_scores /= jnp.sqrt(self.head_dim)
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        attn_scores = jnp.where(causal_mask, attn_scores, -jnp.inf)
        attn_scores = jax.nn.softmax(attn_scores, axis=-1)

        out = jnp.einsum("h i j, h j d -> h i d", attn_scores, v)  # num_heads seq_len head_dim
        out = out.transpose(1, 0, 2).reshape(seq_len, model_dim) # seq_len model_dim
        out = jax.vmap(self.out_proj)(out)                            # seq_len model_dim
        
        return out


class SlidingWindowAttention(eqx.Module):
    """We only attend to the previous tokens in the window."""
    qkv_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    window_size: int = eqx.field(static=True)

    def __init__(self, model_dim: int, window_size: int, *, key: PRNGKeyArray):
        key1, key2 = jax.random.split(key, 2)

        self.window_size = window_size
        self.qkv_proj = eqx.nn.Linear(model_dim, 3 * model_dim, key=key1, use_bias=False)
        self.out_proj = eqx.nn.Linear(model_dim, model_dim, key=key2, use_bias=False)
    
    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[Array, "seq_len model_dim"]) -> Float[Array, "seq_len model_dim"]:
        seq_len, model_dim = x.shape
