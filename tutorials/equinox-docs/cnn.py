import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, Int

class CNN(eqx.Module):
    layers: list

    def __init__(self, key: jax.random.KeyArray):
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.layers = [
            eqx.nn.Conv2d(key=key, in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(1728, 512, key=key2),
            jax.nn.sigmoid,
            eqx.nn.Linear(512, 64, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(64, 10, key=key4),
            jax.nn.log_softmax,
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "1 10"]:
        for layer in self.layers:
            x = layer(x)
        return x


def loss(model: CNN, x: Float[Array, "batch 1 28 28"], y: Int[Array, "batch"]) -> Float[Array, ""]:
    # input is (batch, 1, 28, 28)
    # model takes (1, 28, 28)
    # therefore we use jax.vmap which maps our model over the leading batch axis
    pred_y = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)

def cross_entropy(y: Int[Array, "batch"], probs: Float[Array, "batch 10"]) -> Float[Array, ""]:
    batch = y.shape[0]
    y_pred = probs[jnp.arange(batch), y]
    return -jnp.mean(pred_y)

def compute_accuracy(model: CNN, x: Float[Array, "batch 1 28 28"], y: Int[Array, "batch"]):
    probs = jax.vmap(model)(x)  # batch 10
    pred_y = jnp.argmax(pred_y, axis=1)  # batch
    return jnp.mean(y == pred_y)

def train(
    model: CNN,
    trainloader: 
)