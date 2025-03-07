import jax
import jax.numpy as jnp


def random_layer_params(
        key: jax.random.key,
        in_dim: int,
        out_dim: int,
        b_scale: float = 0.0,
) -> tuple[jnp.array]:
    
    w_key, b_key = jax.random.split(key, 2)
    return  (
        jax.random.uniform(
           w_key,
           shape=(out_dim, in_dim),
           minval=-jnp.sqrt(1/in_dim),
           maxval=jnp.sqrt(1/in_dim)
        ),
      b_scale * jax.random.uniform(
           b_key,
           shape=(out_dim,),
           minval=-jnp.sqrt(1/in_dim),
           maxval=jnp.sqrt(1/in_dim),
        ),
    )

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(key: jax.random.key, layer_sizes: list[int], b_scale: int = 0.0) -> list[int]:
    keys = jax.random.split(key, len(layer_sizes))
    return [
        random_layer_params(k, in_dim, out_dim, b_scale) 
        for in_dim, out_dim, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)
    ]

def predict_single(params: jnp.array, x: jnp.array) -> jnp.array:
    def relu(z: jnp.array):
        return jnp.maximum(0, z)
    def tanh(z: jnp.array):
        return jnp.tanh(z)
    def sigmoid(z: jnp.array):
        return 1/(1 + jnp.exp(-z))
    def silu(z: jnp.array):
        return z * sigmoid(z)
    def elu(z: jnp.array, alpha: float = 1.0):
        return jnp.where(z > 0, z, alpha*(jnp.exp(z)-1))
    
    h = x
    readin_w, readin_b = params[0]
    u = jnp.dot(readin_w, h) + readin_b
    h = elu(u)

    for (w,b) in params[1:-1]:
        u = jnp.dot(w, h) + b
        h = elu(u)
    
    readout_w, readout_b = params[-1]
    return jnp.dot(readout_w, h) + readout_b