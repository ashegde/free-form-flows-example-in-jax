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
      jnp.sqrt(2/(in_dim)) * jax.random.normal(
           w_key,
           shape=(in_dim, out_dim),
        ),
      b_scale * jax.random.normal(
           b_key,
           shape=(out_dim,),
        ),
    )

def init_network_params(key: jax.random.key, layer_sizes: list[int], b_scale: int = 0.0) -> list[int]:
    keys = jax.random.split(key, len(layer_sizes))
    return [
        random_layer_params(k, in_dim, out_dim, b_scale) 
        for in_dim, out_dim, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)
    ]

def predict_single(params: jnp.array, x: jnp.array) -> jnp.array:
    # def relu(z: jnp.array):
    #     return jnp.maximum(0, z)
    def tanh(z: jnp.array):
        return jnp.tanh(z)
    # def sigmoid(z: jnp.array):
    #     return 1/(1 + jnp.exp(-z))
    # def selu(z: jnp.array):
    #     return z * sigmoid(z)
    
    h = x
    for (w,b) in params[:-1]:
        h = jnp.dot(h,w) + b
        # h = relu(h)  
        h = tanh(h)
    
    final_w, final_b = params[-1]
    return x + jnp.dot(h, final_w) + final_b