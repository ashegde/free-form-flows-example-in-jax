"""
In this file, we consider a simple example of normalizing flows, based on the free-form formulation in, 

Draxler, F., Sorrenson, P., Zimmermann, L., Rousselot, A., & Köthe, U. (2024, April). 
Free-form flows: Make any architecture a normalizing flow. 
In International Conference on Artificial Intelligence and Statistics (pp. 2197-2205). PMLR.

As documented in the reference, the advantage of this formulation is that it avoids the need to explicitly use
invertible neural networks. Rather, this is enforced through a penalty that promotes reconstruction.

In the following code, we will use only basic jax code as a learning exersize, akin to

https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html

Despite only using a simple MLP and SGD, we are able to generate non-trivial results 
-- although after substantial tuning of certain hyper-parameters, such as beta and the network.  
"""

import time
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

from prepare_data import generate_dataset
from model import init_network_params, predict_single
from loss import fff_components_single

# Settings and constants
key = jax.random.key(0)
NUMBER_OF_EPOCHS = 1200
BATCH_SIZE = 256
DATA_SIZE = 10000
LEARNING_RATE = 1e-3
BETA = 20

# Define model

layer_sizes = [2, 32, 64, 64, 32, 2]
b_scale = 1e-3

key, subkey1, subkey2 = jax.random.split(key, 3)
encoder_params = init_network_params(subkey1, layer_sizes, b_scale)
decoder_params = init_network_params(subkey2, layer_sizes, b_scale)

# Define batch loss

batch_fff_components = jax.vmap(fff_components_single, in_axes=(0, None, None, None, 0))

def batch_loss(
    key:  jax.random.key,
    encoder_params: list[jnp.array],
    decoder_params: list[jnp.array],
    x: jnp.array,
):
    keys = jax.random.split(key, x.shape[0])
    nll, L_reconstr = batch_fff_components(keys, predict_single, encoder_params, decoder_params, x)
    return (nll + BETA * L_reconstr).mean()

# SGD update

@jax.jit
def sgd_update(
    key:  jax.random.key,
    encoder_params: list[jnp.array],
    decoder_params: list[jnp.array],
    x: jnp.array,
    lr: float,
)->tuple[list]:
    grad_fn = jax.grad(batch_loss, argnums=(1, 2))
    encoder_grads, decoder_grads = grad_fn(key, encoder_params, decoder_params, x)

    return (
        [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(encoder_params, encoder_grads)],
        [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(decoder_params, decoder_grads)],
    )

# Prepare data

train_data = jnp.array(generate_dataset(n=DATA_SIZE))
valid_data = jnp.array(generate_dataset(n=100))

# Training loop

def batch_iterate(key: jax.random.key, batch_size: int, x: jnp.array):
    perm = jnp.array(jax.random.permutation(key, x.shape[0]))
    for s in range(0, x.shape[0], batch_size):
        ids = perm[s : s + batch_size]
        yield x[ids]

losses = []
lr = LEARNING_RATE
lr_decay = 0.99
for e in range(NUMBER_OF_EPOCHS):
    key, subkey1 = jax.random.split(key, 2)
    start_time = time.time()
    for xb in batch_iterate(subkey1, BATCH_SIZE, train_data):
        key, subkey2 = jax.random.split(key, 2)
        encoder_params, decoder_params = sgd_update(
            subkey2,
            encoder_params,
            decoder_params,
            xb,
            lr,
        )

    subkeys = jax.random.split(key, train_data.shape[0]+1)
    key = subkeys[0]
    train_nll, train_reconstr = batch_fff_components(
        subkeys[1:],
        predict_single,
        encoder_params,
        decoder_params,
        train_data,
    )
    train_nll = train_nll.mean()
    train_reconstr = train_reconstr.mean()
    train_loss = (train_nll + BETA * train_reconstr).mean()

    subkeys = jax.random.split(key, valid_data.shape[0]+1)
    key = subkeys[0]
    valid_nll, valid_reconstr = batch_fff_components(
        subkeys[1:],
        predict_single,
        encoder_params,
        decoder_params,
        valid_data,
    )
    valid_nll = valid_nll.mean()
    valid_reconstr = valid_reconstr.mean()
    valid_loss = (valid_nll + BETA * valid_reconstr).mean()
    epoch_time = time.time() - start_time
    print(
        f"Epoch {e} | lr: {lr:.2e} | Train nll, reconstr, loss: {train_nll.item():.3f}, {train_reconstr.item():.3f}, {train_loss.item():.3f} | Valid loss {valid_loss.item():.3f} | Epoch time {epoch_time:.4f}s",
    )
    lr *= lr_decay


# Plot results
batch_predict = jax.vmap(predict_single, in_axes=(None, 0))

num_samples = 1000
key, subkey = jax.random.split(key, 2)
z = jax.random.normal(subkey, shape=(num_samples, 2))
x_gen = batch_predict(decoder_params, z)

fig = plt.figure(figsize=(6,5))
plt.scatter(x_gen[:, 0], x_gen[:, 1], c='blue', marker='.', label='decoded/generated')
plt.scatter(z[:, 0], z[:, 1], marker='o', facecolors='none', edgecolors='black', alpha=0.2, label='latents')
plt.scatter(train_data[:, 0], train_data[:, 1], c='red', marker='.', alpha = 1e-2, label='data')
leg = plt.legend()
for lhandle in leg.legend_handles: 
    lhandle.set_alpha(1)
plt.title('Generated samples vs. original data')
plt.savefig('generated_samples.png')