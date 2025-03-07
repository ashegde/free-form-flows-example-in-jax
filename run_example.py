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

from loss import batch_loss_components
from model import init_network_params, predict_single
from optimizer import adamw_step, clip_grads, moving_average_step
from prepare_data import generate_dataset

# Settings and constants
key = jax.random.key(0)
NUMBER_OF_EPOCHS = 1000
BATCH_SIZE = 512
DATA_SIZE = 10000
LEARNING_RATE = 1e-3
BETA = 1e2

# Define model

layer_sizes = [2, 128, 128, 128, 128, 2]
b_scale = 1.0

key, subkey1, subkey2 = jax.random.split(key, 3)
encoder_params = init_network_params(subkey1, layer_sizes, b_scale)
decoder_params = init_network_params(subkey2, layer_sizes, b_scale)
params = encoder_params + decoder_params

# Define batch loss

def batch_loss(
    params: list[jnp.array],
    x: jnp.array,
    v: jnp.array,
):
    nll, L_reconstr = batch_loss_components(params, x, v)
    return (nll + BETA * L_reconstr).mean()

# Prepare data

train_data = jnp.array(generate_dataset(n=DATA_SIZE))
valid_data = jnp.array(generate_dataset(n=100))

# Training loop

key, subkey1, subkey2 = jax.random.split(key, 3)
encoder_params = init_network_params(subkey1, layer_sizes, b_scale)
decoder_params = init_network_params(subkey2, layer_sizes, b_scale)
params = encoder_params + decoder_params

# training loop

train_losses = []
valid_losses = []

lr = LEARNING_RATE
scale = 0.999999
lr_min = 1e-5

# optimization
beta1 = 0.9
beta2 = 0.999
weight_decay = 1e-6

momentum = [(jnp.zeros_like(w), jnp.zeros_like(b)) for w,b in params]
velocity = [(jnp.zeros_like(w), jnp.zeros_like(b)) for w,b in params]

value_and_grad_fn = jax.value_and_grad(batch_loss, argnums=0)

@jax.jit
def update(
    params: list[jnp.array],
    x: jnp.array,
    v: jnp.array,
    momentum: list[jnp.array],
    velocity: list[jnp.array],
    lr: float,
    epoch: int,
)->tuple:
    
    loss_val, grads = value_and_grad_fn(params, x, v)
    clipped_grads = clip_grads(grads, 1.0)

    # adamW, based on:
    # Loshchilov, I., & Hutter, F. (2017). Fixing weight decay regularization in adam.
    # arXiv preprint arXiv:1711.05101, 5.

    momentum = moving_average_step(momentum, clipped_grads, beta1)
    velocity = moving_average_step(velocity, [(w**2, b**2) for w, b in clipped_grads], beta2)
    
    # bias corrections for moving average initializations
    mhat = [(w / (1-beta1**(epoch+1)),  b / (1-beta1**(epoch+1))) for w, b in momentum]
    vhat = [(w / (1-beta2**(epoch+1)),  b / (1-beta2**(epoch+1))) for w, b in velocity]

    return loss_val, adamw_step(params, mhat, vhat, lr, weight_decay, eps=1e-7), momentum, velocity

def batch_iterate(key: jax.random.key, batch_size: int, x: jnp.array):
    perm = jnp.array(jax.random.permutation(key, x.shape[0]))
    for s in range(0, x.shape[0], batch_size):
        ids = perm[s : s + batch_size]
        yield x[ids]

for epoch in range(NUMBER_OF_EPOCHS):
    key, subkey1 = jax.random.split(key, 2)
    start_time = time.time()
    for xb in batch_iterate(subkey1, BATCH_SIZE, train_data):
        key, subkey = jax.random.split(key, 2)

        # Hutchinson trace approximation works better with test vectors v on \sqrt{dim}-sphere.
        # https://www.ethanepperly.com/index.php/2024/01/28/dont-use-gaussians-in-stochastic-trace-estimation/
        #
        # following the main reference, we will just use a single test vector
        vb = jax.random.normal(subkey, shape=xb.shape)
        vb *= jnp.sqrt(vb.shape[-1]) / jnp.sqrt(jnp.square(vb).sum(axis=-1, keepdims=True))

        loss_val, params, momentum, velocity = update(
            params,
            xb,
            vb,
            momentum,
            velocity,
            lr,
            epoch,
        )

    
    # full training loss
    key, subkey = jax.random.split(key, 2)
    v = jax.random.normal(subkey, shape=train_data.shape)
    v *= jnp.sqrt(v.shape[-1]) / jnp.sqrt(jnp.square(v).sum(axis=-1, keepdims=True))

    train_nll, train_reconstr = batch_loss_components(
        params,
        train_data,
        v,
    )
    train_nll = train_nll.mean()
    train_reconstr = train_reconstr.mean()
    train_loss = train_nll + BETA * train_reconstr

    # full validation loss
    key, subkey = jax.random.split(key, 2)
    v = jax.random.normal(subkey, shape=valid_data.shape)
    v *= jnp.sqrt(v.shape[-1]) / jnp.sqrt(jnp.square(v).sum(axis=-1, keepdims=True))

    valid_nll, valid_reconstr = batch_loss_components(
        params,
        valid_data,
        v,
    )
    valid_nll = valid_nll.mean()
    valid_reconstr = valid_reconstr.mean()
    valid_loss = valid_nll + BETA * valid_reconstr
    
    train_losses.append(train_loss.item())
    valid_losses.append(valid_loss.item())
        
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch} | lr: {lr:.2e} | Train nll, reconstr, loss: {train_nll.item():.3f}, {train_reconstr.item():.3f}, {train_loss.item():.3f} | Valid loss {valid_loss.item():.3f} | Epoch time {epoch_time:.4f}s")
    lr = max(lr*scale, lr_min)


# Plot results
batch_predict = jax.vmap(predict_single, in_axes=(None, 0))

encoder_depth = len(params) // 2
encoder_params = params[:encoder_depth]
decoder_params = params[encoder_depth:]

num_samples = 1000
key, subkey = jax.random.split(key, 2)
z = jax.random.normal(subkey, shape=(num_samples, 2))
x_gen = batch_predict(decoder_params, z)

plt.scatter(x_gen[:, 0], x_gen[:, 1], c='blue', marker='.', label='decoded/generated')
plt.scatter(z[:, 0], z[:, 1], marker='o', facecolors='none', edgecolors='black', alpha=0.2, label='latents')
plt.scatter(train_data[:, 0], train_data[:, 1], c='red', marker='.', alpha = 1e-2, label='data')
leg = plt.legend()
for lhandle in leg.legend_handles: 
    lhandle.set_alpha(1)
plt.title('Generated samples vs. original data')
plt.savefig('generated_samples.png')
