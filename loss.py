from typing import Callable
import jax
import jax.numpy as jnp

from model import predict_single

def loss_components_single(
    params: list[jnp.array],
    x: jnp.array,
    v: jnp.array
):
    """
    Free-form flow (fff) loss function, as defined in Algorithm 1 of the
    primary reference and derived in Appendix A.2. 
    """

    # note, x is assumed of dimension (d,) 
    # also, jacobian calculations are wrt inputs (not parameters)
    
    encoder_depth = len(params) // 2

    def encoder_fn(x1: jnp.array):
       return predict_single(params[:encoder_depth], x1)
    
    def decoder_fn(z1: jnp.array):
        return predict_single(params[encoder_depth:], z1)
    
    z, func_vjp = jax.vjp(encoder_fn, x)
    v1 = func_vjp(v)[0]
    xr, v2 = jax.jvp(decoder_fn, [z,], [v,])
 
    # v, z, v1, xr, v2 are all lists containing a (B, 1, d) array
    log_jac_det = jax.lax.stop_gradient(v2) * v1
    nll = 0.5 * jnp.square(z).sum(axis=-1) - log_jac_det.sum(axis=-1)
    L_reconstr = jnp.square(xr - x).sum(axis=-1)

    return nll, L_reconstr

# define vmap over rng keys and batches
batch_loss_components = jax.vmap(loss_components_single, in_axes=(None, 0, 0))
