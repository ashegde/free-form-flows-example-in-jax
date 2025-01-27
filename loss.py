from typing import Callable
import jax
import jax.numpy as jnp

def fff_components_single(
    key:  jax.random.key,
    predict_fn: Callable,
    encoder_params: list[jnp.array],
    decoder_params: list[jnp.array],
    x: jnp.array,
):
    """
    Free-form flow (fff) loss function, as defined in Algorithm 1 of the
    primary reference and derived in Appendix A.2. 
    """

    # note, x is assumed of dimension (d,) 
    # also, jacobian calculations are wrt inputs (not parameters)
    
    def encoder_fn(x1: jnp.array):
       return predict_fn(encoder_params, x1)
    
    def decoder_fn(z1: jnp.array):
        return predict_fn(decoder_params, z1)
    
    # Hutchinson trace approximation works better with test vectors v on \sqrt{dim}-sphere.
    # https://www.ethanepperly.com/index.php/2024/01/28/dont-use-gaussians-in-stochastic-trace-estimation/
    #
    # following the main reference, we will just use a single test vector

    
    v = jax.random.normal(key, shape=x.shape)
    v *= jnp.sqrt(v.shape[-1]) / jnp.sqrt(jnp.square(v).sum(axis=-1, keepdims=True))
    
    z, func_vjp = jax.vjp(encoder_fn, x)
    v1 = func_vjp(v)[0]
    xr, v2 = jax.jvp(decoder_fn, [z,], [v,])
 
    
    # v, z, v1, xr, v2 are all lists containing a (B, 1, d) array
    log_jac_det = jax.lax.stop_gradient(v2) * v1
    nll = 0.5 * jnp.square(z).sum(axis=-1) - log_jac_det.sum(axis=-1)
    L_reconstr = jnp.square(xr - x).sum(axis=-1)

    return nll, L_reconstr

