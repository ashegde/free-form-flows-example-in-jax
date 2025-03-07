import jax
import jax.numpy as jnp

def moving_average_step(
    past_state: list[jnp.array],
    new_state: list[jnp.array],
    beta: float = 0.9,
) -> list[jnp.array]:
    return [
        (beta * w_ps + (1-beta) * w_ns, beta * b_ps +(1-beta) * b_ns)
        for (w_ps, b_ps), (w_ns, b_ns) in zip(past_state, new_state)
    ]
    
def clip_grads(grads: list[jnp.array], clip_threshold: float = 1.0) -> list[jnp.array]:
     return [(_clip(w, clip_threshold), _clip(b, clip_threshold)) for (w, b) in grads]
    
def _clip(g: jnp.array, clip_threshold: float, eps: float = 1e-7) -> jnp.array:
    g_norm = jnp.max(jnp.array([jnp.linalg.norm(g), eps]))
    return jnp.min(jnp.array([clip_threshold/g_norm, 1.0])) * g

def gradient_step(
    params: list[jnp.array],
    grads: list[jnp.array],
    lr: float,
    weight_decay: float = 0,
) -> list[jnp.array]:
    
    return [
        ( (1 - lr * weight_decay) * w - lr * dw, (1 - lr * weight_decay) * b - lr * db) for (w, b), (dw, db) in zip(params, grads)
    ] 

def adamw_step(
    params: list[jnp.array],
    momentum: list[jnp.array],
    velocity: list[jnp.array],
    lr: float,
    weight_decay: float = 1e-4,
    eps: float = 1e-8,
) -> list[jnp.array]:
    
    adjusted_grads = [(w_g / (jnp.sqrt(w_v) + eps), b_g / (jnp.sqrt(b_v) + eps)) for (w_g, b_g), (w_v, b_v) in zip(momentum, velocity)]
    return [
        ( (1 - weight_decay) * w - lr * dw, (1 - weight_decay) * b - lr * db) for (w, b), (dw, db) in zip(params, adjusted_grads)
    ] 

def cosine_lr_scheduler(min_lr: float, max_lr: float, current_epoch: int, epochs_per_cycle: int, decay_rate: float = 1.0) -> float:
     adjusted_max_lr = decay_rate**current_epoch * max_lr
     return min_lr + 0.5 * (adjusted_max_lr - min_lr) * (1 + jnp.cos( (current_epoch % epochs_per_cycle) / epochs_per_cycle * jnp.pi))

def decay_lr_scheduler(lr: float, decay_rate: float = 0.95) -> float:
     return lr * decay_rate