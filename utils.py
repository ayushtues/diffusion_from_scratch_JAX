
import torch
# import torch.nn.functional as F
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt


def get_cosine_schedule(num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """

    return betas_for_alpha_bar(
        num_diffusion_timesteps,
        lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
    )


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return jnp.array(betas)

def get_values():
  beta_1 = 1e-4
  beta_T = 0.02

  beta_ts = jnp.linspace(beta_1, beta_T, 1000)
#   beta_ts = get_cosine_schedule(1000)
  alpha_ts = 1 - beta_ts
  alpha_hat_ts  = jnp.cumprod(alpha_ts, 0)
  alpha_hat_ts_prev = jnp.pad(alpha_hat_ts[:-1], (1, 0), 'constant', constant_values=1.0)

  sqrt_alpha_ts = jnp.sqrt(alpha_ts)
  sqrt_alpha_hat_ts = jnp.sqrt(alpha_hat_ts)
  sqrt_alpha_hat_ts_2 = jnp.sqrt(1-alpha_hat_ts)
  post_std = jnp.sqrt(((1-alpha_hat_ts_prev)/(1-alpha_hat_ts))*beta_ts)

  return sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std

def print_stats(x, name):
  print(f"{name} max: {jnp.max(x)}, min: {jnp.min(x)}, mean: {jnp.mean(x)}, std: {jnp.std(x)}")