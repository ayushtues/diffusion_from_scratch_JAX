import jax.numpy as jnp 
from dataloader import get_dataloader
from datetime import datetime
from utils import get_values, print_stats
import os
from diffusion import Diffusion
from models import UNet, get_position_embeddings
from jax import random
import jax
import numpy as np
import orbax.checkpoint
from typing import Any
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK']='True'

sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std = get_values()
model = Diffusion(sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std, 1, 1)
sqrt_alpha_ts = jnp.sqrt(alpha_ts)
sigma_ts = jnp.sqrt(beta_ts)
alpha_ts_2 = 1 - alpha_ts

rng = random.PRNGKey(0)
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
raw_restored = orbax_checkpointer.restore('ckpt/3000/default')
state = raw_restored['state']


@jax.jit
def sample_diffusion(params, x, t_embed, y, z, alpha_ts_2, sqrt_alpha_hat_ts_2, sqrt_alpha_ts):
        eps_pred=  model.apply({'params' : params}, x, t_embed, y=y, train=True, method='forward')
        eps_pred = (
            jnp.expand_dims(
                jnp.expand_dims(jnp.expand_dims(alpha_ts_2, -1), -1), -1
            )
            / jnp.expand_dims(
                jnp.expand_dims(
                    jnp.expand_dims(sqrt_alpha_hat_ts_2, -1), -1
                ),
                -1,
            )
        ) * eps_pred
        x = x - eps_pred
        x = x * (
            1
            / jnp.expand_dims(
                jnp.expand_dims(jnp.expand_dims(sqrt_alpha_ts, -1), -1), -1
            )
        )
        x = x + z
        return x

y = None
rng, key = random.split(rng, 2)
x = random.normal(key, [1, 32, 32, 1])
x_returned = []
params = state['params']

for i in tqdm(reversed(range(1000))):
    rng, key = random.split(rng, 2)
    t_embed = jnp.expand_dims(get_position_embeddings(i), 0)
    if i != 0:
        z = random.normal(key, x.shape)
        z = (
            jnp.expand_dims(
                jnp.expand_dims(jnp.expand_dims(sigma_ts[i], -1), -1), -1
            )
            * z
        )

    else:
        z = jnp.zeros(x.shape)
    x = sample_diffusion(params, x, t_embed, y, z, alpha_ts_2[i], sqrt_alpha_hat_ts_2[i], sqrt_alpha_ts[i])

    if i % 50 == 0:
        x_img = (x + 1.0) / 2
        x_returned.append(jnp.squeeze(x_img, 0))

sample = np.array(x_returned)
np.save('sample.npy', sample)
print(len(sample))