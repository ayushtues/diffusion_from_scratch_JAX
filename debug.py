import torch
from torchvision import transforms 
import matplotlib.pyplot as plt
import numpy as np 
from dataloader import get_dataloader
from diffusion_th import Diffusion as Diffusion_th
from diffusion import Diffusion
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils_th import get_values as get_values_th, print_stats
from models_th import get_position_embeddings as get_position_embeddings_th
from utils import get_values
from models import get_position_embeddings
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from jax import random
import jax.numpy as jnp
import jax

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

dataloader = get_dataloader()
sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std = get_values_th(device)
model = Diffusion_th(sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std, 1, 1)
model = model.to(device)
x, y, t = next(iter(dataloader))
y_one = torch.nn.functional.one_hot(y, 10).float()
x = x.to(device)
y_one = y_one.to(device)
# x = x.view(x.shape[0], -1, 1, 1)
x = x * 2 - 1
t = t.to(device)
print(t.shape)
t = t.squeeze(-1)
t_embed = get_position_embeddings_th(t, device)
# t_embed = t
eps = torch.randn_like(x)


# Make predictions for this batch
eps_pred_th = model(x, t, t_embed, eps, y_one)
# print(eps_pred_th.shape)
samples_th = model.sample(device)


sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std = get_values()
sqrt_alpha_ts = jnp.sqrt(alpha_ts)
sigma_ts = jnp.sqrt(beta_ts)
alpha_ts_2 = 1 - alpha_ts

model = Diffusion(
    sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std, 1, 1
)

x = x.transpose(1, 2).transpose(2, 3)
eps = eps.transpose(1, 2).transpose(2, 3)
t = t.unsqueeze(-1)
# print(t.shape)
x = x.cpu().numpy()
y = y.cpu().numpy()
t = t.cpu().numpy().astype(jnp.int32)
t_embed = get_position_embeddings(jnp.squeeze(t, -1))
# t_embed = t_embed.cpu().numpy()
eps = eps.cpu().numpy()

rng = random.PRNGKey(0)
rng, eps_key, init_key = random.split(rng, 3)
variables = model.init(init_key, x, t, t_embed, eps, None)
params = {}
# eps_pred_jax = model.apply(
#     {"params": params},
#     x,
#     t,
#     t_embed,
#     eps,
#     None,
#     train=True,
# )

# print(torch.mean(eps_pred_th))
# print(jnp.mean(eps_pred_jax))

x_returned = []
x = jnp.ones([1, 32, 32, 1])


@jax.jit
def sample_diffusion(params, x, t_embed, y, z, alpha_ts_2, sqrt_alpha_hat_ts_2, sqrt_alpha_ts):
        eps_pred =  model.apply({'params' : params}, x, t_embed, y=y, train=True, method='forward')
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

for i in (reversed(range(2))):
    t_embed = jnp.expand_dims(get_position_embeddings(i), 0)
    if i != 0:
        z = jnp.ones(x.shape)
        z = (
            jnp.expand_dims(
                jnp.expand_dims(jnp.expand_dims(sigma_ts[i], -1), -1), -1
            )
            * z
        )

    else:
        z = jnp.ones(x.shape)
    x = sample_diffusion(params, x, t_embed, y, z, alpha_ts_2[i], sqrt_alpha_hat_ts_2[i], sqrt_alpha_ts[i])
    x_img = (x + 1.0) / 2
    x_returned.append(jnp.squeeze(x_img, 0))

samples_th = torch.stack(samples_th)
x_returned = jnp.array(x_returned)

print(jnp.mean(x_returned))
print(torch.mean(samples_th))
