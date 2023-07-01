import jax.numpy as jnp 
from dataloader import get_dataloader
from datetime import datetime
from utils import get_values, print_stats
import os
from diffusion import Diffusion
from models import UNet, get_position_embeddings
from jax import random
from flax.training import train_state, checkpoints, orbax_utils
import optax
import jax
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import orbax.checkpoint
from typing import Any
# from jax import config
# config.update("jax_disable_jit", True)

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class TrainState(train_state.TrainState):
  batch_stats: Any


dataloader = get_dataloader()
sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std = get_values()
model = Diffusion(sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std, 1, 1)

rng = random.PRNGKey(0)
rng, eps_key, init_key = random.split(rng, 3)
x, y, t = next(iter(dataloader))
x = x.transpose(1,2).transpose(2, 3)
x = x.cpu().numpy()
y = y.cpu().numpy()
t = t.cpu().numpy().astype(jnp.int32)
eps = random.normal(eps_key, x.shape)
t_embed = get_position_embeddings(jnp.squeeze(t, -1))
variables = model.init(init_key, x, t, t_embed, eps, None)
params = variables['params']
batch_stats = variables['batch_stats']
state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    batch_stats=batch_stats,
    tx=optax.adam(1e-3),
)

ckpt = {'state' : state}
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
raw_restored = orbax_checkpointer.restore('ckpt/colab/default')
state = raw_restored['state']
# print(state)
# for key in state.keys():
    # print(key)

# exit()


sample, updates = model.apply({'params': state['params'], 'batch_stats': state['batch_stats']}, rng, mutable=['batch_stats'], method='sample')
sample = np.array(sample)
np.save('sample.npy', sample)
print(len(sample))