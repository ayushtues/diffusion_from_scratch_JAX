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

@jax.jit
def squared_error(x1, x2):
    return jnp.inner(x1-x2, x1-x2) / 2.0
@jax.jit
def train_step(state, batch, rng):
    x, y, t = batch
    eps = random.normal(rng, x.shape)
    t_embed = get_position_embeddings(jnp.squeeze(t, -1))

    def loss_fn(params, batch_stats):
        eps_pred, updates = model.apply({'params': params, 'batch_stats': batch_stats}, x, t, t_embed, eps, None, train=True, mutable=['batch_stats'])
        return jnp.mean(jax.vmap(squared_error)(eps, eps_pred)), updates
    (loss, updates), grads  = jax.value_and_grad(loss_fn, has_aux=True)(state.params, state.batch_stats)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    return loss, state


dataloader = get_dataloader()
sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std = get_values()
model = Diffusion(sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std, 1, 1)

epoch_number = 0
batches = 10000
EPOCHS = int(batches / len(dataloader) + 1)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_path = "runs/fashion_trainer_{}".format(timestamp)
writer = SummaryWriter(run_path)

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
save_args = orbax_utils.save_args_from_target(ckpt)

options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
checkpoint_manager = orbax.checkpoint.CheckpointManager(
    'ckpt', orbax_checkpointer, options)

def train_one_epoch(rng, state, epoch_index, batches, tb_writer, run_path, save_freq=2000):
    running_loss = 0.0

    for i, data in enumerate(dataloader):
        batch = epoch_index * len(dataloader) + i + 1
        if batch == batches:
            return running_loss / (i + 1)
        x, y, t = data
        x = x.transpose(1,2).transpose(2, 3)
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        t = t.cpu().numpy().astype(jnp.int32)
        data = (x, y, t)
        rng, key = random.split(rng, 2)
        loss, state = train_step(state, data, key)
        loss = np.array(loss)
        running_loss += loss
        if i % 10 == 0:
            print("  batch {} loss: {}".format(batch, loss))
            tb_writer.add_scalar("Loss/train", loss, batch)
        
        # if i % 500 == 0 :
        #     x = model.sample(device)
        #     show_grid_images(x, batch, run_path)

        # # Track best performance, and save the model's state
        if i % save_freq == 0:
            ckpt = {'state': state}
            checkpoint_manager.save(i, ckpt, save_kwargs={'save_args': save_args})

    return running_loss / len(dataloader), state, rng


for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch_number + 1))

    avg_loss, state, rng = train_one_epoch(rng, state, epoch_number, batches, writer, run_path)
    print(f"EPOCH : {epoch+1} loss : {avg_loss}")
    epoch_number += 1

# raw_restored = orbax_checkpointer.restore('ckpt/0/default')
# print(raw_restored)











