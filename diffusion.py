from models import UNet
from models import get_position_embeddings
from utils import print_stats
from copy import deepcopy
from flax import linen as nn
from typing import Optional
import jax.numpy as jnp

class Diffusion(nn.Module):
    sqrt_alpha_hat_ts : float
    sqrt_alpha_hat_ts_2 : float
    alpha_ts : float
    beta_ts : float
    post_std : float
    n_channels : int
    n_classes : int
    bilinear : Optional[bool] = False
    class_conditioned : Optional[bool] = False

    def setup(self):
        self.model = UNet(self.n_channels, self.n_classes, bilinear=False)
        self.ema_model = deepcopy(self.model)
        self.ema_decay = 0.999
        self.ema_start = 1000
        self.ema_update_rate = 1
        self.step = 0

        self.sqrt_alpha_ts = jnp.sqrt(self.alpha_ts)
        self.sigma_ts = jnp.sqrt(self.beta_ts)
        self.alpha_ts_2 = 1 - self.alpha_ts

    # def update_ema(self):
    #     self.step += 1
    #     if self.step % self.ema_update_rate == 0:
    #         if self.step < self.ema_start:
    #             self.ema_model.load_state_dict(self.model.state_dict())
    #         else:
    #             for current_params, ema_params in zip(self.model.parameters(), self.ema_model.parameters()):
    #                 old, new = ema_params.data, current_params.data
    #                 if old is not None:
    #                     ema_params.data = old * self.ema_decay + new * (1 - self.ema_decay)

    def __call__(self, x, t, t_embed, eps, y=None):
        c1 = jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(jnp.take_along_axis(self.sqrt_alpha_hat_ts, jnp.squeeze(t, -1), 0), -1), -1), -1)
        # TODO, move this to the dataset itself instead of using gather
        c2 = jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(jnp.take_along_axis(self.sqrt_alpha_hat_ts_2, jnp.squeeze(t, -1), 0), -1), -1), -1)

        input_x = x * c1 + eps * c2
        if self.class_conditioned:
            eps_pred = self.model(input_x, t_embed, y)
        else:
            eps_pred = self.model(input_x, t_embed)

        return eps_pred

    # @torch.no_grad()
    # def sample(self, device, y=None, classifier=None):
    #     if y is None:
    #         y = torch.zeros([1], device=device, dtype=torch.long)
    #         y = F.one_hot(y, 10).float()
    #     x = torch.randn([1, 1, 32, 32], device=device)
    #     x_returned = []
    #     for i in reversed(range(1000)):
    #         t_embed = get_position_embeddings(i, device).unsqueeze(0)
    #         if self.class_conditioned:
    #             eps_pred = self.ema_model(x, t_embed, y)
    #         else:
    #             eps_pred = self.ema_model(x, t_embed)
    #         eps_pred = (
    #             self.alpha_ts_2[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #             / self.sqrt_alpha_hat_ts_2[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #         ) * eps_pred
    #         x_old = x
    #         x = x - eps_pred
    #         x = x * (
    #             1 / self.sqrt_alpha_ts[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #         )
    #         if i != 0:
    #             z = torch.randn_like(x, device=device)
    #             z = self.sigma_ts[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * z

    #             if classifier is not None:
    #                 t_embed2 = get_position_embeddings(i-1, device).unsqueeze(0)
    #                 y_index = torch.argmax(y, dim=1)
    #                 gradient = cond_fn(x_old, t_embed2, classifier, y_index)
    #                 x = x + self.beta_ts[i]*gradient*1.0
    #         else:
    #             z = torch.zeros_like(x, device=device)
    #         x = x + z

    #         if i % 50 == 0:
    #             x_img = (x + 1.0) / 2
    #             x_returned.append(x_img.squeeze(0).detach())

    #     return x_returned