from __future__ import annotations

from typing import Any, NamedTuple

import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax


class DQN(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.astype(jnp.float32) / 255.0

        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        return nn.Dense(features=self.num_actions)(x)