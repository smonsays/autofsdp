# Copyright (c) Simon Schug
# All rights reserved.

# MIT License

# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the “Software”), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:

# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

"""Example of fully-sharded data parallel training of a transformer on mock data."""

import dataclasses
import logging

import flax.linen as nn
import flax.struct as flax_struct
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import optax
from absl import app
from flax.training.train_state import TrainState
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import autofsdp


@dataclasses.dataclass
class Config:
  batch_size: int = 64  # needs to divide the number of devices
  seq_len: int = 16
  learning_rate: float = 0.001
  n_batches: int = 10
  d_data: int = 7
  n_classes: int = 11
  d_ffw: int = 128
  n_layers: int = 2
  d_model: int = 32
  n_heads: int = 4
  seed: int = 0


@flax_struct.dataclass
class Batch:
  x: jt.Float[jt.Array, 'B T D']
  y: jt.Float[jt.Array, 'B T']


class Feedforward(nn.Module):
  d_model: int
  d_ffw: int

  @nn.compact
  def __call__(self, x: jt.Float[jt.Array, 'B L D']) -> jt.Float[jt.Array, 'B L D']:
    x = nn.Dense(self.d_ffw, use_bias=False)(x)
    x = nn.gelu(x)
    x = nn.Dense(self.d_model, use_bias=False)(x)
    return x


class TransformerBlock(nn.Module):
  d_ffw: int
  n_heads: int
  d_model: int

  @nn.compact
  def __call__(self, x: jt.Float[jt.Array, 'B L D']) -> jt.Float[jt.Array, 'B L D']:
    _, seq_len, _ = x.shape

    x = nn.LayerNorm(use_bias=False)(x)
    causal_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.bool))
    x += nn.SelfAttention(self.n_heads, use_bias=False)(x, mask=causal_mask)

    x = nn.LayerNorm(use_bias=False)(x)
    x += Feedforward(self.d_model, self.d_ffw)(x)

    return x


class Transformer(nn.Module):
  d_ffw: int
  n_heads: int
  n_layers: int
  d_model: int
  d_out: int

  @nn.compact
  def __call__(self, inputs: jt.Float[jt.Array, 'B L D']) -> jt.Float[jt.Array, 'B L O']:
    x = nn.Dense(self.d_model, use_bias=False)(inputs)

    for i in range(self.n_layers):
      x = TransformerBlock(self.d_ffw, self.n_heads, self.d_model, name=f'block_{i}')(x)

    x = nn.LayerNorm(use_bias=False)(x)
    return nn.Dense(self.d_out, use_bias=False)(x)


def train(c: Config) -> None:
  """Training loop with automatic fully-sharded data parallelism (FSDP)"""
  logging.info(f'{jax.device_count()} available devices.')
  rng = jax.random.key(c.seed)
  transformer = Transformer(c.d_ffw, c.n_heads, c.n_layers, c.d_model, c.n_classes)

  # In the following, we use autofsdp to automatically infer the shardings of
  # the model parameters and the optimizer state.
  # In a first pass, we only infer the shapes and shardings. These can then be passed to
  # jit to actually instantiate the arrays already on the respective devices.
  def init_train_state(rng: jt.PRNGKeyArray, inputs: jax.Array) -> TrainState:
    params = transformer.init(rng, inputs)
    return TrainState.create(
      apply_fn=transformer.apply,
      params=params['params'],
      tx=optax.adam(c.learning_rate),
    )

  inputs = jax.ShapeDtypeStruct(shape=(1, 1, c.d_data), dtype=jnp.float32)
  state = jax.eval_shape(init_train_state, rng, inputs)
  mesh = jax.make_mesh((jax.device_count(),), ('data',))
  shardings = autofsdp.infer_fsdp_sharding(state, mesh)
  state = jax.jit(init_train_state, out_shardings=shardings)(rng, inputs)

  # We can visualize whether the sharding had the intended affect on the parameters
  jax.debug.visualize_array_sharding(
    state.params['block_0']['SelfAttention_0']['query']['kernel'][:, 0, :]
  )

  # In the training loop, we only need to place a sharding constraint on the data,
  # jit then automatically handles the rest for us.
  @jax.jit
  def update_train_state(state: TrainState, batch: Batch) -> TrainState:
    def loss_fn(params: jt.PyTree):
      logits = state.apply_fn({'params': state.params}, batch.x)
      loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch.y)
      return jnp.mean(loss)

    def shard_data(data: jax.Array):
      return jax.lax.with_sharding_constraint(data, NamedSharding(mesh, P('data')))

    batch = jax.tree.map(shard_data, batch)
    grads = jax.grad(loss_fn)(state.params)

    return state.apply_gradients(grads=grads)

  for i in range(c.n_batches):
    logging.info(f'Iteration {i}')
    batch = Batch(  # mock data
      x=jnp.ones((c.batch_size, c.seq_len, c.d_data)),
      y=i * jnp.ones((c.batch_size, c.seq_len), dtype=jnp.int32),
    )
    state = update_train_state(state, batch)


def main(argv: list[str]) -> None:
  del argv
  train(Config())


if __name__ == '__main__':
  app.run(main)
