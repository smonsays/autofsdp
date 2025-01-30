"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the “Software”), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from typing import TypeVar

import jax
import jax.sharding as shd
import jaxtyping as jt

T = TypeVar('T')


def _fsdp_partition_spec(shape: tuple, n_devices: int) -> shd.PartitionSpec:
  """Infer fsdp-like partition spec."""

  if len(shape) < 2:
    return shd.PartitionSpec()  # Replicate biases and scalars

  # Shard over largest dim that is divisible by n_devices
  largest_divisible = max((x for x in shape if x % n_devices == 0), default=None)

  partitions = [None] * len(shape)
  if largest_divisible is not None:
    index = shape.index(largest_divisible)
    partitions[index] = 'data'

  return shd.PartitionSpec(*partitions)


def infer_fsdp_sharding(pytree: jt.PyTree, mesh: shd.Mesh) -> jt.PyTree:
  """Infer fsdp-like sharding tree."""

  def f(x: object) -> shd.Sharding | None:
    if hasattr(x, 'shape'):
      pspec = _fsdp_partition_spec(x.shape, len(mesh.devices))
      return shd.NamedSharding(mesh, pspec)
    else:
      return None

  return jax.tree.map(f, pytree)


def shard_pytree(
  pytree: jt.PyTree, shardings: jt.PyTree[shd.Sharding | None]
) -> jt.PyTree:
  """Shard a pytree according to given sharding tree."""

  return jax.tree.map(lambda x,s: jax.device_put(x, s), pytree, shardings)
