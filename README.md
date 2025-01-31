# autofsdp

`autofsdp` is a utility to add Fully-Sharded Data Parallelism (FSDP) with very few lines of code using jax primitives (no magic!). 

## Installation

Install `autofsdp` using pip:

```bash
pip install git+https://github.com/smonsays/autofsdp.git
```

## Example usage

`autofsdp.infer_fsdp_sharding` automatically infers shardings that can either be passed to `jax.jit`, or directly applied using `jax.device_put`.

```python
import autofsdp

mesh = jax.make_mesh((jax.device_count(),), ('data',))
shardings = autofsdp.infer_fsdp_sharding(state_shapes, mesh)

# Usage as sharding constraint with jit
state = jax.jit(init_train_state, out_shardings=shardings)(...)

# Or, directly apply the sharding using device_put
state = jax.tree.map(lambda x, s: jax.device_put(x, s), state, shardings)
```

## Full example

For a small, self-contained example, training a Transformer with `autofsdp`, see [examples/training.py](https://github.com/smonsays/autofsdp/tree/main/examples/training.py).

## How it works

All of the heavy lifting is done by the excellent automatic compiler-based parallelization of `jax.jit`.
The only thing we need to do is to provide a sharding specification of each `jax.Array` which specifices for each axis whether it is replicated or sharded along an axis of our mesh of devices.

For FSDP the mesh is one dimensional, with a single axis we call `'data'`:
```
mesh = jax.make_mesh((jax.device_count(),), ('data',))
```

All that `autofsdp.infer_fsdp_sharding` does for each array is to identify the largest dimension that is divisible by the number of devices and return a respective sharding annotation.
For example for an array `x` with `x.shape = (16, 32, 101)` and `jax.device_count()=8`, it would return `(None, 'data', None)`.


