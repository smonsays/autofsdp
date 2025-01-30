# autofsdp

`autofsdp` is a utility to add Fully-Sharded Data Parallelism (FSDP) with minimal code changes in jax. 

## Installation

Install `autofsdp` using pip:

```bash
pip install git+https://github.com/smonsays/autofsdp.git
```

## Example usage

`autofsdp.infer_fsdp_sharding` automatically infers shardings that can either be passed to `jax.jit`, or directly applied using `autofsdp.shard_pytree`.

```python
import autofsdp

mesh = jax.make_mesh((jax.device_count(),), ('data',))
shardings = autofsdp.infer_fsdp_sharding(state_shapes, mesh)

# Usage as sharding constraint with jit
state = jax.jit(init_train_state, out_shardings=shardings)(...)

# Or, directly apply the sharding using device_put
autofsdp.shard_pytree(state, shardings)
```

## Full example

For a small, self-contained example training a Transformer with `autofsdp`, see [examples/training.py](https://github.com/smonsays/autofsdp/tree/main/examples/training.py).


