# FIDax

A JAX implementation of the Fréchet Inception Distance (FID) metric for evaluating generative models in form of a sklearn-compatible metric.

## Features

- **Pure JAX Implementation**: Leverages JAX's JIT compilation for fast computation
- **Pre-computed Statistics**: Supports using pre-computed real image statistics for faster evaluation
- **GPU Accelerated**: Optimized for CUDA-enabled GPUs
- **Torchmetrics Compatible**: Results match torchmetrics implementation up to 1e-1 absolute tolerance with FP32 execution of the InceptionV3 model and FP64 for the metric computation on CIFAR10 tests

## Installation

```bash
# Clone the repository
git clone git@github.com:wittenator/fidax.git
cd fidax

# Install dependencies using uv
uv sync --frozen
```

or install it directly as a dependency with e.g. uv:
```
uv add git+https://github.com/wittenator/fidax.git
```

## Quick Start

```python
import jax 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from fidax.fid import FrechetInceptionDistance

# Initialize FID metric
fid = FrechetInceptionDistance(max_samples=10000)

# Update with real images (shape: [N, 299, 299, 3], range: [-1, 1])
real_images = jnp.random.uniform(-1, 1, (100, 299, 299, 3))
fid.update(real_images, real=True)

# Update with generated/fake images
fake_images = jnp.random.uniform(-1, 1, (100, 299, 299, 3))
fid.update(fake_images, real=False)

# Compute FID score
fid_score = fid.compute()
print(f"FID Score: {fid_score}")
```

## Advanced Usage

### Batched Processing

```python
# Process large datasets in batches
fid = FrechetInceptionDistance(max_samples=50000)

batch_size = 64
for i in range(0, len(real_images), batch_size):
    batch = real_images[i:i+batch_size]
    fid.update(batch, real=True)
```

### Pre-computed Statistics

```python
# Use pre-computed real statistics for faster evaluation
real_stats = {
    "mu": mu_real,      # Mean of real activations
    "sigma": sigma_real # Covariance of real activations
}

fid = FrechetInceptionDistance(max_samples=10000, real_stats=real_stats)
# Only need to update with fake images
fid.update(fake_images, real=False)
```

## Requirements

- Python ≥ 3.12
- JAX with CUDA support
- Flax
- NumPy

See [`pyproject.toml`](pyproject.toml) for complete dependency list.

## Development

This project uses a development container with GPU support. To set up the development environment:

```bash
# The dev container will automatically install dependencies
# Run tests
uv run pytest src/fidax/test_fid_metric.py
```

## Testing

The implementation includes tests against torchmetrics:

```bash
uv run pytest src/fidax/test_fid_metric.py -v
```

Tests verify:
- Equivalence with torchmetrics implementation
- Pre-computed statistics functionality
- Real-world performance on CIFAR-10 dataset

## License

Apache 2.0 License - see [`LICENSE`](LICENSE) for details.

## Related Projects

- **[jax-fid-parallel](https://github.com/kvfrans/jax-fid-parallel)** - Parallel implementation of FID computation in JAX
- **[jax-fid](https://github.com/matthias-wright/jax-fid)** - Original JAX implementation of FID that inspired this project