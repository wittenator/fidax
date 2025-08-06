from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.training.metrics import MetricState

from fidax.inception import get_fid_network


class FrechetInceptionDistance(nnx.Metric):
    """Fréchet Inception Distance (FID) metric implemented in JAX."""

    def __init__(
        self,
        max_samples: int = 50000,
        metric_dtype=jnp.float64,
        model_dtype=jnp.float32,
        real_stats: dict | None = None,
    ) -> None:
        """Initialize FID metric.

        Args:
            max_samples: Maximum number of samples to store
            metric_dtype: Data type for computations for model execution and metric computation (default: jnp.float64)
            model_dtype: Data type for the model (default: jnp.float64)
            real_stats: Pre-computed statistics for real images (mu and sigma)
        """
        self.max_samples = max_samples
        self.real_stats = real_stats
        self.metric_dtype = metric_dtype
        self.model_dtype = model_dtype

        # Initialize feature extractor
        self.model = get_fid_network(dtype=model_dtype)

        # Initialize storage for activations using MetricState
        self.real_acts = MetricState(jnp.zeros((max_samples, 2048), dtype=metric_dtype))
        self.fake_acts = MetricState(jnp.zeros((max_samples, 2048), dtype=metric_dtype))
        self.real_count = MetricState(jnp.array(0))
        self.fake_count = MetricState(jnp.array(0))

    @nnx.jit(static_argnames=("real",))
    def update(self, imgs, real: bool, **kwargs: Any) -> None:
        """Update the metric with new images.

        Args:
            imgs: Input images, shape [N, H, W, C], range [-1, 1]
            real: Whether the images are real (True) or generated (False)
        """
        # Handle grayscale images by repeating channels to RGB
        if imgs.shape[-1] == 1:
            imgs = jnp.repeat(imgs, 3, axis=-1)

        imgs = jax.image.resize(
            imgs, (imgs.shape[0], 299, 299, 3), method="bilinear"
        )  # Resize to InceptionV3 input size

        # Extract features using InceptionV3
        acts = self.model(imgs, train=False)[..., 0, 0, :].astype(self.metric_dtype)  # [N, 2048]
        n = acts.shape[0]

        if real:
            end = jnp.minimum(self.real_count.value + n, self.max_samples)
            self.real_acts.value = jax.lax.dynamic_update_slice(self.real_acts.value, acts, (self.real_count.value, 0))
            self.real_count.value = end
        else:
            end = jnp.minimum(self.fake_count.value + n, self.max_samples)
            self.fake_acts.value = jax.lax.dynamic_update_slice(self.fake_acts.value, acts, (self.fake_count.value, 0))
            self.fake_count.value = end

    def compute(self) -> float:
        """Compute the FID score between real and fake image distributions."""
        real_acts = self.real_acts.value[: self.real_count.value]
        fake_acts = self.fake_acts.value[: self.fake_count.value]

        mu1 = jnp.mean(fake_acts, axis=0)
        sigma1 = jnp.cov(fake_acts, rowvar=False)

        if self.real_stats is not None:
            mu2 = self.real_stats["mu"]
            sigma2 = self.real_stats["sigma"]
        else:
            mu2 = jnp.mean(real_acts, axis=0)
            sigma2 = jnp.cov(real_acts, rowvar=False)

        return self._fid_from_stats(mu1, sigma1, mu2, sigma2)

    @staticmethod
    @jax.jit
    def _fid_from_stats(mu1, sigma1, mu2, sigma2) -> jnp.ndarray:
        """Compute FID score from distribution statistics.
        Adapted from https://github.com/Lightning-AI/torchmetrics/blob/27e1dbe39ac50d6c84f72a16afbb7bf1eb19221e/src/torchmetrics/image/fid.py"""
        sigma1 = sigma1 + jnp.eye(sigma1.shape[0]) * 1e-6  # Add small value for numerical stability
        sigma2 = sigma2 + jnp.eye(sigma2.shape[0]) * 1e-6  # Add small value for numerical stability
        diff = jnp.sum((mu1 - mu2) ** 2, axis=-1)
        traces = jnp.trace(sigma1) + jnp.trace(sigma2)

        eigvals = jnp.linalg.eigvals(sigma1 @ sigma2)
        covmean = jnp.sum(jnp.sqrt(jnp.abs(eigvals)).real)
        return diff + traces - 2 * covmean

    def reset(self) -> None:
        """Reset the metric state."""
        self.real_acts.value = jnp.zeros_like(self.real_acts.value)
        self.fake_acts.value = jnp.zeros_like(self.fake_acts.value)
        self.real_count.value = jnp.array(0)
        self.fake_count.value = jnp.array(0)
