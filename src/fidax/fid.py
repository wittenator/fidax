from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.training.metrics import MetricState

from fidax.inception import get_fid_network

if TYPE_CHECKING:
    from jaxtyping import Array, ArrayLike, Float, Int


class Stats(TypedDict):
    """Typed container for real/fake distribution statistics used by FID."""

    mu: Float[Array, "feat"]
    sigma: Float[Array, "feat feat"]


def _extract_activations(model, imgs: Float[Array, "batch h w c"]) -> Float[Array, "batch 2048"]:
    """Run Inception and return [N, 2048] activations (model's dtype)."""
    # Handle grayscale images by repeating channels to RGB
    if imgs.shape[-1] == 1:
        imgs = jnp.tile(imgs, (1, 1, 1, 3))

    imgs = jax.image.resize(imgs, (imgs.shape[0], 299, 299, 3), method="bilinear")
    acts = model(imgs, train=False)[..., 0, 0, :]
    return acts


class FrechetInceptionDistance(nnx.Metric):
    """Fréchet Inception Distance (FID) metric implemented in JAX."""

    def __init__(
        self,
        metric_dtype: jnp.dtype = jnp.float64,
        model_dtype: str = "float32",
        real_stats: Stats | None = None,
        weights_cache_dir: str | None = "data",
    ) -> None:
        """Initialize FID metric.

        Args:
            metric_dtype: Data type for computations for model execution and metric computation (default: jnp.float64)
            model_dtype: Data type for the model (default: jnp.float32)
            real_stats: Pre-computed statistics for real images (mu and sigma)
            weights_cache_dir: Directory to cache/download Inception weights
        """
        self.real_stats = real_stats
        self.metric_dtype = metric_dtype
        self.model_dtype = model_dtype
        self.weights_cache_dir = weights_cache_dir

        # Initialize feature extractor
        self.model = get_fid_network(dtype=model_dtype, ckpt_dir=self.weights_cache_dir)

        # Dimension of Inception feature activations
        self._feat_dim = 2048

        # Streaming accumulators on DEVICE (constant memory wrt number of samples)
        # Maintain (count, mean[D], M2[D,D]) per split as JAX arrays
        self._real_n = MetricState(jnp.array(0, dtype=jnp.int32))
        self._real_mean = MetricState(jnp.zeros((self._feat_dim,), dtype=self.metric_dtype))
        self._real_M2 = MetricState(jnp.zeros((self._feat_dim, self._feat_dim), dtype=self.metric_dtype))

        self._fake_n = MetricState(jnp.array(0, dtype=jnp.int32))
        self._fake_mean = MetricState(jnp.zeros((self._feat_dim,), dtype=self.metric_dtype))
        self._fake_M2 = MetricState(jnp.zeros((self._feat_dim, self._feat_dim), dtype=self.metric_dtype))

    @nnx.jit(static_argnames=("real",))
    def update(self, imgs: Float[ArrayLike, "batch h w c"], real: bool, **kwargs: Any) -> None:  # noqa: ARG002, D417, RUF100
        """Update the metric with new images.

        Args:
            imgs: Input images, shape [N, H, W, C], range [0, 1]
            real: Whether the images are real (True) or generated (False)
            kwargs: Additional keyword arguments (unused)

        Note:
            This method computes activations on the accelerator and aggregates
            statistics on the host, avoiding large device-resident buffers.
        """
        imgs = imgs.astype(self.model_dtype) * 2.0 - 1.0  # Scale to [-1, 1]
        # Run model forward on device and update accumulators on device as well
        acts = _extract_activations(self.model, imgs).astype(self.metric_dtype)  # [B, D]
        nb = acts.shape[0]
        mb = jnp.mean(acts, axis=0)
        # Compute batch second moment and convert to centered sum of squares (M2b)
        S2b = acts.T @ acts  # (D,D)
        nb_f = jnp.array(nb, dtype=self.metric_dtype)
        M2b = S2b - nb_f * jnp.outer(mb, mb)  # (D,D)

        # Select current split accumulators
        n = self._real_n.value if real else self._fake_n.value
        mean = self._real_mean.value if real else self._fake_mean.value
        M2 = self._real_M2.value if real else self._fake_M2.value

        # Combine using Chan's formula (works also when n == 0)
        n_f = n.astype(self.metric_dtype)
        nb_f = jnp.array(nb, dtype=self.metric_dtype)
        n_new = n + jnp.array(nb, dtype=n.dtype)
        n_new_f = n_f + nb_f
        mean_new = jnp.where(n_new_f > 0, (mean * n_f + mb * nb_f) / n_new_f, jnp.zeros_like(mean))
        delta = mb - mean
        M2_new = (
            M2
            + M2b
            + jnp.outer(delta, delta)
            * jnp.where(n_new_f > 0, (n_f * nb_f) / n_new_f, jnp.array(0, dtype=self.metric_dtype))
        )

        if real:
            self._real_n.value = n_new
            self._real_mean.value = mean_new
            self._real_M2.value = M2_new
        else:
            self._fake_n.value = n_new
            self._fake_mean.value = mean_new
            self._fake_M2.value = M2_new

    @nnx.jit
    def compute(self) -> float:
        """Compute the FID score between real and fake image distributions."""
        # Derive stats for fake (device)
        mu1, sigma1 = self._stats_from_accumulators_jax(self._fake_n.value, self._fake_mean.value, self._fake_M2.value)

        # Derive stats for real, or use precomputed
        if self.real_stats is not None:
            mu2 = self.real_stats["mu"]
            sigma2 = self.real_stats["sigma"]
        else:
            mu2, sigma2 = self._stats_from_accumulators_jax(
                self._real_n.value, self._real_mean.value, self._real_M2.value
            )

        return self._fid_from_stats(mu1, sigma1, mu2, sigma2)

    @staticmethod
    @jax.jit
    def _fid_from_stats(
        mu1: Float[Array, "feat"],
        sigma1: Float[Array, "feat feat"],
        mu2: Float[Array, "feat"],
        sigma2: Float[Array, "feat feat"],
    ) -> Float[Array, ""]:
        """Compute FID score from distribution statistics.

        Adapted from https://github.com/Lightning-AI/torchmetrics/blob/27e1dbe39ac50d6c84f72a16afbb7bf1eb19221e/src/torchmetrics/image/fid.py
        """
        sigma1 = sigma1 + jnp.eye(sigma1.shape[0]) * 1e-6  # Add small value for numerical stability
        sigma2 = sigma2 + jnp.eye(sigma2.shape[0]) * 1e-6  # Add small value for numerical stability
        diff = jnp.sum((mu1 - mu2) ** 2, axis=-1)
        traces = jnp.trace(sigma1) + jnp.trace(sigma2)

        eigvals = jnp.linalg.eigvals(sigma1 @ sigma2)
        covmean = jnp.sum(jnp.sqrt(jnp.abs(eigvals)).real)
        return diff + traces - 2 * covmean

    def reset(self) -> None:
        """Reset the metric state."""
        self._real_n.value = jnp.array(0, dtype=jnp.int32)
        self._real_mean.value = jnp.zeros((self._feat_dim,), dtype=self.metric_dtype)
        self._real_M2.value = jnp.zeros((self._feat_dim, self._feat_dim), dtype=self.metric_dtype)
        self._fake_n.value = jnp.array(0, dtype=jnp.int32)
        self._fake_mean.value = jnp.zeros((self._feat_dim,), dtype=self.metric_dtype)
        self._fake_M2.value = jnp.zeros((self._feat_dim, self._feat_dim), dtype=self.metric_dtype)

    # --------------------
    # Helper methods (host-side)
    # --------------------
    def _stats_from_accumulators_jax(
        self,
        n: Int[Array, ""],
        mean: Float[Array, "feat"],
        M2: Float[Array, "feat feat"],
    ) -> tuple[Float[Array, "feat"], Float[Array, "feat feat"]]:
        """Return (mu, sigma) from accumulators on device; ddof=1 if n>1, else zeros."""
        n = jnp.asarray(n)
        n_f = n.astype(self.metric_dtype)
        mu = mean
        sigma = jnp.where(n >= 2, M2 / jnp.maximum(n_f - 1.0, 1.0), jnp.zeros_like(M2))
        return mu, sigma

    # Public helpers for tests and external use
    def get_real_stats(self) -> tuple[Float[Array, "feat"], Float[Array, "feat feat"]]:
        return self._stats_from_accumulators_jax(self._real_n.value, self._real_mean.value, self._real_M2.value)

    def get_fake_stats(self) -> tuple[Float[Array, "feat"], Float[Array, "feat feat"]]:
        return self._stats_from_accumulators_jax(self._fake_n.value, self._fake_mean.value, self._fake_M2.value)

    @property
    def real_count(self) -> int:
        return int(self._real_n.value)

    @property
    def fake_count(self) -> int:
        return int(self._fake_n.value)
