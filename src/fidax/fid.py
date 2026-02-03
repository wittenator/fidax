from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.training.metrics import MetricState

from fidax.models import get_fid_network

if TYPE_CHECKING:
    from jaxtyping import Array, ArrayLike, Float, Int


class Stats(TypedDict):
    """Typed container for real/fake distribution statistics used by FID."""

    mu: Float[Array, "feat"]
    sigma: Float[Array, "feat feat"]


class _FIDBase:
    def __init__(
        self,
        metric_dtype: jnp.dtype = jnp.float64,
        model_dtype: str = "float32",
        real_stats: Stats | None = None,
        weights_cache_dir: str | None = "data",
        model_name: str = "inception_v3",
        model: nnx.Module | None = None,
        image_processor: nnx.Module | None = None,
        feature_dim: int = 2048,
    ) -> None:
        self.real_stats = nnx.data(real_stats)
        self.metric_dtype = metric_dtype
        self.model_dtype = model_dtype
        self.weights_cache_dir = weights_cache_dir

        # Initialize feature extractor
        if model is not None and image_processor is not None:
            self.model = model
            self.image_preprocessor = image_processor
        elif model_name is not None:
            self.image_preprocessor, self.model = get_fid_network(
                model_name=model_name, dtype=model_dtype, ckpt_dir=self.weights_cache_dir
            )
        else:
            raise ValueError("Either model and image_processor or model_name must be provided.")

        # Dimension of Inception feature activations
        self._feat_dim = feature_dim

    def _extract_activations(self, imgs: Float[ArrayLike, "batch h w c"]) -> jnp.ndarray:
        imgs = imgs.astype(self.model_dtype)
        return self.model(**self.image_preprocessor(imgs)).astype(self.metric_dtype)

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

    def _stats_from_accumulators(
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

    def _stack_stats(self, acts_list: list[jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        if not acts_list:
            return (
                jnp.zeros((self._feat_dim,), dtype=self.metric_dtype),
                jnp.zeros((self._feat_dim, self._feat_dim), dtype=self.metric_dtype),
            )
        acts = jnp.concatenate(acts_list, axis=0)
        mu = jnp.mean(acts, axis=0)
        sigma = jnp.cov(acts, rowvar=False, bias=False).astype(self.metric_dtype)
        return mu, sigma


class FrechetInceptionDistance(nnx.Metric, _FIDBase):
    """Fréchet Inception Distance (FID) metric implemented in JAX."""

    def __init__(
        self,
        metric_dtype: jnp.dtype = jnp.float64,
        model_dtype: str = "float32",
        real_stats: Stats | None = None,
        weights_cache_dir: str | None = "data",
        model_name: str = "inception_v3",
        model: nnx.Module | None = None,
        image_processor: nnx.Module | None = None,
        feature_dim: int = 2048,
    ) -> None:
        """Initialize FID metric.

        Args:
            metric_dtype: Data type for computations for model execution and metric computation (default: jnp.float64)
            model_dtype: Data type for the model (default: jnp.float32)
            real_stats: Pre-computed statistics for real images (mu and sigma)
            weights_cache_dir: Directory to cache/download Inception weights
            model_name: Name of the model to use for FID computation. Use "inception_v3" to use an InceptionV3 backbone or a Flax-compatible DinoV2 model from HF transformers (default: "inception_v3")
            model: Pre-initialized feature extractor model (overrides model_name if provided)
            image_processor: Pre-initialized image processor (not used if model is provided)
            feature_dim: Dimension of the feature activations (default: 2048)
        """
        _FIDBase.__init__(
            self,
            metric_dtype=metric_dtype,
            model_dtype=model_dtype,
            real_stats=real_stats,
            weights_cache_dir=weights_cache_dir,
            model_name=model_name,
            model=model,
            image_processor=image_processor,
            feature_dim=feature_dim,
        )

        # Streaming accumulators (constant memory wrt number of samples)
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
        # Run model forward on device and update accumulators on device as well
        acts = self._extract_activations(imgs)
        nb = acts.shape[0]
        mb = jnp.mean(acts, axis=0)
        # Compute batch second moment and convert to centered sum of squares (M2b)
        S2b = acts.T @ acts  # (D,D)
        nb_f = jnp.array(nb, dtype=self.metric_dtype)
        M2b = S2b - nb_f * jnp.outer(mb, mb)  # (D,D)

        # Select current split accumulators
        n = self._real_n[...] if real else self._fake_n[...]
        mean = self._real_mean[...] if real else self._fake_mean[...]
        M2 = self._real_M2[...] if real else self._fake_M2[...]

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
            self._real_n[...] = n_new
            self._real_mean[...] = mean_new
            self._real_M2[...] = M2_new
        else:
            self._fake_n[...] = n_new
            self._fake_mean[...] = mean_new
            self._fake_M2[...] = M2_new

    @nnx.jit
    def compute(self) -> float:
        """Compute the FID score between real and fake image distributions."""
        # Derive stats for fake (device)
        mu1, sigma1 = self._stats_from_accumulators(self._fake_n[...], self._fake_mean[...], self._fake_M2[...])

        # Derive stats for real, or use precomputed
        if self.real_stats is not None:
            mu2 = self.real_stats["mu"]
            sigma2 = self.real_stats["sigma"]
        else:
            mu2, sigma2 = self._stats_from_accumulators(self._real_n[...], self._real_mean[...], self._real_M2[...])

        return self._fid_from_stats(mu1, sigma1, mu2, sigma2)

    def reset(self) -> None:
        """Reset the metric state."""
        self._real_n[...] = jnp.array(0, dtype=jnp.int32)
        self._real_mean[...] = jnp.zeros((self._feat_dim,), dtype=self.metric_dtype)
        self._real_M2[...] = jnp.zeros((self._feat_dim, self._feat_dim), dtype=self.metric_dtype)
        self._fake_n[...] = jnp.array(0, dtype=jnp.int32)
        self._fake_mean[...] = jnp.zeros((self._feat_dim,), dtype=self.metric_dtype)
        self._fake_M2[...] = jnp.zeros((self._feat_dim, self._feat_dim), dtype=self.metric_dtype)

    # Public helpers for tests and external use
    def get_real_stats(self) -> tuple[Float[Array, "feat"], Float[Array, "feat feat"]]:
        return self._stats_from_accumulators(self._real_n[...], self._real_mean[...], self._real_M2[...])

    def get_fake_stats(self) -> tuple[Float[Array, "feat"], Float[Array, "feat feat"]]:
        return self._stats_from_accumulators(self._fake_n[...], self._fake_mean[...], self._fake_M2[...])

    @property
    def real_count(self) -> int:
        return int(self._real_n[...])

    @property
    def fake_count(self) -> int:
        return int(self._fake_n[...])


class StandardFrechetInceptionDistance(nnx.Metric, _FIDBase):
    """Reference FID that stores activations and computes stats at the end."""

    def __init__(
        self,
        metric_dtype: jnp.dtype = jnp.float64,
        model_dtype: str = "float32",
        real_stats: Stats | None = None,
        weights_cache_dir: str | None = "data",
        model_name: str = "inception_v3",
    ) -> None:
        _FIDBase.__init__(
            self,
            metric_dtype=metric_dtype,
            model_dtype=model_dtype,
            real_stats=real_stats,
            weights_cache_dir=weights_cache_dir,
            model_name=model_name,
        )
        self._real_acts: list[jnp.ndarray] = []
        self._fake_acts: list[jnp.ndarray] = []

    def _append_acts(self, imgs: Float[ArrayLike, "batch h w c"], real: bool) -> None:
        acts = self._extract_activations(imgs)
        if real:
            self._real_acts.append(acts)
        else:
            self._fake_acts.append(acts)

    def update(self, imgs: Float[ArrayLike, "batch h w c"], real: bool) -> None:
        self._append_acts(imgs, real)

    def reset(self) -> None:
        self._real_acts.clear()
        self._fake_acts.clear()

    def compute(self) -> float:
        # Fake stats always derived from stored activations
        mu1, sigma1 = self._stack_stats(self._fake_acts)

        if self.real_stats is not None:
            mu2 = jnp.asarray(self.real_stats["mu"], dtype=self.metric_dtype)
            sigma2 = jnp.asarray(self.real_stats["sigma"], dtype=self.metric_dtype)
        else:
            mu2, sigma2 = self._stack_stats(self._real_acts)

        return float(self._fid_from_stats(mu1, sigma1, mu2, sigma2))

    @property
    def real_count(self) -> int:
        return int(sum(a.shape[0] for a in self._real_acts))

    @property
    def fake_count(self) -> int:
        return int(sum(a.shape[0] for a in self._fake_acts))


class CachedRealFrechetInceptionDistance(nnx.Metric, _FIDBase):
    """FID that caches real activations but streams fake statistics."""

    def __init__(
        self,
        metric_dtype: jnp.dtype = jnp.float64,
        model_dtype: str = "float32",
        real_stats: Stats | None = None,
        weights_cache_dir: str | None = "data",
        model_name: str = "inception_v3",
        feature_dim: int = 2048,
    ) -> None:
        _FIDBase.__init__(
            self,
            metric_dtype=metric_dtype,
            model_dtype=model_dtype,
            real_stats=real_stats,
            weights_cache_dir=weights_cache_dir,
            model_name=model_name,
            feature_dim=feature_dim,
        )
        self._real_acts: list[jnp.ndarray] = []

        self._fake_n = MetricState(jnp.array(0, dtype=jnp.int32))
        self._fake_mean = MetricState(jnp.zeros((self._feat_dim,), dtype=self.metric_dtype))
        self._fake_M2 = MetricState(jnp.zeros((self._feat_dim, self._feat_dim), dtype=self.metric_dtype))

    def _append_real_acts(self, imgs: Float[ArrayLike, "batch h w c"]) -> None:
        acts = self._extract_activations(imgs)
        self._real_acts.append(acts)

    def _update_fake_stats(self, imgs: Float[ArrayLike, "batch h w c"]) -> None:
        acts = self._extract_activations(imgs)
        self._update_fake_stats_from_acts(acts)

    def _update_fake_stats_from_acts(self, acts: jnp.ndarray) -> None:
        nb = acts.shape[0]
        mb = jnp.mean(acts, axis=0)
        S2b = acts.T @ acts
        nb_f = jnp.array(nb, dtype=self.metric_dtype)
        M2b = S2b - nb_f * jnp.outer(mb, mb)

        n = self._fake_n[...]
        mean = self._fake_mean[...]
        M2 = self._fake_M2[...]
        n_f = n.astype(self.metric_dtype)
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

        self._fake_n[...] = n_new
        self._fake_mean[...] = mean_new
        self._fake_M2[...] = M2_new

    def update(self, imgs: Float[ArrayLike, "batch h w c"], real: bool) -> None:
        if real:
            self._append_real_acts(imgs)
        else:
            self._update_fake_stats(imgs)

    def reset(self) -> None:
        self._real_acts.clear()
        self._fake_n[...] = jnp.array(0, dtype=jnp.int32)
        self._fake_mean[...] = jnp.zeros((self._feat_dim,), dtype=self.metric_dtype)
        self._fake_M2[...] = jnp.zeros((self._feat_dim, self._feat_dim), dtype=self.metric_dtype)

    def _fake_stats(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self._stats_from_accumulators(self._fake_n[...], self._fake_mean[...], self._fake_M2[...])

    def compute(self) -> float:
        mu1, sigma1 = self._fake_stats()

        if self.real_stats is not None:
            mu2 = jnp.asarray(self.real_stats["mu"], dtype=self.metric_dtype)
            sigma2 = jnp.asarray(self.real_stats["sigma"], dtype=self.metric_dtype)
        else:
            mu2, sigma2 = self._stack_stats(self._real_acts)

        return float(self._fid_from_stats(mu1, sigma1, mu2, sigma2))

    @property
    def real_count(self) -> int:
        return int(sum(a.shape[0] for a in self._real_acts))

    @property
    def fake_count(self) -> int:
        return int(self._fake_n[...])
