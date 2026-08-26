"""Improved Precision and Recall for evaluating generative models in JAX.

Implements the manifold-based precision and recall metrics from:
    Kynkäänniemi et al., "Improved Precision and Recall Metric for Assessing
    Generative Models", NeurIPS 2019.

Uses k-nearest neighbor hyperspheres to estimate the support of real and
generated distributions in a learned feature space.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from fidax.fid import _FIDBase

if TYPE_CHECKING:
    from jaxtyping import Array, ArrayLike, Float


def _subsample(features: jnp.ndarray, n: int, seed: int = 0) -> jnp.ndarray:
    """Deterministically subsample ``features`` to ``n`` rows (no-op if already <= n).

    Uses a fixed-seed permutation rather than head-truncation so an ordered
    stream (e.g. a class-sorted real dataset) can't skew the retained subset.
    Host arrays (e.g. precomputed-reference memmaps) are gathered on host so
    only the n selected rows ever reach the device.
    """
    if features.shape[0] <= n:
        return features
    idx = jax.random.permutation(jax.random.key(seed), features.shape[0])[:n]
    if isinstance(features, np.ndarray):
        return jnp.asarray(features[np.asarray(idx)])
    return features[idx]


def _knn_radii(features: jnp.ndarray, k: int, batch_size: int = 4096) -> jnp.ndarray:
    """Compute k-th nearest neighbor squared distance for each sample.

    Processes the distance matrix in row-batches to avoid materializing
    the full (n, n) matrix.

    Args:
        features: Feature matrix, shape (n, d).
        k: Number of neighbors (the k-th NN distance is returned, skipping self).
        batch_size: Number of rows to process at once.

    Returns:
        Array of shape (n,) with the squared L2 distance to the k-th neighbor.
    """
    n = features.shape[0]
    # Pre-compute squared norms for efficient distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    norms_sq = jnp.sum(features**2, axis=1)  # (n,)

    radii_parts = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # (batch, d) @ (d, n) -> (batch, n)
        dots = features[start:end] @ features.T
        d_block = norms_sq[start:end, None] + norms_sq[None, :] - 2.0 * dots
        d_block = jnp.maximum(d_block, 0.0)  # numerical safety
        # k-th neighbor (index k) skips self-distance at index 0
        radii_parts.append(jnp.sort(d_block, axis=1)[:, k])

    return jnp.concatenate(radii_parts, axis=0)


def _check_coverage(
    query_features: jnp.ndarray,
    ref_features: jnp.ndarray,
    ref_radii: jnp.ndarray,
    batch_size: int = 4096,
) -> jnp.ndarray:
    """For each query sample, check if it falls inside any reference hypersphere.

    Args:
        query_features: (m, d) features to test.
        ref_features: (n, d) features defining the manifold.
        ref_radii: (n,) k-NN radii for each reference sample.
        batch_size: Number of query rows per batch.

    Returns:
        Boolean array of shape (m,) — True if the query is covered.
    """
    ref_norms_sq = jnp.sum(ref_features**2, axis=1)  # (n,)

    covered_parts = []
    for start in range(0, query_features.shape[0], batch_size):
        end = min(start + batch_size, query_features.shape[0])
        q_batch = query_features[start:end]  # (batch, d)
        q_norms_sq = jnp.sum(q_batch**2, axis=1)  # (batch,)
        dots = q_batch @ ref_features.T  # (batch, n)
        d_block = q_norms_sq[:, None] + ref_norms_sq[None, :] - 2.0 * dots
        d_block = jnp.maximum(d_block, 0.0)
        # query j is covered if any ref i satisfies d(j, i) <= radii[i]
        covered_parts.append(jnp.any(d_block <= ref_radii[None, :], axis=1))

    return jnp.concatenate(covered_parts, axis=0)


@partial(jax.jit, static_argnames=("k",))
def _knn_precision_recall_small(
    real_features: Float[Array, "n d"],
    fake_features: Float[Array, "m d"],
    k: int = 3,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """JIT-compiled precision/recall for small sample counts (fits in memory)."""
    # Squared L2 distances via broadcasting
    norms_real = jnp.sum(real_features**2, axis=1)
    norms_fake = jnp.sum(fake_features**2, axis=1)

    d_rr = norms_real[:, None] + norms_real[None, :] - 2.0 * (real_features @ real_features.T)
    d_ff = norms_fake[:, None] + norms_fake[None, :] - 2.0 * (fake_features @ fake_features.T)
    d_rf = norms_real[:, None] + norms_fake[None, :] - 2.0 * (real_features @ fake_features.T)

    d_rr = jnp.maximum(d_rr, 0.0)
    d_ff = jnp.maximum(d_ff, 0.0)
    d_rf = jnp.maximum(d_rf, 0.0)

    radii_real = jnp.sort(d_rr, axis=1)[:, k]  # (n,)
    radii_fake = jnp.sort(d_ff, axis=1)[:, k]  # (m,)

    # Precision: fraction of fake inside real manifold
    precision = jnp.mean(jnp.any(d_rf <= radii_real[:, None], axis=0).astype(jnp.float32))
    # Recall: fraction of real inside fake manifold
    recall = jnp.mean(jnp.any(d_rf <= radii_fake[None, :], axis=1).astype(jnp.float32))

    return precision, recall


def knn_precision_recall(
    real_features: jnp.ndarray,
    fake_features: jnp.ndarray,
    k: int = 3,
    max_full_size: int = 10_000,
    batch_size: int = 4096,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Improved Precision and Recall using k-NN manifold estimation.

    For small sample counts (<= max_full_size), uses a single JIT-compiled
    kernel. For larger counts, falls back to a batched implementation that
    never materializes the full distance matrix.

    Args:
        real_features: Feature vectors for real images, shape (n, d).
        fake_features: Feature vectors for generated images, shape (m, d).
        k: Number of nearest neighbors for manifold estimation.
        max_full_size: Threshold below which the full JIT path is used.
        batch_size: Row batch size for the large-scale path.

    Returns:
        (precision, recall) as scalar arrays.
    """
    n, m = real_features.shape[0], fake_features.shape[0]

    if n <= max_full_size and m <= max_full_size:
        return _knn_precision_recall_small(real_features, fake_features, k=k)

    # Large-scale batched path
    radii_real = _knn_radii(real_features, k=k, batch_size=batch_size)
    radii_fake = _knn_radii(fake_features, k=k, batch_size=batch_size)

    # Precision: fraction of fake samples inside real manifold
    fake_covered = _check_coverage(fake_features, real_features, radii_real, batch_size=batch_size)
    precision = jnp.mean(fake_covered.astype(jnp.float32))

    # Recall: fraction of real samples inside fake manifold
    real_covered = _check_coverage(real_features, fake_features, radii_fake, batch_size=batch_size)
    recall = jnp.mean(real_covered.astype(jnp.float32))

    return precision, recall


class PrecisionRecallState:
    """Shared state that accumulates real and fake feature vectors.

    This is separated so that an ImprovedPrecision and ImprovedRecall
    instance can share one state object (and one feature extractor),
    avoiding duplicate forward passes when both are registered in a
    MultiMetric.
    """

    def __init__(
        self,
        k: int = 3,
        weights_cache_dir: str | None = "data",
        model_name: str = "inception_v3",
        model_dtype: str = "float32",
        feature_dim: int = 2048,
        model: nnx.Module | None = None,
        image_processor: nnx.Module | None = None,
    ) -> None:
        self.k = k
        self._real_acts: list[jnp.ndarray] = []
        self._fake_acts: list[jnp.ndarray] = []

        # Build feature extractor via fidax. When ``model`` and
        # ``image_processor`` are provided, the extractor is reused so a
        # single encoder can be shared across several metrics.
        if model is not None and image_processor is not None:
            self._fid_base = _FIDBase(
                weights_cache_dir=weights_cache_dir,
                model=model,
                image_processor=image_processor,
                model_dtype=model_dtype,
                feature_dim=feature_dim,
            )
        else:
            self._fid_base = _FIDBase(
                weights_cache_dir=weights_cache_dir,
                model_name=model_name,
                model_dtype=model_dtype,
                feature_dim=feature_dim,
            )

    def update(self, imgs: Float[ArrayLike, "batch h w c"], real: bool) -> None:
        acts = self._fid_base._extract_activations(imgs)
        if real:
            self._real_acts.append(acts)
        else:
            self._fake_acts.append(acts)

    def compute(self) -> tuple[Float[Array, ""], Float[Array, ""]]:
        if not self._real_acts or not self._fake_acts:
            return jnp.array(0.0), jnp.array(0.0)

        # Single-entry lists (the precomputed-reference injection path) skip
        # the concat: jnp.concatenate would device-put the full matrix before
        # _subsample discards most of its rows.
        real_features = self._real_acts[0] if len(self._real_acts) == 1 else jnp.concatenate(self._real_acts, axis=0)
        fake_features = self._fake_acts[0] if len(self._fake_acts) == 1 else jnp.concatenate(self._fake_acts, axis=0)
        # k-NN radii shrink with set density, so unequal set sizes bias
        # precision and recall in opposite directions; match sizes before
        # estimating the manifolds (Kynkäänniemi et al. use equal-sized sets).
        n = min(real_features.shape[0], fake_features.shape[0])
        return knn_precision_recall(_subsample(real_features, n), _subsample(fake_features, n), k=self.k)

    def reset(self) -> None:
        self._real_acts.clear()
        self.reset_fake()

    def reset_fake(self) -> None:
        """Clear only the fake-side activations, keeping the real-side features."""
        self._fake_acts.clear()


class ImprovedPrecision(nnx.Metric):
    """Improved Precision metric (Kynkäänniemi et al., 2019).

    Measures the fraction of generated samples that fall within the
    support of the real data distribution, estimated via k-NN manifolds
    in a learned feature space.

    Can share a ``PrecisionRecallState`` with an ``ImprovedRecall`` instance
    to avoid duplicate feature extraction. If no shared state is provided,
    creates its own.

    Example::

        state = PrecisionRecallState(k=3, model_name="facebook/dinov2-base", feature_dim=768)
        precision = ImprovedPrecision(state=state)
        recall = ImprovedRecall(state=state)

        for batch in real_loader:
            precision.update(batch, real=True)
        for batch in fake_loader:
            precision.update(batch, real=False)

        print(precision.compute(), recall.compute())
    """

    def __init__(
        self,
        k: int = 3,
        weights_cache_dir: str | None = "data",
        model_name: str = "inception_v3",
        model_dtype: str = "float32",
        feature_dim: int = 2048,
        state: PrecisionRecallState | None = None,
    ) -> None:
        if state is not None:
            self._state = state
        else:
            self._state = PrecisionRecallState(
                k=k,
                weights_cache_dir=weights_cache_dir,
                model_name=model_name,
                model_dtype=model_dtype,
                feature_dim=feature_dim,
            )

    def update(self, imgs: Float[ArrayLike, "batch h w c"], real: bool, **kwargs) -> None:
        self._state.update(imgs, real)

    def compute(self) -> Float[Array, ""]:
        precision, _ = self._state.compute()
        return precision

    def reset(self) -> None:
        self._state.reset()

    def reset_fake(self) -> None:
        """Reset only the fake side of the shared state; real features are kept."""
        self._state.reset_fake()


class ImprovedRecall(nnx.Metric):
    """Improved Recall metric (Kynkäänniemi et al., 2019).

    Measures the fraction of real samples that fall within the support
    of the generated data distribution, estimated via k-NN manifolds.

    Should share a ``PrecisionRecallState`` with an ``ImprovedPrecision``
    instance to avoid duplicate feature extraction and accumulation.

    Example::

        state = PrecisionRecallState(k=3, model_name="facebook/dinov2-base", feature_dim=768)
        precision = ImprovedPrecision(state=state)
        recall = ImprovedRecall(state=state)
        # update only via precision (shared state); compute from either
    """

    def __init__(self, state: PrecisionRecallState) -> None:
        self._state = state

    def update(self, imgs: Float[ArrayLike, "batch h w c"], real: bool, **kwargs) -> None:
        # No-op when sharing state with ImprovedPrecision — it handles updates.
        pass

    def compute(self) -> Float[Array, ""]:
        _, recall = self._state.compute()
        return recall

    def reset(self) -> None:
        # No-op: reset is handled by the paired ImprovedPrecision.
        pass

    def reset_fake(self) -> None:
        # No-op: fake-side reset is handled by the paired ImprovedPrecision.
        pass
