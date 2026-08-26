"""Jax implementation of Vendi Score compatible with Flax nnx.Metric.

This implementation provides a streaming metric for computing diversity scores
on large datasets (tested up to 50k samples) using automatic primal/dual selection.

Supports two modes:
  * Pixel-space (default): flattened raw pixels as feature vectors — measures
    *pixel* diversity.
  * Embedding-space: images are passed through a fidax feature extractor
    (e.g. InceptionV3 or DINOv2) first — measures *semantic* diversity, the
    variant used in the Vendi Score paper for image generation benchmarks.
"""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from fidax.fid import _FIDBase

# Core Vendi Score Functions


@partial(jax.jit, static_argnames=("q",))
def entropy_q(p: Float[Array, "n"], q: float = 1.0, eps: float = 1e-10) -> Float[Array, ""]:
    """Compute q-entropy of probability distribution (JIT-compatible).

    Args:
        p: Probability distribution (e.g., normalized eigenvalues)
        q: Order of entropy (1 for Shannon entropy, 'inf' for min-entropy)
        eps: Threshold for numerical stability

    Returns:
        q-entropy value
    """
    # Create mask for valid probabilities
    mask = p > eps

    # Replace invalid values with 1.0 to avoid log(0) or division issues
    # (these will be masked out in the final sum)
    p_safe = jnp.where(mask, p, 1.0)

    if q == 1:
        # Shannon entropy: -sum(p * log(p)) only for p > eps
        log_p = jnp.log(p_safe)
        contrib = p_safe * log_p
        # Zero out masked elements
        contrib = jnp.where(mask, contrib, 0.0)
        return -jnp.sum(contrib)
    elif q == float("inf") or q == "inf":
        # Min-entropy: -log(max(p)) for p > eps
        # Set invalid values to -inf so they don't affect max
        p_masked = jnp.where(mask, p, -jnp.inf)
        max_p = jnp.max(p_masked)
        # Handle case where all values are masked
        return jnp.where(jnp.any(mask), -jnp.log(max_p), 0.0)
    else:
        # Rényi entropy: log(sum(p^q)) / (1-q) for p > eps
        p_q = p_safe**q
        # Zero out masked elements
        p_q = jnp.where(mask, p_q, 0.0)
        sum_p_q = jnp.sum(p_q)
        # Handle case where all values are masked
        return jnp.where(sum_p_q > 0, jnp.log(sum_p_q) / (1 - q), 0.0)


def normalize_features(X: Float[Array, "n d"]) -> Float[Array, "n d"]:
    """L2 normalize feature vectors (rows of X).

    Args:
        X: Feature matrix of shape (n, d)

    Returns:
        Normalized feature matrix
    """
    norms = jnp.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero
    norms = jnp.where(norms > 0, norms, 1.0)
    return X / norms


def weight_K(K: Float[Array, "n n"], p: Float[Array, "n"] | None = None) -> Float[Array, "n n"]:
    """Apply weights to kernel matrix.

    Args:
        K: Kernel matrix of shape (n, n)
        p: Optional weight vector of shape (n,). If None, uniform weights.

    Returns:
        Weighted kernel matrix
    """
    if p is None:
        return K / K.shape[0]
    else:
        sqrt_p = jnp.sqrt(p)
        return K * jnp.outer(sqrt_p, sqrt_p)


def normalize_K(K: Float[Array, "n n"]) -> Float[Array, "n n"]:
    """Normalize kernel matrix to have unit diagonal.

    Args:
        K: Kernel matrix of shape (n, n)

    Returns:
        Normalized kernel matrix
    """
    d = jnp.sqrt(jnp.diagonal(K))
    # Avoid division by zero
    d = jnp.where(d > 0, d, 1.0)
    return K / jnp.outer(d, d)


@partial(jax.jit, static_argnames=("normalize", "q"))
def score_K(
    K: Float[Array, "n n"], q: float = 1.0, p: Float[Array, "n"] | None = None, normalize: bool = False
) -> Float[Array, ""]:
    """Compute Vendi score from kernel matrix.

    Args:
        K: Kernel/similarity matrix of shape (n, n)
        q: Order parameter for entropy (default=1 for Shannon)
        p: Optional sample weights of shape (n,)
        normalize: Whether to normalize K to unit diagonal

    Returns:
        Vendi score (exponential of entropy of eigenvalues)
    """
    if normalize:
        K = normalize_K(K)

    K_weighted = weight_K(K, p)

    # Compute eigenvalues (only values needed, not vectors)
    eigenvalues = jnp.linalg.eigvalsh(K_weighted)

    # Eigenvalues should be non-negative for valid kernel, but clip for safety
    eigenvalues = jnp.maximum(eigenvalues, 0.0)

    # Normalize to get probability distribution
    eigenvalues = eigenvalues / jnp.sum(eigenvalues)

    return jnp.exp(entropy_q(eigenvalues, q=q))


@partial(jax.jit, static_argnames=("normalize", "q"))
def score_X(
    X: Float[Array, "n d"], q: float = 1.0, p: Float[Array, "n"] | None = None, normalize: bool = True
) -> Float[Array, ""]:
    """Compute Vendi score from feature matrix (primal method).

    Use when n < d (fewer samples than dimensions).
    Builds kernel matrix K = X @ X.T of shape (n, n).

    Args:
        X: Feature matrix of shape (n, d)
        q: Order parameter for entropy
        p: Optional sample weights
        normalize: Whether to L2 normalize features

    Returns:
        Vendi score
    """
    if normalize:
        X = normalize_features(X)

    K = X @ X.T
    return score_K(K, q=q, p=p, normalize=False)


@partial(jax.jit, static_argnames=("normalize", "q"))
def score_dual(X: Float[Array, "n d"], q: float = 1.0, normalize: bool = True) -> Float[Array, ""]:
    """Compute Vendi score from feature matrix (dual method).

    Use when n >= d (more samples than dimensions).
    Builds Gram matrix S = X.T @ X of shape (d, d) instead of (n, n).
    More memory efficient for large n.

    Args:
        X: Feature matrix of shape (n, d)
        q: Order parameter for entropy
        normalize: Whether to L2 normalize features

    Returns:
        Vendi score
    """
    if normalize:
        X = normalize_features(X)

    n = X.shape[0]
    S = X.T @ X / n

    # Compute eigenvalues of scaled Gram matrix
    eigenvalues = jnp.linalg.eigvalsh(S)

    # Filter positive eigenvalues and normalize
    eigenvalues = jnp.maximum(eigenvalues, 0.0)
    eigenvalues = eigenvalues / jnp.sum(eigenvalues)

    return jnp.exp(entropy_q(eigenvalues, q=q))


def score_auto(
    X: Float[Array, "n d"], q: float = 1.0, p: Float[Array, "n"] | None = None, normalize: bool = True
) -> Float[Array, ""]:
    """Compute Vendi score with automatic primal/dual selection.

    Automatically chooses the more efficient method based on n vs d.
    Note: This function is not JIT-compiled due to dynamic control flow.

    Args:
        X: Feature matrix of shape (n, d)
        q: Order parameter for entropy
        p: Optional sample weights (only used in primal method)
        normalize: Whether to normalize features

    Returns:
        Vendi score
    """
    n, d = X.shape

    if n < d:
        # Fewer samples than dimensions: use primal (n x n kernel)
        return score_X(X, q=q, p=p, normalize=normalize)
    else:
        # More samples than dimensions: use dual (d x d Gram matrix)
        # Note: dual method doesn't support weighted samples
        if p is not None:
            # Fall back to primal if weights are provided
            return score_X(X, q=q, p=p, normalize=normalize)
        return score_dual(X, q=q, normalize=normalize)


class VendiScore(nnx.Metric):
    """Streaming Vendi Score metric for measuring dataset diversity.

    Accumulates feature vectors across batches and computes the Vendi score
    at the end. Automatically selects between primal and dual computation
    methods for efficiency.

    Two variants are supported:
      * Pixel-space (default, ``model_name=None``): images are flattened and
        cosine-similarity is computed directly between raw pixel vectors —
        measures *pixel* diversity.
      * Embedding-space (``model_name="inception_v3"`` or
        ``model_name="facebook/dinov2-base"``, etc.): images are passed
        through a fidax feature extractor first — measures *semantic*
        diversity. Use the same ``model_name`` / ``feature_dim`` /
        ``weights_cache_dir`` as the corresponding FID/MIFD/PR metrics.

    Example:
        >>> # Pixel-space (legacy behaviour)
        >>> metric = VendiScore()
        >>> # DINOv2-based semantic Vendi score
        >>> metric = VendiScore(
        ...     model_name="facebook/dinov2-base",
        ...     feature_dim=768,
        ...     weights_cache_dir="data",
        ... )
        >>> for batch in dataset:
        >>>     # batch shape: (B, H, W, C) or (B, H, W)
        >>>     metric.update(batch)
        >>> score = metric.compute()
        >>> print(f"Vendi Score: {score:.2f}")

    Attributes:
        q: Order parameter for entropy (default=1.0 for Shannon entropy)
        normalize: Whether to L2 normalize features (default=True)
    """

    def __init__(
        self,
        q: float = 1.0,
        normalize: bool = True,
        model_name: str | None = None,
        weights_cache_dir: str | None = "data",
        feature_dim: int = 2048,
        model_dtype: str = "float32",
        model: nnx.Module | None = None,
        image_processor: nnx.Module | None = None,
    ):
        """Initialize VendiScore metric.

        Args:
            q: Order parameter for entropy (1.0 for Shannon entropy)
            normalize: Whether to L2 normalize features before computing score
            model_name: If set (e.g. ``"inception_v3"`` or
                ``"facebook/dinov2-base"``), images are passed through the
                corresponding fidax feature extractor before the kernel is
                built. ``None`` (default) keeps the pixel-space behaviour.
            weights_cache_dir: Directory used by fidax to cache extractor
                weights. Ignored when ``model_name`` is ``None``.
            feature_dim: Expected extractor output dimension (2048 for
                InceptionV3, 768 for dinov2-base). Ignored when
                ``model_name`` is ``None``.
            model_dtype: Forward-pass dtype for the extractor. Ignored when
                ``model_name`` is ``None``.
            model: Pre-built extractor model. When combined with
                ``image_processor``, the extractor is reused instead of
                being loaded from ``model_name`` — useful to share one
                encoder across several metrics in a ``MultiMetric``.
            image_processor: Pre-built image preprocessor paired with
                ``model``.
        """
        self.q = q
        self.normalize = normalize

        # Optional feature extractor (semantic-space Vendi).
        self._fid_base: _FIDBase | None
        if model is not None and image_processor is not None:
            self._fid_base = _FIDBase(
                weights_cache_dir=weights_cache_dir,
                model=model,
                image_processor=image_processor,
                model_dtype=model_dtype,
                feature_dim=feature_dim,
            )
        elif model_name is not None:
            self._fid_base = _FIDBase(
                weights_cache_dir=weights_cache_dir,
                model_name=model_name,
                model_dtype=model_dtype,
                feature_dim=feature_dim,
            )
        else:
            self._fid_base = None

        # Initialize state variables
        self.features = nnx.Variable(jnp.array([]).reshape(0, 0))
        self.count = nnx.Variable(jnp.array(0, dtype=jnp.int32))

    def update(
        self, imgs: Float[Array, "B H W C"] | Float[Array, "B H W"], real: bool | None = None, **kwargs
    ) -> None:
        """Accumulate images from a batch.

        In pixel mode images are flattened to feature vectors. When a feature
        extractor is configured, images are passed through it instead.

        Args:
            imgs: Image batch of shape (B, H, W, C) or (B, H, W). When a
                feature extractor is configured, a 4D (B, H, W, C) tensor is
                required (that is what the fidax preprocessor expects).
        """
        # Skip real images if specified
        if real:
            return

        batch_size = imgs.shape[0]
        if self._fid_base is not None:
            # Semantic-space: extract activations via the fidax encoder.
            features = self._fid_base._extract_activations(imgs)
        else:
            # Pixel-space: validate input dimensions and flatten.
            if imgs.ndim not in [3, 4]:
                raise ValueError(f"Images must be 3D (B, H, W) or 4D (B, H, W, C), got shape {imgs.shape}")
            features = imgs.reshape(batch_size, -1)

        # Initialize features array on first update
        if self.features.value.size == 0:
            self.features.value = features
            self.count.value = jnp.array(batch_size, dtype=jnp.int32)
        else:
            # Concatenate new features
            self.features.value = jnp.concatenate([self.features.value, features], axis=0)
            self.count.value = self.count.value + batch_size

    def compute(self) -> Float[Array, ""]:
        """Compute the Vendi score from accumulated features.

        Returns:
            Vendi score (float). Returns 1.0 if no samples accumulated.
        """
        if self.count.value == 0 or self.features.value.size == 0:
            return jnp.array(1.0)  # No diversity with no samples

        return score_auto(
            self.features.value,
            q=self.q,
            p=None,  # Uniform weights
            normalize=self.normalize,
        )

    def reset(self) -> None:
        """Reset the metric state, clearing all accumulated features."""
        self.features.value = jnp.array([]).reshape(0, 0)
        self.count.value = jnp.array(0, dtype=jnp.int32)

    def reset_fake(self) -> None:
        """Alias for :meth:`reset` — Vendi only accumulates generated samples."""
        self.reset()


def prepare_images(images: Float[Array, "n H W C"] | Float[Array, "n H W"]) -> Float[Array, "n d"]:
    """Prepare images for Vendi score computation by flattening.

    Args:
        images: Image array of shape (n, H, W) or (n, H, W, C)

    Returns:
        Flattened feature matrix of shape (n, d) where d = H*W or H*W*C
    """
    batch_size = images.shape[0]
    return images.reshape(batch_size, -1)


def vendi_score_images(
    images: Float[Array, "n H W C"] | Float[Array, "n H W"], q: float = 1.0, normalize: bool = True
) -> Float[Array, ""]:
    """Compute Vendi score directly from image array.

    Convenience function for one-shot computation without metric accumulation.

    Args:
        images: Image array of shape (n, H, W) or (n, H, W, C)
        q: Order parameter for entropy
        normalize: Whether to normalize features

    Returns:
        Vendi score
    """
    features = prepare_images(images)
    return score_auto(features, q=q, p=None, normalize=normalize)
