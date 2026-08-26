"""Pytest tests comparing JAX Vendi Score implementation against the reference NumPy/PyTorch version.

Tests on MNIST and CIFAR-10 datasets to ensure correctness.

Run with: pytest test_vendi_jax.py -v
"""

import jax.numpy as jnp
import numpy as np
import pytest
import torchvision
from torchvision import transforms
from vendi_score import vendi as vendi_ref
from vendi_score.image_utils import get_pixel_vectors

from fidax.vendi import (
    VendiScore,
    entropy_q,
    normalize_features,
    score_dual,
    score_X,
    vendi_score_images,
)


@pytest.fixture
def mnist_data():
    """Load MNIST test data."""
    dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    return dataset


@pytest.fixture
def cifar10_data():
    """Load CIFAR-10 test data."""
    dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
    return dataset


def get_sample_images(dataset, n_samples=100, seed=42):
    """Extract n_samples from dataset as numpy arrays."""
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    images = []
    for idx in indices:
        img, _ = dataset[idx]
        # Convert from torch tensor (C, H, W) to numpy (H, W, C)
        img_np = img.numpy().transpose(1, 2, 0)
        images.append(img_np)

    return np.array(images)


def reference_pixel_vendi(images, resize=None):
    """Compute Vendi score using reference implementation."""
    from PIL import Image

    # Convert numpy arrays to PIL Images for reference implementation
    pil_images = []
    for img in images:
        # Handle grayscale vs RGB
        if img.shape[-1] == 1:
            img_uint8 = (img.squeeze() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8, mode="L")
        else:
            img_uint8 = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8, mode="RGB")
        pil_images.append(pil_img)

    # Use reference implementation
    X = get_pixel_vectors(pil_images, resize=resize)
    n, d = X.shape

    if n < d:
        return vendi_ref.score_X(X)
    return vendi_ref.score_dual(X)


class TestEntropyQ:
    """Test entropy_q function against reference."""

    def test_shannon_entropy(self):
        """Test Shannon entropy (q=1)."""
        # Uniform distribution
        p_np = np.ones(10) / 10
        p_jax = jnp.array(p_np)

        ref_entropy = vendi_ref.entropy_q(p_np, q=1)
        jax_entropy = float(entropy_q(p_jax, q=1))

        assert np.isclose(ref_entropy, jax_entropy, rtol=1e-5)

    def test_shannon_entropy_nonuniform(self):
        """Test Shannon entropy with non-uniform distribution."""
        p_np = np.array([0.1, 0.2, 0.3, 0.4])
        p_jax = jnp.array(p_np)

        ref_entropy = vendi_ref.entropy_q(p_np, q=1)
        jax_entropy = float(entropy_q(p_jax, q=1))

        assert np.isclose(ref_entropy, jax_entropy, rtol=1e-5)

    def test_renyi_entropy(self):
        """Test Rényi entropy with q=2."""
        p_np = np.array([0.1, 0.2, 0.3, 0.4])
        p_jax = jnp.array(p_np)

        ref_entropy = vendi_ref.entropy_q(p_np, q=2)
        jax_entropy = float(entropy_q(p_jax, q=2))

        assert np.isclose(ref_entropy, jax_entropy, rtol=1e-5)

    def test_entropy_with_zeros(self):
        """Test entropy with some zero probabilities."""
        p_np = np.array([0.0, 0.3, 0.0, 0.7])
        p_jax = jnp.array(p_np)

        ref_entropy = vendi_ref.entropy_q(p_np, q=1)
        jax_entropy = float(entropy_q(p_jax, q=1))

        assert np.isclose(ref_entropy, jax_entropy, rtol=1e-5)


class TestCoreVendiFunctions:
    """Test core Vendi score functions."""

    def test_score_X_small_random(self):
        """Test score_X on small random data."""
        np.random.seed(42)
        X_np = np.random.randn(50, 100)
        X_jax = jnp.array(X_np)

        ref_score = vendi_ref.score_X(X_np, normalize=True)
        jax_score = float(score_X(X_jax, normalize=True))

        assert np.isclose(ref_score, jax_score, rtol=1e-4), (
            f"score_X mismatch: ref={ref_score:.6f}, jax={jax_score:.6f}"
        )

    def test_score_dual_small_random(self):
        """Test score_dual on small random data."""
        np.random.seed(42)
        X_np = np.random.randn(100, 50)
        X_jax = jnp.array(X_np)

        ref_score = vendi_ref.score_dual(X_np, normalize=True)
        jax_score = float(score_dual(X_jax, normalize=True))

        assert np.isclose(ref_score, jax_score, rtol=1e-4), (
            f"score_dual mismatch: ref={ref_score:.6f}, jax={jax_score:.6f}"
        )

    def test_normalization_consistency(self):
        """Test that normalization produces same results."""
        np.random.seed(42)
        X_np = np.random.randn(30, 20)
        X_jax = jnp.array(X_np)

        # Reference normalization
        from sklearn.preprocessing import normalize as sk_normalize

        X_norm_ref = sk_normalize(X_np, axis=1)

        # JAX normalization
        X_norm_jax = np.array(normalize_features(X_jax))

        assert np.allclose(X_norm_ref, X_norm_jax, rtol=1e-5), (
            "Feature normalization differs between reference and JAX"
        )


class TestMNIST:
    """Test on MNIST dataset."""

    def test_mnist_small_sample(self, mnist_data):
        """Test Vendi score on small MNIST sample."""
        images = get_sample_images(mnist_data, n_samples=50, seed=42)

        # Reference implementation
        ref_score = reference_pixel_vendi(images, resize=28)

        # JAX implementation (grayscale MNIST: H, W, C where C=1)
        images_jax = jnp.array(images)
        jax_score = float(vendi_score_images(images_jax, normalize=True))

        print(f"\nMNIST (n=50): ref={ref_score:.6f}, jax={jax_score:.6f}")

        assert np.isclose(ref_score, jax_score, rtol=1e-3), (
            f"MNIST Vendi score mismatch: ref={ref_score:.6f}, jax={jax_score:.6f}"
        )

    def test_mnist_medium_sample(self, mnist_data):
        """Test Vendi score on medium MNIST sample."""
        images = get_sample_images(mnist_data, n_samples=200, seed=123)

        ref_score = reference_pixel_vendi(images, resize=28)

        images_jax = jnp.array(images)
        jax_score = float(vendi_score_images(images_jax, normalize=True))

        print(f"\nMNIST (n=200): ref={ref_score:.6f}, jax={jax_score:.6f}")

        assert np.isclose(ref_score, jax_score, rtol=1e-3), (
            f"MNIST Vendi score mismatch: ref={ref_score:.6f}, jax={jax_score:.6f}"
        )

    def test_mnist_streaming_metric(self, mnist_data):
        """Test VendiScore metric with streaming updates on MNIST."""
        images = get_sample_images(mnist_data, n_samples=100, seed=42)

        # Reference score (all at once)
        ref_score = reference_pixel_vendi(images, resize=28)

        # JAX streaming metric (batched)
        metric = VendiScore(q=1.0, normalize=True)
        batch_size = 20

        images_jax = jnp.array(images)
        for i in range(0, len(images), batch_size):
            batch = images_jax[i : i + batch_size]
            metric.update(batch)

        jax_score = float(metric.compute())

        print(f"\nMNIST streaming (n=100): ref={ref_score:.6f}, jax={jax_score:.6f}")

        assert np.isclose(ref_score, jax_score, rtol=1e-3), (
            f"MNIST streaming Vendi score mismatch: ref={ref_score:.6f}, jax={jax_score:.6f}"
        )


class TestCIFAR10:
    """Test on CIFAR-10 dataset."""

    def test_cifar10_small_sample(self, cifar10_data):
        """Test Vendi score on small CIFAR-10 sample."""
        images = get_sample_images(cifar10_data, n_samples=50, seed=42)

        ref_score = reference_pixel_vendi(images, resize=32)

        # CIFAR-10 is RGB: (H, W, C) where C=3
        images_jax = jnp.array(images)
        jax_score = float(vendi_score_images(images_jax, normalize=True))

        print(f"\nCIFAR-10 (n=50): ref={ref_score:.6f}, jax={jax_score:.6f}")

        assert np.isclose(ref_score, jax_score, rtol=1e-3), (
            f"CIFAR-10 Vendi score mismatch: ref={ref_score:.6f}, jax={jax_score:.6f}"
        )

    def test_cifar10_medium_sample(self, cifar10_data):
        """Test Vendi score on medium CIFAR-10 sample."""
        images = get_sample_images(cifar10_data, n_samples=200, seed=123)

        ref_score = reference_pixel_vendi(images, resize=32)

        images_jax = jnp.array(images)
        jax_score = float(vendi_score_images(images_jax, normalize=True))

        print(f"\nCIFAR-10 (n=200): ref={ref_score:.6f}, jax={jax_score:.6f}")

        assert np.isclose(ref_score, jax_score, rtol=1e-3), (
            f"CIFAR-10 Vendi score mismatch: ref={ref_score:.6f}, jax={jax_score:.6f}"
        )

    def test_cifar10_streaming_metric(self, cifar10_data):
        """Test VendiScore metric with streaming updates on CIFAR-10."""
        images = get_sample_images(cifar10_data, n_samples=100, seed=42)

        ref_score = reference_pixel_vendi(images, resize=32)

        # JAX streaming metric
        metric = VendiScore(q=1.0, normalize=True)
        batch_size = 25

        images_jax = jnp.array(images)
        for i in range(0, len(images), batch_size):
            batch = images_jax[i : i + batch_size]
            metric.update(batch)

        jax_score = float(metric.compute())

        print(f"\nCIFAR-10 streaming (n=100): ref={ref_score:.6f}, jax={jax_score:.6f}")

        assert np.isclose(ref_score, jax_score, rtol=1e-3), (
            f"CIFAR-10 streaming Vendi score mismatch: ref={ref_score:.6f}, jax={jax_score:.6f}"
        )


class TestMetricOperations:
    """Test metric-specific operations."""

    def test_reset(self, mnist_data):
        """Test that reset() clears the metric state."""
        images = get_sample_images(mnist_data, n_samples=50, seed=42)
        images_jax = jnp.array(images)

        metric = VendiScore()
        metric.update(images_jax)
        score1 = metric.compute()

        metric.reset()
        assert metric.count.value == 0
        assert metric.features.value.size == 0

        # Score after reset should be 1.0 (no diversity)
        score_after_reset = metric.compute()
        assert float(score_after_reset) == 1.0


class TestEdgeCases:
    """Test edge cases and special conditions."""

    def test_single_sample(self):
        """Test with single sample."""
        image = np.random.randn(1, 28, 28, 1)
        image_jax = jnp.array(image)

        score = float(vendi_score_images(image_jax))
        # Single sample should have Vendi score of 1
        assert np.isclose(score, 1.0, rtol=1e-5)

    def test_identical_samples(self):
        """Test with identical samples (no diversity)."""
        # Create 10 identical images
        image = np.random.randn(28, 28, 1)
        images = np.stack([image] * 10, axis=0)
        images_jax = jnp.array(images)

        score = float(vendi_score_images(images_jax))
        # Identical samples should have Vendi score close to 1
        assert score < 1.5, f"Expected low diversity score, got {score:.6f}"

    def test_grayscale_vs_rgb_shapes(self):
        """Test that both grayscale and RGB shapes work."""
        # Grayscale
        gray_images = np.random.randn(20, 28, 28)
        gray_jax = jnp.array(gray_images)
        gray_score = float(vendi_score_images(gray_jax))

        # RGB
        rgb_images = np.random.randn(20, 28, 28, 3)
        rgb_jax = jnp.array(rgb_images)
        rgb_score = float(vendi_score_images(rgb_jax))

        # Both should produce valid scores
        assert gray_score > 0
        assert rgb_score > 0
        print(f"\nGrayscale score: {gray_score:.6f}, RGB score: {rgb_score:.6f}")


class TestScalability:
    """Test performance on larger datasets."""

    @pytest.mark.slow
    def test_large_mnist_sample(self, mnist_data):
        """Test on larger MNIST sample (1000 images)."""
        images = get_sample_images(mnist_data, n_samples=1000, seed=42)

        ref_score = reference_pixel_vendi(images, resize=28)

        images_jax = jnp.array(images)
        jax_score = float(vendi_score_images(images_jax, normalize=True))

        print(f"\nMNIST (n=1000): ref={ref_score:.6f}, jax={jax_score:.6f}")

        assert np.isclose(ref_score, jax_score, rtol=1e-3), (
            f"Large MNIST Vendi score mismatch: ref={ref_score:.6f}, jax={jax_score:.6f}"
        )

    @pytest.mark.slow
    def test_streaming_memory_efficiency(self, cifar10_data):
        """Test streaming with many small batches."""
        images = get_sample_images(cifar10_data, n_samples=500, seed=42)

        ref_score = reference_pixel_vendi(images, resize=32)

        # Stream in very small batches
        metric = VendiScore(q=1.0, normalize=True)
        batch_size = 10

        images_jax = jnp.array(images)
        for i in range(0, len(images), batch_size):
            batch = images_jax[i : i + batch_size]
            metric.update(batch)

        jax_score = float(metric.compute())

        print(f"\nCIFAR-10 streaming many batches (n=500): ref={ref_score:.6f}, jax={jax_score:.6f}")

        assert np.isclose(ref_score, jax_score, rtol=1e-3)
