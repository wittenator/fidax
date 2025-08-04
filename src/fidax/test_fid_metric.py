"""Test for Fréchet Inception Distance (FID) metric implementation."""

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"  # Disable preallocation to use only 50% of GPU memory
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchmetrics
import torchvision
import torchvision.transforms as T
import torchvision.transforms.v2 as T2

from fidax.fid import FrechetInceptionDistance

# activate fp64
jax.config.update("jax_enable_x64", True)


def test_fid_equivalence_to_torchmetrics() -> None:
    """Test JAX FID implementation against torchmetrics for equivalence, using batched updates for larger N."""
    # Generate random fake and real images in [-1, 1], shape [N, 299, 299, 3] for jax
    N = 128  # Larger N
    batch_size = 32
    np.random.seed(0)  # For reproducibility
    fake_imgs = np.random.uniform(size=(N, 299, 299, 3), low=-1, high=1)
    real_imgs = np.random.uniform(size=(N, 299, 299, 3), low=-1, high=1)

    # Torchmetrics expects [N, C, H, W] in [0, 1]
    fake_imgs_torch = torch.from_numpy(np.array(fake_imgs)).permute(0, 3, 1, 2)
    real_imgs_torch = torch.from_numpy(np.array(real_imgs)).permute(0, 3, 1, 2)
    # Convert from [-1, 1] to [0, 1] range
    fake_imgs_torch = (fake_imgs_torch + 1) / 2
    real_imgs_torch = (real_imgs_torch + 1) / 2

    # PyTorch FID (use CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_imgs_torch = fake_imgs_torch.to(device)
    real_imgs_torch = real_imgs_torch.to(device)
    fid_torch = torchmetrics.image.FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid_torch = fid_torch.set_dtype(torch.float64)
    for i in range(0, N, batch_size):
        fid_torch.update(real_imgs_torch[i : i + batch_size], real=True)
        fid_torch.update(fake_imgs_torch[i : i + batch_size], real=False)
    fid_torch_score = fid_torch.compute().item()
    # clear CUDA memory
    torch.cuda.empty_cache()

    # JAX FrechetInceptionDistance (NNX style)
    fid_jax = FrechetInceptionDistance(max_samples=N)
    for i in range(0, N, batch_size):
        fid_jax.update(real_imgs[i : i + batch_size], True)
        fid_jax.update(fake_imgs[i : i + batch_size], False)
    jax_score = fid_jax.compute()

    # Allow a small tolerance due to possible implementation/model differences
    assert np.allclose(jax_score, fid_torch_score, rtol=1e-5, atol=1e-5), (
        f"JAX FID {jax_score} vs Torchmetrics {fid_torch_score}"
    )

    # Test reset functionality
    fid_jax.reset()
    assert fid_jax.real_count == 0
    assert fid_jax.fake_count == 0


def test_fid_with_precomputed_stats() -> None:
    """Test JAX FID implementation with pre-computed real statistics, using batched updates."""
    # Generate random fake images and pre-compute real stats
    N = 1024
    batch_size = 64
    np.random.seed(0)  # For reproducibility
    fake_imgs = np.random.uniform(size=(N, 299, 299, 3), low=-1, high=1)
    real_imgs = np.random.uniform(size=(N, 299, 299, 3), low=-1, high=1)

    # First calculate regular FID to get real stats (batched)
    fid_jax = FrechetInceptionDistance(max_samples=N)
    for i in range(0, N, batch_size):
        fid_jax.update(real_imgs[i : i + batch_size], True)

    # Extract pre-computed stats
    real_acts = fid_jax.real_acts[: fid_jax.real_count]
    mu_real = jnp.mean(real_acts, axis=0)
    sigma_real = jnp.cov(real_acts, rowvar=False)
    real_stats = {"mu": mu_real, "sigma": sigma_real}

    # Now create new FID with pre-computed stats (batched)
    fid_jax_precomputed = FrechetInceptionDistance(max_samples=N, real_stats=real_stats)
    for i in range(0, N, batch_size):
        fid_jax_precomputed.update(fake_imgs[i : i + batch_size], False)
    jax_score_precomputed = fid_jax_precomputed.compute()

    # Update original FID with fake images and compute (batched)
    for i in range(0, N, batch_size):
        fid_jax.update(fake_imgs[i : i + batch_size], False)
    jax_score = fid_jax.compute()

    # Both scores should be identical
    assert np.allclose(jax_score, jax_score_precomputed, rtol=1e-5, atol=1e-5), (
        f"Standard FID {jax_score} vs Pre-computed stats FID {jax_score_precomputed}"
    )


def test_fid_on_cifar10_real_vs_modified() -> None:
    """Test FID on CIFAR-10 real images vs. a modified version (e.g., noisy), using batched updates."""
    # Download CIFAR-10 and select a small subset for speed
    transform = T.Compose(
        [
            T.Resize((299, 299)),
            T.ToTensor(),
        ]
    )
    cifar10 = torchvision.datasets.CIFAR10(root="/tmp/cifar10", train=False, download=True, transform=transform)
    N = 1024
    batch_size = 64
    real_imgs_torch = torch.stack([cifar10[i][0] for i in range(N)])  # [N, C, H, W], [0,1]

    # Create a modified version: add Gaussian noise
    noise = torch.randn_like(real_imgs_torch) * 0.1
    fake_imgs_torch = (real_imgs_torch + noise).clamp(0, 1)

    # PyTorch FID (use CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_imgs_torch = fake_imgs_torch.to(device)
    real_imgs_torch = real_imgs_torch.to(device)
    fid_torch = torchmetrics.image.FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid_torch = fid_torch.set_dtype(torch.float64)
    for i in range(0, N, batch_size):
        fid_torch.update(real_imgs_torch[i : i + batch_size], real=True)
        fid_torch.update(fake_imgs_torch[i : i + batch_size], real=False)
    fid_torch_score = fid_torch.compute().item()
    # Move tensors back to CPU for further processing if needed
    fake_imgs_torch = fake_imgs_torch.cpu()
    real_imgs_torch = real_imgs_torch.cpu()
    # clear CUDA memory
    torch.cuda.empty_cache()

    # Convert to JAX format: [N, 299, 299, 3], [-1, 1]
    real_imgs = real_imgs_torch.permute(0, 2, 3, 1).numpy() * 2 - 1
    fake_imgs = fake_imgs_torch.permute(0, 2, 3, 1).numpy() * 2 - 1
    real_imgs = jnp.array(real_imgs)
    fake_imgs = jnp.array(fake_imgs)

    # JAX FID (batched)
    fid_jax = FrechetInceptionDistance(max_samples=N)
    for i in range(0, N, batch_size):
        fid_jax.update(real_imgs[i : i + batch_size], True)
        fid_jax.update(fake_imgs[i : i + batch_size], False)
    jax_score = fid_jax.compute()

    # FID should be > 0 (since fake is noisy version of real)
    assert jax_score > 0.0
    assert fid_torch_score > 0.0
    # The two implementations should be close
    assert np.allclose(jax_score, fid_torch_score, rtol=1e-4, atol=1e-4), (
        f"JAX FID {jax_score} vs Torchmetrics {fid_torch_score}"
    )


def test_fid_on_cifar10_real_vs_random_erasing() -> None:
    """Test FID on CIFAR-10 real images vs. a RandomErasing-augmented version, using batched updates."""
    # Download CIFAR-10 and select a small subset for speed
    transform = T.Compose(
        [
            T.Resize((299, 299)),
            T.ToTensor(),
        ]
    )
    cifar10 = torchvision.datasets.CIFAR10(root="/tmp/cifar10", train=False, download=True, transform=transform)
    N = 1024
    batch_size = 64
    real_imgs_torch = torch.stack([cifar10[i][0] for i in range(N)])  # [N, C, H, W], [0,1]

    # Create a modified version: apply RandomErasing
    random_erasing = T2.RandomErasing(p=1.0, scale=(0.2, 0.4), ratio=(0.3, 3.3))
    fake_imgs_torch = torch.stack([random_erasing(img) for img in real_imgs_torch])

    # PyTorch FID (use CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_imgs_torch = fake_imgs_torch.to(device)
    real_imgs_torch = real_imgs_torch.to(device)
    fid_torch = torchmetrics.image.FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid_torch = fid_torch.set_dtype(torch.float64)
    for i in range(0, N, batch_size):
        fid_torch.update(real_imgs_torch[i : i + batch_size], real=True)
        fid_torch.update(fake_imgs_torch[i : i + batch_size], real=False)
    fid_torch_score = fid_torch.compute().item()
    # Move tensors back to CPU for further processing if needed
    fake_imgs_torch = fake_imgs_torch.cpu()
    real_imgs_torch = real_imgs_torch.cpu()
    # clear CUDA memory
    torch.cuda.empty_cache()

    # Convert to JAX format: [N, 299, 299, 3], [-1, 1]
    real_imgs = real_imgs_torch.permute(0, 2, 3, 1).numpy() * 2 - 1
    fake_imgs = fake_imgs_torch.permute(0, 2, 3, 1).numpy() * 2 - 1
    real_imgs = jnp.array(real_imgs)
    fake_imgs = jnp.array(fake_imgs)

    # JAX FID (batched)
    fid_jax = FrechetInceptionDistance(max_samples=N)
    for i in range(0, N, batch_size):
        fid_jax.update(real_imgs[i : i + batch_size], True)
        fid_jax.update(fake_imgs[i : i + batch_size], False)
    jax_score = fid_jax.compute()

    # FID should be > 0 (since fake is RandomErasing version of real)
    assert jax_score > 0.0
    assert fid_torch_score > 0.0
    # The two implementations should be close
    assert np.allclose(jax_score, fid_torch_score, rtol=1e-4, atol=1e-4), (
        f"JAX FID {jax_score} vs Torchmetrics {fid_torch_score}"
    )
