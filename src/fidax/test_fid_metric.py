"""Test for Fréchet Inception Distance (FID) metric implementation."""

import logging
import os
import pickle
import tempfile
import time
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"  # Disable preallocation to use only 50% of GPU memory
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchmetrics
import torchvision
import torchvision.transforms as T
import torchvision.transforms.v2 as T2
from jaxtyping import install_import_hook

with install_import_hook("scripts", "typeguard.typechecked"):
    from fidax.fid import FrechetInceptionDistance

# activate fp64
jax.config.update("jax_enable_x64", True)

logger = logging.getLogger("fidax.tests.timing")


def test_fid_equivalence_to_torchmetrics() -> None:
    """Test JAX FID implementation against torchmetrics for equivalence, using batched updates for larger N."""
    # Generate random fake and real images in [-1, 1], shape [N, 299, 299, 3] for jax
    N = 128  # Larger N
    batch_size = 32
    np.random.seed(0)  # For reproducibility
    fake_imgs = np.random.uniform(size=(N, 299, 299, 3), low=-1, high=1)
    fake_imgs = fake_imgs / 2 + 0.5
    real_imgs = np.random.uniform(size=(N, 299, 299, 3), low=-1, high=1)
    real_imgs = real_imgs / 2 + 0.5

    # Torchmetrics expects [N, C, H, W] in [0, 1]
    fake_imgs_torch = torch.tensor(np.array(fake_imgs)).permute(0, 3, 1, 2)
    real_imgs_torch = torch.tensor(np.array(real_imgs)).permute(0, 3, 1, 2)

    # PyTorch FID (use CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_imgs_torch = fake_imgs_torch.to(device)
    real_imgs_torch = real_imgs_torch.to(device)
    fid_torch = torchmetrics.image.FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid_torch = fid_torch.set_dtype(torch.float64)
    t0 = time.perf_counter()
    for i in range(0, N, batch_size):
        fid_torch.update(real_imgs_torch[i : i + batch_size], real=True)
        fid_torch.update(fake_imgs_torch[i : i + batch_size], real=False)
    fid_torch_score = fid_torch.compute().item()
    t_torch = time.perf_counter() - t0
    # clear CUDA memory
    torch.cuda.empty_cache()

    # JAX FrechetInceptionDistance (NNX style)
    fid_jax = FrechetInceptionDistance()
    t0 = time.perf_counter()
    for i in range(0, N, batch_size):
        fid_jax.update(real_imgs[i : i + batch_size], True)
        fid_jax.update(fake_imgs[i : i + batch_size], False)
    jax_score = float(fid_jax.compute())
    t_jax = time.perf_counter() - t0

    logger.info(
        "timing equivalence: torchmetrics=%.3fs | jax=%.3fs | speedup x%.2f",
        t_torch,
        t_jax,
        (t_torch / max(t_jax, 1e-9)),
    )

    # Allow a small tolerance due to possible implementation/model differences
    assert np.allclose(jax_score, fid_torch_score, rtol=1e-1, atol=1e-1), (
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
    fake_imgs = fake_imgs / 2 + 0.5
    real_imgs = np.random.uniform(size=(N, 299, 299, 3), low=-1, high=1)
    real_imgs = real_imgs / 2 + 0.5

    # First calculate regular FID to get real stats (batched)
    fid_jax = FrechetInceptionDistance()
    t0 = time.perf_counter()
    for i in range(0, N, batch_size):
        fid_jax.update(real_imgs[i : i + batch_size], True)
    warmup = time.perf_counter() - t0

    # Extract pre-computed stats using the metric helper
    mu_real, sigma_real = fid_jax.get_real_stats()
    real_stats = {"mu": mu_real, "sigma": sigma_real}

    # Now create new FID with pre-computed stats (batched)
    fid_jax_precomputed = FrechetInceptionDistance(real_stats=real_stats)
    t0 = time.perf_counter()
    for i in range(0, N, batch_size):
        fid_jax_precomputed.update(fake_imgs[i : i + batch_size], False)
    jax_score_precomputed = float(fid_jax_precomputed.compute())
    t_precomp = time.perf_counter() - t0

    # Update original FID with fake images and compute (batched)
    t0 = time.perf_counter()
    for i in range(0, N, batch_size):
        fid_jax.update(fake_imgs[i : i + batch_size], False)
    jax_score = float(fid_jax.compute())
    t_full = time.perf_counter() - t0

    logger.info(
        "timing precomp: warmup=%.3fs | precomp=%.3fs | full=%.3fs",
        warmup,
        t_precomp,
        t_full,
    )

    # Both scores should be identical
    assert np.allclose(jax_score, jax_score_precomputed, rtol=1e-5, atol=1e-5), (
        f"Standard FID {jax_score} vs Pre-computed stats FID {jax_score_precomputed}"
    )


def test_fid_on_cifar10_real_vs_modified() -> None:
    """Test FID on CIFAR-10 real images vs. a modified version (e.g., noisy), using batched updates."""
    # Download CIFAR-10 and select a small subset for speed
    transform = T.Compose(
        [
            T.ToTensor(),
        ]
    )
    cifar10 = torchvision.datasets.CIFAR10(root="/tmp/cifar10", train=False, download=True, transform=transform)
    N = 9984
    batch_size = 64
    real_imgs_torch = torch.stack([cifar10[i][0] for i in range(N)])  # [N, C, H, W], [0,1]

    # Create a modified version: add Gaussian noise
    noise = torch.randn_like(real_imgs_torch) * 0.1
    fake_imgs_torch = (real_imgs_torch + noise).clamp(0, 1)

    # save one image for visual inspection
    torchvision.utils.save_image(fake_imgs_torch[0], "./fake_cifar10_image.png")
    # print max and min values
    logger.info(f"Fake image max: {fake_imgs_torch[0].max().item()}")

    # PyTorch FID (use CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_imgs_torch = fake_imgs_torch.to(device)
    real_imgs_torch = real_imgs_torch.to(device)
    fid_torch = torchmetrics.image.FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid_torch = fid_torch.set_dtype(torch.float64)
    t0 = time.perf_counter()
    for i in range(0, N, batch_size):
        fid_torch.update(real_imgs_torch[i : i + batch_size], real=True)
        fid_torch.update(fake_imgs_torch[i : i + batch_size], real=False)
    fid_torch_score = fid_torch.compute().item()
    t_torch = time.perf_counter() - t0
    # Move tensors back to CPU for further processing if needed
    fake_imgs_torch = fake_imgs_torch.cpu()
    real_imgs_torch = real_imgs_torch.cpu()
    # clear CUDA memory
    torch.cuda.empty_cache()

    # Convert to JAX format: [N, 299, 299, 3]
    real_imgs = real_imgs_torch.permute(0, 2, 3, 1).numpy()
    fake_imgs = fake_imgs_torch.permute(0, 2, 3, 1).numpy()
    real_imgs = jnp.array(real_imgs)
    fake_imgs = jnp.array(fake_imgs)

    # JAX FID (batched)
    fid_jax = FrechetInceptionDistance()
    t0 = time.perf_counter()
    for i in range(0, N, batch_size):
        fid_jax.update(real_imgs[i : i + batch_size], True)
        fid_jax.update(fake_imgs[i : i + batch_size], False)
    jax_score = float(fid_jax.compute())
    t_jax = time.perf_counter() - t0

    logger.info(
        "timing CIFAR noisy: torchmetrics=%.3fs | jax=%.3fs | speedup x%.2f",
        t_torch,
        t_jax,
        (t_torch / max(t_jax, 1e-9)),
    )

    # log the scores
    logger.info("FID Scores: JAX=%.3f | Torchmetrics=%.3f", jax_score, fid_torch_score)

    # FID should be > 0 (since fake is noisy version of real)
    assert jax_score > 0.0
    assert fid_torch_score > 0.0
    # The two implementations should be close
    assert np.allclose(jax_score, fid_torch_score, rtol=1e-1, atol=1e-1), (
        f"JAX FID {jax_score} vs Torchmetrics {fid_torch_score}"
    )


def test_fid_on_cifar10_real_vs_random_erasing() -> None:
    """Test FID on CIFAR-10 real images vs. a RandomErasing-augmented version, using batched updates."""
    # Download CIFAR-10 and select a small subset for speed
    transform = T.Compose(
        [
            T.ToTensor(),
        ]
    )
    cifar10 = torchvision.datasets.CIFAR10(root="/tmp/cifar10", train=False, download=True, transform=transform)
    N = 9984
    batch_size = 64
    real_imgs_torch = torch.stack([cifar10[i][0] for i in range(N)])  # [N, C, H, W], [0,1]

    # Create a modified version: apply RandomErasing
    random_erasing = T2.RandomErasing(p=1.0, scale=(0.2, 0.4), ratio=(0.3, 3.3))
    fake_imgs_torch = torch.stack([random_erasing(img) for img in real_imgs_torch])
    # save one image for visual inspection
    torchvision.utils.save_image(fake_imgs_torch[0], "./erased_fake_cifar10_image.png")
    # print max and min values
    logger.info(f"Fake image max: {fake_imgs_torch[0].max().item()}")

    # PyTorch FID (use CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_imgs_torch = fake_imgs_torch.to(device)
    real_imgs_torch = real_imgs_torch.to(device)
    fid_torch = torchmetrics.image.FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid_torch = fid_torch.set_dtype(torch.float64)
    t0 = time.perf_counter()
    for i in range(0, N, batch_size):
        fid_torch.update(real_imgs_torch[i : i + batch_size], real=True)
        fid_torch.update(fake_imgs_torch[i : i + batch_size], real=False)
    fid_torch_score = fid_torch.compute().item()
    t_torch = time.perf_counter() - t0
    # Move tensors back to CPU for further processing if needed
    fake_imgs_torch = fake_imgs_torch.cpu()
    real_imgs_torch = real_imgs_torch.cpu()
    # clear CUDA memory
    torch.cuda.empty_cache()

    # Convert to JAX format: [N, 299, 299, 3], [-1, 1]
    real_imgs = real_imgs_torch.permute(0, 2, 3, 1).numpy()
    fake_imgs = fake_imgs_torch.permute(0, 2, 3, 1).numpy()
    real_imgs = jnp.array(real_imgs)
    fake_imgs = jnp.array(fake_imgs)

    # JAX FID (batched)
    fid_jax = FrechetInceptionDistance()
    t0 = time.perf_counter()
    for i in range(0, N, batch_size):
        fid_jax.update(real_imgs[i : i + batch_size], True)
        fid_jax.update(fake_imgs[i : i + batch_size], False)
    jax_score = float(fid_jax.compute())
    t_jax = time.perf_counter() - t0

    logger.info(
        "timing CIFAR erase: torchmetrics=%.3fs | jax=%.3fs | speedup x%.2f",
        t_torch,
        t_jax,
        (t_torch / max(t_jax, 1e-9)),
    )

    # log the scores
    logger.info("FID Scores: JAX=%.3f | Torchmetrics=%.3f", jax_score, fid_torch_score)

    # FID should be > 0 (since fake is RandomErasing version of real)
    assert jax_score > 0.0
    assert fid_torch_score > 0.0
    # The two implementations should be close
    assert np.allclose(jax_score, fid_torch_score, rtol=1e-1, atol=1e-1), (
        f"JAX FID {jax_score} vs Torchmetrics {fid_torch_score}"
    )


def test_fid_weights_cache_dir_uses_custom_path(tmp_path, monkeypatch) -> None:
    """Verify that passing weights_cache_dir stores weights in that directory and passes it to downloader.

    We monkeypatch fidax.inception.download to avoid network and to assert the ckpt_dir used.
    """
    seen = {}

    def fake_download(_url: str, ckpt_dir: str | None = "data") -> str:
        # Record the directory the metric passed to the downloader
        seen["ckpt_dir"] = ckpt_dir
        # Emulate download logic: if no dir provided, use system temp dir
        target_dir = Path(ckpt_dir) if ckpt_dir is not None else Path(tempfile.gettempdir())
        target_dir.mkdir(parents=True, exist_ok=True)
        # Create an empty, but valid pickle file for weights
        fname = "inception_v3_weights_fid.pickle"
        fpath = target_dir / fname
        if not fpath.exists():
            with fpath.open("wb") as f:
                pickle.dump({}, f)
        return str(fpath)

    # Patch the download function used by the Inception model
    monkeypatch.setattr("fidax.inception.download", fake_download)

    cache_dir = tmp_path / "weights-cache"

    # Create the metric with a custom cache directory; this triggers model init and our fake download
    _ = FrechetInceptionDistance(weights_cache_dir=str(cache_dir))

    # The downloader should have been called with our custom directory
    assert seen.get("ckpt_dir") == str(cache_dir)
    # And the weights file should exist in that directory
    assert (cache_dir / "inception_v3_weights_fid.pickle").exists()
