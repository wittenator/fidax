"""Tests comparing fidax metrics with a DinoV2 backbone against torchmetrics with a DinoV2 feature extractor.

torchmetrics has no built-in DinoV2 backend, but it accepts any ``torch.nn.Module`` as the
``feature`` argument. We build a torch DinoV2 CLS-token extractor that uses the canonical
HuggingFace preprocessing (``AutoImageProcessor``: shortest-edge 256 bicubic resize, 224 center
crop, ImageNet normalization) and compare the resulting FID/MiFID scores.
"""

import logging
import time

import jax
import numpy as np
import pytest
import torch
import torchmetrics

from fidax.fid import FrechetInceptionDistance
from fidax.mifid import MemorizationInformedFrechetInceptionDistance

# activate fp64
jax.config.update("jax_enable_x64", True)

logger = logging.getLogger("fidax.tests.timing")

MODEL_NAME = "facebook/dinov2-small"
# 256x256 inputs make the shortest-edge-256 resize an identity in both pipelines, so the
# preprocessing reduces to an exact center-crop + normalize and the comparison is tight up to
# fp32 model numerics. Resize equivalence is covered separately by test_image_preprocessor.py.
IMAGE_SIZE = 256
N = 64
BATCH_SIZE = 16


class TorchDinoV2FeatureExtractor(torch.nn.Module):
    """DinoV2 CLS-token feature extractor for torchmetrics' ``feature`` argument.

    Expects float images in [0, 1] with shape [B, 3, H, W] (torchmetrics passes inputs
    through unchanged when a custom feature module is used and ``normalize=True``).
    """

    def __init__(self, model_name: str = MODEL_NAME, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        from transformers import AutoImageProcessor, Dinov2Model

        # float64 weights so the comparison is not limited by framework-specific fp32 kernels
        self.model = Dinov2Model.from_pretrained(model_name, cache_dir="data", torch_dtype=dtype).eval()
        # torchmetrics reads this to size its accumulators (avoids a dummy forward)
        self.num_features = self.model.config.hidden_size
        self.processor = AutoImageProcessor.from_pretrained(model_name, cache_dir="data")

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        # The HF processor expects uint8/PIL-style images (it rescales by 1/255 itself)
        imgs_uint8 = (imgs.float().clamp(0.0, 1.0) * 255).round().to(torch.uint8)
        inputs = self.processor(images=list(imgs_uint8.permute(0, 2, 3, 1).numpy()), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.model.dtype)
        return self.model(pixel_values=pixel_values).last_hidden_state[:, 0, :]


@pytest.fixture(scope="module")
def torch_dinov2() -> TorchDinoV2FeatureExtractor:
    return TorchDinoV2FeatureExtractor()


@pytest.fixture(scope="module")
def image_sets() -> tuple[np.ndarray, np.ndarray]:
    """Real images and a noisy variant, [N, H, W, 3] in [0, 1].

    Values are quantized to uint8 precision so the float images seen by fidax are bit-identical
    to the uint8 images consumed by the HuggingFace processor on the torchmetrics side.
    """
    rng = np.random.default_rng(0)
    # Blocky low-frequency images so DinoV2 sees some spatial structure
    base = rng.uniform(0.0, 1.0, size=(N, 8, 8, 3))
    real = np.kron(base, np.ones((1, IMAGE_SIZE // 8, IMAGE_SIZE // 8, 1)))
    fake = np.clip(real + rng.normal(0.0, 0.15, size=real.shape), 0.0, 1.0)
    real = np.round(real * 255) / 255
    fake = np.round(fake * 255) / 255
    return real.astype(np.float32), fake.astype(np.float32)


def test_fid_dinov2_matches_torchmetrics(torch_dinov2, image_sets) -> None:
    real_imgs, fake_imgs = image_sets
    real_imgs_torch = torch.from_numpy(real_imgs).permute(0, 3, 1, 2)
    fake_imgs_torch = torch.from_numpy(fake_imgs).permute(0, 3, 1, 2)

    fid_torch = torchmetrics.image.FrechetInceptionDistance(feature=torch_dinov2, normalize=True)
    t0 = time.perf_counter()
    for i in range(0, N, BATCH_SIZE):
        fid_torch.update(real_imgs_torch[i : i + BATCH_SIZE], real=True)
        fid_torch.update(fake_imgs_torch[i : i + BATCH_SIZE], real=False)
    fid_torch_score = fid_torch.compute().item()
    t_torch = time.perf_counter() - t0

    fid_jax = FrechetInceptionDistance(
        model_name=MODEL_NAME, model_dtype="float64", feature_dim=torch_dinov2.num_features
    )
    t0 = time.perf_counter()
    for i in range(0, N, BATCH_SIZE):
        fid_jax.update(real_imgs[i : i + BATCH_SIZE], True)
        fid_jax.update(fake_imgs[i : i + BATCH_SIZE], False)
    jax_score = float(fid_jax.compute())
    t_jax = time.perf_counter() - t0

    logger.info(
        "DinoV2 FID: jax=%.6f (%.2fs) | torchmetrics=%.6f (%.2fs)",
        jax_score,
        t_jax,
        fid_torch_score,
        t_torch,
    )

    # Preprocessed pixels are bit-identical and fidax patches FlaxDinov2's position-embedding
    # interpolation to match the torch reference (see fidax.models.dinov2). The residual comes
    # from torch computing that interpolation internally in fp32; observed agreement is ~1e-6.
    assert np.allclose(jax_score, fid_torch_score, rtol=1e-4, atol=1e-3), (
        f"JAX DinoV2 FID {jax_score} vs Torchmetrics DinoV2 FID {fid_torch_score}"
    )


def test_mifid_dinov2_matches_torchmetrics(torch_dinov2, image_sets) -> None:
    """With clearly non-memorized fakes the penalty is thresholded to 1 and MiFID == FID."""
    real_imgs, fake_imgs = image_sets
    real_imgs_torch = torch.from_numpy(real_imgs).permute(0, 3, 1, 2)
    fake_imgs_torch = torch.from_numpy(fake_imgs).permute(0, 3, 1, 2)

    mifid_torch = torchmetrics.image.MemorizationInformedFrechetInceptionDistance(
        feature=torch_dinov2, normalize=True, cosine_distance_eps=0.1
    )
    for i in range(0, N, BATCH_SIZE):
        mifid_torch.update(real_imgs_torch[i : i + BATCH_SIZE], real=True)
        mifid_torch.update(fake_imgs_torch[i : i + BATCH_SIZE], real=False)
    mifid_torch_score = mifid_torch.compute().item()

    mifid_jax = MemorizationInformedFrechetInceptionDistance(
        model_name=MODEL_NAME,
        model_dtype="float64",
        feature_dim=torch_dinov2.num_features,
        cosine_distance_eps=0.1,
    )
    # fidax MiFID requires all real batches before the first fake batch
    for i in range(0, N, BATCH_SIZE):
        mifid_jax.update(real_imgs[i : i + BATCH_SIZE], True)
    for i in range(0, N, BATCH_SIZE):
        mifid_jax.update(fake_imgs[i : i + BATCH_SIZE], False)
    jax_score = float(mifid_jax.compute())

    logger.info("DinoV2 MiFID: jax=%.6f | torchmetrics=%.6f", jax_score, mifid_torch_score)

    # Sanity-check that this case exercises the thresholded branch (penalty == 1)
    assert float(mifid_jax.get_penalty()) == 1.0

    # Same tolerance rationale as test_fid_dinov2_matches_torchmetrics
    assert np.allclose(jax_score, mifid_torch_score, rtol=1e-4, atol=1e-3), (
        f"JAX DinoV2 MiFID {jax_score} vs Torchmetrics DinoV2 MiFID {mifid_torch_score}"
    )


def test_mifid_dinov2_penalty_branch_matches_torchmetrics(torch_dinov2, image_sets) -> None:
    """Near-duplicate fakes activate the memorization penalty (mean min cosine distance < eps).

    Note on orientation: torchmetrics computes the penalty as the mean over *real* samples of
    the min cosine distance to the fake set, while fidax follows the MiFID paper and averages
    over *fake* samples the min distance to the real set. FID itself is symmetric in the two
    distributions, so feeding torchmetrics with swapped real/fake roles makes both libraries
    compute the same quantity.
    """
    real_imgs, _ = image_sets
    rng = np.random.default_rng(1)
    fake_imgs = np.clip(real_imgs + rng.normal(0.0, 0.02, size=real_imgs.shape), 0.0, 1.0)
    fake_imgs = (np.round(fake_imgs * 255) / 255).astype(np.float32)
    real_imgs_torch = torch.from_numpy(real_imgs).permute(0, 3, 1, 2)
    fake_imgs_torch = torch.from_numpy(fake_imgs).permute(0, 3, 1, 2)

    mifid_torch = torchmetrics.image.MemorizationInformedFrechetInceptionDistance(
        feature=torch_dinov2, normalize=True, cosine_distance_eps=0.1
    )
    for i in range(0, N, BATCH_SIZE):
        # roles swapped on purpose, see docstring
        mifid_torch.update(fake_imgs_torch[i : i + BATCH_SIZE], real=True)
        mifid_torch.update(real_imgs_torch[i : i + BATCH_SIZE], real=False)
    mifid_torch_score = mifid_torch.compute().item()

    mifid_jax = MemorizationInformedFrechetInceptionDistance(
        model_name=MODEL_NAME,
        model_dtype="float64",
        feature_dim=torch_dinov2.num_features,
        cosine_distance_eps=0.1,
    )
    for i in range(0, N, BATCH_SIZE):
        mifid_jax.update(real_imgs[i : i + BATCH_SIZE], True)
    for i in range(0, N, BATCH_SIZE):
        mifid_jax.update(fake_imgs[i : i + BATCH_SIZE], False)
    jax_score = float(mifid_jax.compute())
    penalty = float(mifid_jax.get_penalty())

    logger.info(
        "DinoV2 MiFID (penalty branch): jax=%.6f | torchmetrics=%.6f | penalty=%.6f",
        jax_score,
        mifid_torch_score,
        penalty,
    )

    # Sanity-check that this case exercises the penalty branch
    assert 0.0 < penalty < 0.1

    # Same tolerance rationale as test_fid_dinov2_matches_torchmetrics; observed agreement ~2e-6
    assert np.allclose(jax_score, mifid_torch_score, rtol=1e-4, atol=1e-3), (
        f"JAX DinoV2 MiFID {jax_score} vs Torchmetrics DinoV2 MiFID {mifid_torch_score}"
    )
