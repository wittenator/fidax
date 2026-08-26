"""Tests for fake-only reset: real-side state must survive ``reset_fake``.

Uses a tiny injected feature extractor (a fixed random projection) instead of
a real backbone, so the tests are fast and hermetic while still exercising the
shared-extractor injection path of every metric.
"""

import jax
import numpy as np
from flax import nnx
from jaxtyping import install_import_hook

with install_import_hook("fidax", "typeguard.typechecked"):
    from fidax.fid import FrechetInceptionDistance, StandardFrechetInceptionDistance
    from fidax.mifid import MemorizationInformedFrechetInceptionDistance
    from fidax.precision_recall import ImprovedPrecision, ImprovedRecall, PrecisionRecallState
    from fidax.utils import reset_fake
    from fidax.vendi import VendiScore

# activate fp64
jax.config.update("jax_enable_x64", True)

FEAT_DIM = 8
IMG_SHAPE = (6, 6, 3)


class _StubProcessor(nnx.Module):
    def __call__(self, imgs):
        return {"pixel_values": imgs}


class _StubModel(nnx.Module):
    """Deterministic extractor: flatten and project to FEAT_DIM."""

    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        self.proj = nnx.data(rng.standard_normal((int(np.prod(IMG_SHAPE)), FEAT_DIM)).astype(np.float32))

    def __call__(self, pixel_values):
        flat = pixel_values.reshape(pixel_values.shape[0], -1)
        return flat @ self.proj


def _build_suite() -> nnx.MultiMetric:
    processor, model = _StubProcessor(), _StubModel()
    pr_state = PrecisionRecallState(k=3, model=model, image_processor=processor, feature_dim=FEAT_DIM)
    return nnx.MultiMetric(
        fid=FrechetInceptionDistance(model=model, image_processor=processor, feature_dim=FEAT_DIM),
        standard_fid=StandardFrechetInceptionDistance(model=model, image_processor=processor, feature_dim=FEAT_DIM),
        mifid=MemorizationInformedFrechetInceptionDistance(
            model=model, image_processor=processor, feature_dim=FEAT_DIM
        ),
        improved_precision=ImprovedPrecision(state=pr_state),
        improved_recall=ImprovedRecall(state=pr_state),
        vendi=VendiScore(model=model, image_processor=processor, feature_dim=FEAT_DIM),
    )


def _imgs(seed: int, n: int = 24) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n, *IMG_SHAPE)).astype(np.float32)


def _feed(metrics: nnx.MultiMetric, imgs: np.ndarray, real: bool, batch: int = 8) -> None:
    for i in range(0, imgs.shape[0], batch):
        metrics.update(imgs=imgs[i : i + batch], real=real)


def test_reset_fake_matches_fresh_suite() -> None:
    """After reset_fake + new fakes, every metric equals a fresh suite fed the same data."""
    real, fake_a, fake_b = _imgs(1), _imgs(2), _imgs(3)

    reused = _build_suite()
    _feed(reused, real, real=True)
    _feed(reused, fake_a, real=False)
    reused.compute()  # scores of fake_a, discarded — just forces full evaluation
    reset_fake(reused)
    _feed(reused, fake_b, real=False)
    result_reused = {k: float(v) for k, v in reused.compute().items()}

    fresh = _build_suite()
    _feed(fresh, real, real=True)
    _feed(fresh, fake_b, real=False)
    result_fresh = {k: float(v) for k, v in fresh.compute().items()}

    assert result_reused.keys() == result_fresh.keys()
    for key in result_fresh:
        assert np.allclose(result_reused[key], result_fresh[key], rtol=1e-10, atol=1e-10), (
            f"{key}: {result_reused[key]} != {result_fresh[key]} after reset_fake"
        )


def test_reset_fake_keeps_real_counts() -> None:
    real, fake = _imgs(1), _imgs(2)
    metrics = _build_suite()
    _feed(metrics, real, real=True)
    _feed(metrics, fake, real=False)
    reset_fake(metrics)

    assert metrics.fid.real_count == real.shape[0]
    assert metrics.fid.fake_count == 0
    assert metrics.standard_fid.real_count == real.shape[0]
    assert metrics.standard_fid.fake_count == 0
    assert metrics.mifid.real_count == real.shape[0]
    assert metrics.mifid.fake_count == 0
    assert int(metrics.mifid._penalty_count[...]) == 0
    assert len(metrics.improved_precision._state._real_acts) > 0
    assert len(metrics.improved_precision._state._fake_acts) == 0
    assert int(metrics.vendi.count.value) == 0


def test_full_reset_still_clears_everything() -> None:
    real, fake = _imgs(1), _imgs(2)
    metrics = _build_suite()
    _feed(metrics, real, real=True)
    _feed(metrics, fake, real=False)
    metrics.reset()

    assert metrics.fid.real_count == 0
    assert metrics.standard_fid.real_count == 0
    assert metrics.mifid.real_count == 0
    assert metrics.mifid._real_norm_cache is None
    assert len(metrics.improved_precision._state._real_acts) == 0
