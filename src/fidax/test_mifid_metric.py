"""Tests for Memorization-Informed Frechet Inception Distance (MiFID)."""

import jax
import numpy as np
from jaxtyping import install_import_hook

with install_import_hook("scripts", "typeguard.typechecked"):
    from fidax.fid import CachedRealFrechetInceptionDistance
    from fidax.mifid import MemorizationInformedFrechetInceptionDistance

# activate fp64
jax.config.update("jax_enable_x64", True)


def test_mifid_matches_fid_when_penalty_one() -> None:
    rng = np.random.default_rng(0)
    n = 32
    batch = 16
    real_imgs = rng.uniform(0.0, 1.0, size=(n, 299, 299, 3)).astype(np.float32)
    fake_imgs = rng.uniform(0.0, 1.0, size=(n, 299, 299, 3)).astype(np.float32)

    mifid = MemorizationInformedFrechetInceptionDistance(cosine_distance_eps=0.0)
    fid = CachedRealFrechetInceptionDistance()

    for i in range(0, n, batch):
        mifid.update(real_imgs[i : i + batch], True)
        fid.update(real_imgs[i : i + batch], True)

    for i in range(0, n, batch):
        mifid.update(fake_imgs[i : i + batch], False)
        fid.update(fake_imgs[i : i + batch], False)

    mifid_score = float(mifid.compute())
    fid_score = float(fid.compute())

    assert np.allclose(mifid_score, fid_score, rtol=1e-5, atol=1e-5), f"MiFID {mifid_score} vs FID {fid_score}"


def test_mifid_penalty_thresholded_for_identical_sets() -> None:
    rng = np.random.default_rng(123)
    n = 32
    batch = 16
    imgs = rng.uniform(0.0, 1.0, size=(n, 299, 299, 3)).astype(np.float32)

    mifid = MemorizationInformedFrechetInceptionDistance(cosine_distance_eps=0.1)

    for i in range(0, n, batch):
        mifid.update(imgs[i : i + batch], True)

    for i in range(0, n, batch):
        mifid.update(imgs[i : i + batch], False)

    penalty = mifid.get_penalty()
    assert -1e-6 <= penalty <= 1.0
    assert penalty < 0.1
