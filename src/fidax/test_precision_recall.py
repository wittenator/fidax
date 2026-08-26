"""Tests for Improved Precision and Recall metric."""

import jax.numpy as jnp
import numpy as np

from fidax.precision_recall import (
    ImprovedPrecision,
    ImprovedRecall,
    PrecisionRecallState,
    _knn_precision_recall_small,
    _subsample,
    knn_precision_recall,
)


class TestKnnPrecisionRecall:
    """Test the core k-NN precision/recall computation."""

    def test_identical_distributions(self):
        """Precision and recall should both be ~1 for identical distributions."""
        np.random.seed(42)
        features = jnp.array(np.random.randn(200, 16))
        p, r = _knn_precision_recall_small(features, features, k=3)
        assert float(p) > 0.95
        assert float(r) > 0.95

    def test_subset_precision_high_recall_high(self):
        """When fake is a subset of real, precision should be high."""
        np.random.seed(42)
        real = jnp.array(np.random.randn(300, 16))
        fake = real[:150]  # subset
        p, r = _knn_precision_recall_small(real, fake, k=3)
        assert float(p) > 0.9, f"Precision {p:.4f} should be high for subset"
        assert float(r) > 0.4, f"Recall {r:.4f} should be moderate"

    def test_disjoint_distributions(self):
        """Disjoint distributions should have low precision and recall."""
        np.random.seed(42)
        real = jnp.array(np.random.randn(200, 16) + 100.0)
        fake = jnp.array(np.random.randn(200, 16) - 100.0)
        p, r = _knn_precision_recall_small(real, fake, k=3)
        assert float(p) < 0.05, f"Precision {p:.4f} should be ~0 for disjoint"
        assert float(r) < 0.05, f"Recall {r:.4f} should be ~0 for disjoint"

    def test_mode_collapse(self):
        """Collapsed fake (single point repeated) => high precision, low recall."""
        np.random.seed(42)
        real = jnp.array(np.random.randn(200, 16))
        # Collapse to the mean of real — should land inside the manifold
        center = jnp.mean(real, axis=0, keepdims=True)
        fake = jnp.broadcast_to(center, (200, 16)) + 1e-6 * jnp.array(np.random.randn(200, 16))
        p, r = _knn_precision_recall_small(real, fake, k=3)
        assert float(p) > 0.5, f"Precision {p:.4f} should be high for mode collapse"
        assert float(r) < 0.3, f"Recall {r:.4f} should be low for mode collapse"

    def test_batched_matches_full(self):
        """The batched path should give the same results as the full JIT path."""
        np.random.seed(42)
        real = jnp.array(np.random.randn(500, 32))
        fake = jnp.array(np.random.randn(500, 32))

        p_full, r_full = _knn_precision_recall_small(real, fake, k=3)
        # Force batched path by setting max_full_size=0
        p_batched, r_batched = knn_precision_recall(real, fake, k=3, max_full_size=0, batch_size=128)

        assert np.isclose(float(p_full), float(p_batched), atol=1e-5), (
            f"Precision mismatch: full={p_full:.6f}, batched={p_batched:.6f}"
        )
        assert np.isclose(float(r_full), float(r_batched), atol=1e-5), (
            f"Recall mismatch: full={r_full:.6f}, batched={r_batched:.6f}"
        )

    def test_k_parameter(self):
        """Different k values should produce valid results."""
        np.random.seed(42)
        real = jnp.array(np.random.randn(200, 16))
        fake = jnp.array(np.random.randn(200, 16) * 0.5)

        for k in [1, 3, 5, 10]:
            p, r = _knn_precision_recall_small(real, fake, k=k)
            assert 0.0 <= float(p) <= 1.0, f"Precision {p:.4f} out of [0,1] for k={k}"
            assert 0.0 <= float(r) <= 1.0, f"Recall {r:.4f} out of [0,1] for k={k}"


class TestMetricClasses:
    """Test the nnx.Metric wrapper classes."""

    def test_shared_state(self):
        """ImprovedPrecision and ImprovedRecall should share state correctly."""
        state = PrecisionRecallState.__new__(PrecisionRecallState)
        state.k = 3
        state._real_acts = []
        state._fake_acts = []

        # Manually push features (bypass feature extractor)
        np.random.seed(42)
        real_feats = jnp.array(np.random.randn(100, 32))
        fake_feats = jnp.array(np.random.randn(100, 32))
        state._real_acts.append(real_feats)
        state._fake_acts.append(fake_feats)

        precision_metric = ImprovedPrecision(state=state)
        recall_metric = ImprovedRecall(state=state)

        p = float(precision_metric.compute())
        r = float(recall_metric.compute())

        assert 0.0 <= p <= 1.0
        assert 0.0 <= r <= 1.0

    def test_reset(self):
        """Reset should clear accumulated features."""
        state = PrecisionRecallState.__new__(PrecisionRecallState)
        state.k = 3
        state._real_acts = [jnp.ones((10, 8))]
        state._fake_acts = [jnp.ones((10, 8))]

        precision_metric = ImprovedPrecision(state=state)
        precision_metric.reset()

        assert len(state._real_acts) == 0
        assert len(state._fake_acts) == 0

        # compute on empty state returns 0
        p = float(precision_metric.compute())
        assert p == 0.0

    def test_unequal_sizes_matched_before_knn(self):
        """State.compute subsamples the larger set to the smaller one's size."""
        state = PrecisionRecallState.__new__(PrecisionRecallState)
        state.k = 3
        np.random.seed(42)
        real_feats = jnp.array(np.random.randn(300, 32))
        fake_feats = jnp.array(np.random.randn(100, 32))
        state._real_acts = [real_feats]
        state._fake_acts = [fake_feats]

        p, r = state.compute()
        assert 0.0 <= float(p) <= 1.0
        assert 0.0 <= float(r) <= 1.0

        # The subsample is deterministic, so it must equal recomputing over
        # the same fixed-seed subset directly.
        p2, r2 = knn_precision_recall(_subsample(real_feats, 100), fake_feats, k=3)
        assert float(p) == float(p2)
        assert float(r) == float(r2)

    def test_subsample_deterministic_noop_when_small(self):
        """_subsample is a no-op at-or-below target size and deterministic above it."""
        np.random.seed(42)
        feats = jnp.array(np.random.randn(50, 8))
        assert _subsample(feats, 50) is feats
        assert _subsample(feats, 100) is feats
        sub_a = _subsample(feats, 20)
        sub_b = _subsample(feats, 20)
        assert sub_a.shape == (20, 8)
        assert jnp.array_equal(sub_a, sub_b)

    def test_recall_update_is_noop(self):
        """ImprovedRecall.update should be a no-op (shared state)."""
        state = PrecisionRecallState.__new__(PrecisionRecallState)
        state.k = 3
        state._real_acts = []
        state._fake_acts = []

        recall_metric = ImprovedRecall(state=state)
        # Calling update should not crash and should not add features
        recall_metric.update(imgs=jnp.ones((2, 8, 8, 3)), real=True)
        assert len(state._real_acts) == 0
        assert len(state._fake_acts) == 0
