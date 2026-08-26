"""Utilities for composing fidax metrics."""

from __future__ import annotations

from flax import nnx


def reset_fake(metrics: nnx.MultiMetric) -> None:
    """Reset only the fake side of every metric in a ``MultiMetric``.

    Real-side features and statistics are kept, so several generated sample
    sets can be scored against one fixed real distribution without
    re-streaming the real images through the feature extractors.

    Every wrapped metric must expose a ``reset_fake()`` method (all fidax
    metrics do); a metric without one raises ``AttributeError`` rather than
    silently falling back to a full reset that would drop its real state.
    """
    for name in metrics._metric_names:
        getattr(metrics, name).reset_fake()
