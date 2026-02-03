from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax.nnx.training.metrics import MetricState

from fidax.fid import CachedRealFrechetInceptionDistance

if TYPE_CHECKING:
    from jaxtyping import Array, ArrayLike, Float


class MemorizationInformedFrechetInceptionDistance(CachedRealFrechetInceptionDistance):
    """Memorization-Informed Frechet Inception Distance (MiFID)."""

    def __init__(
        self,
        metric_dtype: jnp.dtype = jnp.float64,
        model_dtype: str = "float32",
        real_stats: dict[str, jnp.ndarray] | None = None,
        weights_cache_dir: str | None = "data",
        model_name: str = "inception_v3",
        feature_dim: int = 2048,
        cosine_distance_eps: float = 0.1,
    ) -> None:
        super().__init__(
            metric_dtype=metric_dtype,
            model_dtype=model_dtype,
            real_stats=real_stats,
            weights_cache_dir=weights_cache_dir,
            model_name=model_name,
            feature_dim=feature_dim,
        )
        self.cosine_distance_eps = cosine_distance_eps
        self._penalty_sum = MetricState(jnp.array(0.0, dtype=self.metric_dtype))
        self._penalty_count = MetricState(jnp.array(0, dtype=jnp.int32))

    @staticmethod
    def _normalize_rows(x: jnp.ndarray) -> jnp.ndarray:
        norm = jnp.linalg.norm(x, axis=1, keepdims=True)
        return x / jnp.maximum(norm, jnp.array(1e-12, dtype=x.dtype))

    def _update_penalty_from_acts(self, fake_acts: jnp.ndarray) -> None:
        if not self._real_acts:
            raise ValueError("Real features must be cached before updating fake features for MiFID.")

        real_acts = jnp.concatenate(self._real_acts, axis=0)
        real_norm = self._normalize_rows(real_acts)
        fake_norm = self._normalize_rows(fake_acts)

        cosine_sim = fake_norm @ real_norm.T
        cosine_dist = 1.0 - jnp.abs(cosine_sim)
        min_dist = jnp.min(cosine_dist, axis=1)
        self._penalty_sum[...] = self._penalty_sum[...] + jnp.sum(min_dist, dtype=self.metric_dtype)
        self._penalty_count[...] = self._penalty_count[...] + jnp.array(min_dist.shape[0], dtype=jnp.int32)

    def update(self, imgs: Float[ArrayLike, "batch h w c"], real: bool) -> None:
        acts = self._extract_activations(imgs)
        if real:
            self._real_acts.append(acts)
        else:
            self._update_fake_stats_from_acts(acts)
            self._update_penalty_from_acts(acts)

    def compute(self) -> Array:
        fid = super().compute()
        count = self._penalty_count[...]
        mean_distance = jnp.where(
            count > 0,
            self._penalty_sum[...] / count.astype(self.metric_dtype),
            jnp.array(1.0, dtype=self.metric_dtype),
        )
        distance = jnp.where(
            mean_distance < jnp.array(self.cosine_distance_eps, dtype=mean_distance.dtype),
            mean_distance,
            jnp.array(1.0, dtype=mean_distance.dtype),
        )
        mifid = jnp.where(
            fid > jnp.array(1e-8, dtype=self.metric_dtype),
            fid / (distance + jnp.array(1e-14, dtype=self.metric_dtype)),
            jnp.array(0.0, dtype=self.metric_dtype),
        )
        return mifid

    def reset(self) -> None:
        super().reset()
        self._penalty_sum[...] = jnp.array(0.0, dtype=self.metric_dtype)
        self._penalty_count[...] = jnp.array(0, dtype=jnp.int32)

    def get_penalty(self) -> Array:
        count = int(self._penalty_count[...])
        if count == 0:
            return 1.0
        mean_distance = self._penalty_sum[...] / jnp.array(count, dtype=self.metric_dtype)
        distance = jnp.where(
            mean_distance < jnp.array(self.cosine_distance_eps, dtype=mean_distance.dtype),
            mean_distance,
            jnp.array(1.0, dtype=mean_distance.dtype),
        )
        return distance
