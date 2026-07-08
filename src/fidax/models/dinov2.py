import math

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from transformers import FlaxDinov2Model
from transformers.models.dinov2 import modeling_flax_dinov2


def _interpolate_pos_encoding_like_torch(self, config, hidden_states, height, width, position_embeddings):
    """Replacement for FlaxDinov2Embeddings.interpolate_pos_encoding that matches the torch model.

    The stock Flax implementation interpolates the position embeddings with a corner-aligned
    scale-and-translate (the original DINOv2 repo's ``+ 0.1`` trick), while the torch
    implementation in transformers uses half-pixel bicubic interpolation. At non-native input
    resolutions (anything but 518px) the two produce visibly different embeddings, so FlaxDinov2
    features diverge from the canonical torch DinoV2 by ~5-10% per feature. This replacement
    replicates the torch resampling exactly.
    """
    num_patches = hidden_states.shape[1] - 1
    num_positions = position_embeddings.shape[1] - 1
    if num_patches == num_positions and height == width:
        return position_embeddings

    class_pos_embed = position_embeddings[:, :1]
    patch_pos_embed = position_embeddings[:, 1:]
    dim = hidden_states.shape[-1]

    new_height = height // config.patch_size
    new_width = width // config.patch_size
    sqrt_num_positions = int(math.sqrt(num_positions))

    patch_pos_embed = patch_pos_embed.reshape((1, sqrt_num_positions, sqrt_num_positions, dim))
    target_dtype = patch_pos_embed.dtype
    # CUBIC_PYTORCH (jax >= 0.10) replicates torch's a=-0.75 bicubic with half-pixel centers;
    # interpolate in float32 exactly like the torch implementation does internally
    patch_pos_embed = jax.image.resize(
        patch_pos_embed.astype(jnp.float32),
        shape=(1, new_height, new_width, dim),
        method=jax.image.ResizeMethod.CUBIC_PYTORCH,
        antialias=False,
    ).astype(target_dtype)
    patch_pos_embed = patch_pos_embed.reshape((1, -1, dim))

    return jnp.concatenate((class_pos_embed, patch_pos_embed), axis=1)


# Align FlaxDinov2Model with the torch reference implementation (see docstring above)
modeling_flax_dinov2.FlaxDinov2Embeddings.interpolate_pos_encoding = _interpolate_pos_encoding_like_torch


class DinoV2FeatureExtractor(nnx.Module):
    def __init__(self, model_name: str, dtype: str = "float32", ckpt_dir: str | None = "data"):
        super().__init__()
        self.model = FlaxDinov2Model.from_pretrained(model_name, dtype=dtype, cache_dir=ckpt_dir)

    @jax.jit
    def __call__(self, pixel_values: Float[Array, "batch h w c"]) -> Float[Array, "batch d"]:
        """
        Forward pass through the DinoV2 model to extract features.
        Args:
            images: Input images of shape [B, H, W, C].
        Returns:
            Features of shape [B, D].
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0, :]  # Use the CLS token representation
