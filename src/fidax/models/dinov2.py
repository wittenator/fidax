import jax
from flax import nnx
from jaxtyping import Array, Float
from transformers import FlaxDinov2Model


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
