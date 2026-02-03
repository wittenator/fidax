"""
JAX Image Processor for DinoV2 and similar vision models.

Assumes all images have the same input size (known at JIT compile time).
This allows using jax.image.resize directly.
"""

import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Union

import jax
import jax.numpy as jnp
from jax import lax


def compute_resize_dims(h: int, w: int, shortest_edge: int) -> tuple[int, int]:
    """Compute output dimensions for shortest-edge resize."""
    if h < w:
        new_h = shortest_edge
        new_w = int(round(w * shortest_edge / h))
    else:
        new_w = shortest_edge
        new_h = int(round(h * shortest_edge / w))
    return new_h, new_w


def preprocess(
    image: jnp.ndarray,
    resize_height: int,
    resize_width: int,
    crop_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> jnp.ndarray:
    """
    Preprocess a single image (jittable).

    Args:
        image: (H, W, 3) array, values in [0, 255].
        resize_height: Height after resize (precomputed from input size).
        resize_width: Width after resize (precomputed from input size).
        crop_size: Final square crop size.
        mean: Normalization mean (R, G, B).
        std: Normalization std (R, G, B).

    Returns:
        (3, crop_size, crop_size) normalized array.
    """
    image = image.astype(jnp.float32)

    # Resize (bicubic)
    image = jax.image.resize(image, (resize_height, resize_width, 3), method="bicubic")

    # Center crop
    top = (resize_height - crop_size) // 2
    left = (resize_width - crop_size) // 2
    image = lax.dynamic_slice(image, (top, left, 0), (crop_size, crop_size, 3))

    # Rescale [0, 255] -> [0, 1]
    image = image / 255.0

    # Normalize
    mean = jnp.array(mean, dtype=jnp.float32)
    std = jnp.array(std, dtype=jnp.float32)
    image = (image - mean) / std

    # HWC -> CHW
    return jnp.transpose(image, (2, 0, 1))


@dataclass
class FlaxImageProcessor:
    """
    Jittable image processor for DinoV2 and similar models.

    Assumes all input images have the same dimensions.
    Call `set_input_size(h, w)` before processing to configure.
    """

    shortest_edge: int = 256
    crop_size: int = 224
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

    # Computed from input size (set via set_input_size)
    _resize_height: int = field(default=256, repr=False)
    _resize_width: int = field(default=256, repr=False)
    _preprocess_fn: callable = field(default=None, repr=False)
    _batch_fn: callable = field(default=None, repr=False)

    def __post_init__(self):
        if isinstance(self.mean, list):
            object.__setattr__(self, "mean", tuple(self.mean))
        if isinstance(self.std, list):
            object.__setattr__(self, "std", tuple(self.std))

    def set_input_size(self, height: int, width: int) -> "FlaxImageProcessor":
        """
        Configure processor for a specific input image size.

        Must be called before processing. Creates jitted functions.

        Args:
            height: Input image height.
            width: Input image width.

        Returns:
            self (for chaining).
        """
        new_h, new_w = compute_resize_dims(height, width, self.shortest_edge)
        object.__setattr__(self, "_resize_height", new_h)
        object.__setattr__(self, "_resize_width", new_w)

        # Create jitted preprocessing functions
        fn = partial(
            preprocess,
            resize_height=new_h,
            resize_width=new_w,
            crop_size=self.crop_size,
            mean=self.mean,
            std=self.std,
        )
        object.__setattr__(self, "_preprocess_fn", jax.jit(fn))
        object.__setattr__(self, "_batch_fn", jax.jit(jax.vmap(fn)))

        return self

    def __call__(self, images: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """
        Preprocess images.

        Args:
            images: (H, W, 3) or (B, H, W, 3) array, values in [0, 255].

        Returns:
            {"pixel_values": (B, 3, crop_size, crop_size) array}
        """
        if self._preprocess_fn is None:
            # Auto-configure from first image
            if images.ndim == 3:
                h, w = images.shape[:2]
            else:
                h, w = images.shape[1:3]
            self.set_input_size(h, w)

        if images.ndim == 3:
            pixel_values = self._preprocess_fn(images)[None]
        else:
            pixel_values = self._batch_fn(images)

        return {"pixel_values": pixel_values}

    @classmethod
    def from_dict(cls, config: dict, **kwargs) -> "FlaxImageProcessor":
        """Create from HuggingFace-style config dict."""
        size = config.get("size", {})
        crop = config.get("crop_size", {})

        return cls(
            shortest_edge=size.get("shortest_edge", 256),
            crop_size=crop.get("height", 224),
            mean=tuple(config.get("image_mean", [0.485, 0.456, 0.406])),
            std=tuple(config.get("image_std", [0.229, 0.224, 0.225])),
        )

    @classmethod
    def from_pretrained(cls, path: Union[str, Path], **kwargs) -> "FlaxImageProcessor":
        """Load from local path or HuggingFace Hub."""
        path = Path(path)
        config_file = path / "preprocessor_config.json"

        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
        else:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(str(path), "preprocessor_config.json")
            with open(downloaded) as f:
                config = json.load(f)

        return cls.from_dict(config, **kwargs)
