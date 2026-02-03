"""
Pytest comparing FlaxImageProcessor against HuggingFace AutoImageProcessor.

Tests that outputs match for facebook/dinov2-base preprocessing.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from fidax.preprocessor.image_preprocessor import FlaxImageProcessor, compute_resize_dims

# Fixtures


@pytest.fixture(scope="module")
def hf_processor():
    """Create HuggingFace BitImageProcessor with dinov2-base config."""
    from transformers import BitImageProcessor

    # DinoV2-base config (hardcoded to avoid network dependency)
    return BitImageProcessor(
        do_resize=True,
        size={"shortest_edge": 256},
        resample=3,  # BICUBIC
        do_center_crop=True,
        crop_size={"height": 224, "width": 224},
        do_rescale=True,
        rescale_factor=1 / 255.0,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        do_convert_rgb=True,
    )


@pytest.fixture(scope="module")
def flax_processor():
    """Create FlaxImageProcessor with dinov2 config."""
    return FlaxImageProcessor(
        shortest_edge=256,
        crop_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


# Helper functions


def create_solid_image(h: int, w: int, color: tuple[int, int, int]) -> np.ndarray:
    """Create solid color image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = color[0]
    img[..., 1] = color[1]
    img[..., 2] = color[2]
    return img


def create_gradient_image(h: int, w: int) -> np.ndarray:
    """Create horizontal gradient image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        gradient = np.linspace(0, 255, w, dtype=np.uint8)
        img[..., c] = np.tile(gradient, (h, 1))
    return img


def create_checkerboard(h: int, w: int, block_size: int = 32) -> np.ndarray:
    """Create checkerboard pattern."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                img[i, j] = [255, 255, 255]
    return img


def create_random_image(h: int, w: int, seed: int = 42) -> np.ndarray:
    """Create random noise image."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def process_with_hf(processor, image: np.ndarray) -> np.ndarray:
    """Process image with HuggingFace processor."""
    from PIL import Image

    pil_image = Image.fromarray(image)
    outputs = processor(pil_image, return_tensors="np")
    return outputs["pixel_values"][0]  # (3, 224, 224)


def process_with_flax(processor: FlaxImageProcessor, image: np.ndarray) -> np.ndarray:
    """Process image with Flax processor."""
    h, w = image.shape[:2]
    processor.set_input_size(h, w)
    outputs = processor(jnp.array(image, dtype=jnp.float32))
    return np.array(outputs["pixel_values"][0])  # (3, 224, 224)


# Tests


class TestSolidColors:
    """Test solid color images."""

    @pytest.mark.parametrize(
        "color,name",
        [
            ((0, 0, 0), "black"),
            ((255, 255, 255), "white"),
            ((255, 0, 0), "red"),
            ((0, 255, 0), "green"),
            ((0, 0, 255), "blue"),
            ((128, 128, 128), "gray"),
        ],
    )
    @pytest.mark.parametrize("size", [(256, 256), (400, 300), (300, 400)])
    def test_solid_color(self, hf_processor, flax_processor, color, name, size):
        """Solid colors should produce identical normalized values."""
        h, w = size
        image = create_solid_image(h, w, color)

        hf_out = process_with_hf(hf_processor, image)
        flax_out = process_with_flax(flax_processor, image)

        # For solid colors, all pixels should be identical
        # Compare center pixels to avoid any edge effects
        center = 112
        np.testing.assert_allclose(
            flax_out[:, center, center],
            hf_out[:, center, center],
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Mismatch for {name} at {size}",
        )


class TestGradients:
    """Test gradient images."""

    @pytest.mark.parametrize("size", [(256, 256), (400, 300), (512, 512)])
    def test_gradient(self, hf_processor, flax_processor, size):
        """Gradient images should match closely."""
        h, w = size
        image = create_gradient_image(h, w)

        hf_out = process_with_hf(hf_processor, image)
        flax_out = process_with_flax(flax_processor, image)

        # Allow slightly larger tolerance due to interpolation differences
        np.testing.assert_allclose(flax_out, hf_out, rtol=0.05, atol=0.05, err_msg=f"Gradient mismatch at {size}")


class TestPatterns:
    """Test patterned images."""

    @pytest.mark.parametrize("size", [(256, 256), (400, 300)])
    def test_checkerboard(self, hf_processor, flax_processor, size):
        """Checkerboard pattern should match.

        Note: High-frequency patterns (like checkerboards) are very sensitive to
        interpolation differences between JAX and PIL bicubic implementations.
        For non-square images requiring resize, we expect ~90% pixel match.
        """
        h, w = size
        image = create_checkerboard(h, w, block_size=32)

        hf_out = process_with_hf(hf_processor, image)
        flax_out = process_with_flax(flax_processor, image)

        # Check that majority of pixels match
        diff = np.abs(flax_out - hf_out)
        pct_within_tol = np.mean(diff < 0.1) * 100
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        # For square images: >99%, for non-square: >85% (due to interpolation)
        is_square = h == w
        min_pct = 99.0 if is_square else 85.0

        assert pct_within_tol > min_pct, f"Only {pct_within_tol:.2f}% pixels within tolerance"
        assert max_diff < 3.0, f"Max difference {max_diff:.4f} too large"
        assert mean_diff < 0.1, f"Mean difference {mean_diff:.4f} too large"


class TestRandomImages:
    """Test random noise images."""

    @pytest.mark.parametrize("size", [(256, 256), (400, 300), (300, 400), (512, 384)])
    @pytest.mark.parametrize("seed", [0, 42, 123])
    def test_random(self, hf_processor, flax_processor, size, seed):
        """Random images should match within tolerance.

        Note: JAX bicubic and PIL bicubic use slightly different algorithms.
        For square images (no resize needed), expect very high match rate.
        For non-square images, ~99.7% of pixels typically match within 0.1.
        """
        h, w = size
        image = create_random_image(h, w, seed=seed)

        hf_out = process_with_hf(hf_processor, image)
        flax_out = process_with_flax(flax_processor, image)

        # Check pixel match statistics
        diff = np.abs(flax_out - hf_out)
        pct_within_tol = np.mean(diff < 0.1) * 100
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        # For square images: >99.9%, for non-square: >99%
        is_square = h == w
        min_pct = 99.9 if is_square else 99.0

        assert pct_within_tol > min_pct, f"Only {pct_within_tol:.2f}% pixels within tolerance at {size}"
        assert max_diff < 1.0, f"Max diff {max_diff:.4f} at {size}"
        assert mean_diff < 0.02, f"Mean diff {mean_diff:.4f} at {size}"


class TestOutputProperties:
    """Test output properties match."""

    def test_output_shape(self, hf_processor, flax_processor):
        """Output shape should be (3, 224, 224)."""
        image = create_random_image(400, 300)

        hf_out = process_with_hf(hf_processor, image)
        flax_out = process_with_flax(flax_processor, image)

        assert hf_out.shape == (3, 224, 224)
        assert flax_out.shape == (3, 224, 224)

    def test_output_dtype(self, hf_processor, flax_processor):
        """Output should be float32."""
        image = create_random_image(256, 256)

        hf_out = process_with_hf(hf_processor, image)
        flax_out = process_with_flax(flax_processor, image)

        assert hf_out.dtype == np.float32
        assert flax_out.dtype == np.float32

    def test_normalization_range(self, hf_processor, flax_processor):
        """Normalized values should be in reasonable range."""
        image = create_random_image(256, 256)

        hf_out = process_with_hf(hf_processor, image)
        flax_out = process_with_flax(flax_processor, image)

        # ImageNet normalization typically produces values in [-3, 3]
        assert hf_out.min() > -5 and hf_out.max() < 5
        assert flax_out.min() > -5 and flax_out.max() < 5


class TestResizeDimensions:
    """Test that resize dimensions are computed correctly."""

    @pytest.mark.parametrize(
        "h,w,expected_h,expected_w",
        [
            (256, 256, 256, 256),  # Square
            (400, 300, 341, 256),  # Portrait -> scale by width
            (300, 400, 256, 341),  # Landscape -> scale by height
            (512, 512, 256, 256),  # Large square
            (1024, 768, 341, 256),  # Large portrait
        ],
    )
    def test_resize_dims(self, h, w, expected_h, expected_w):
        """Verify resize dimension computation."""
        new_h, new_w = compute_resize_dims(h, w, shortest_edge=256)
        assert new_h == expected_h, f"Height mismatch: {new_h} != {expected_h}"
        assert new_w == expected_w, f"Width mismatch: {new_w} != {expected_w}"


class TestBatchProcessing:
    """Test batch processing."""

    def test_batch_matches_single(self, flax_processor):
        """Batch processing should match single image processing."""
        images = [create_random_image(400, 300, seed=i) for i in range(4)]

        # Process individually
        single_outputs = []
        for img in images:
            out = process_with_flax(flax_processor, img)
            single_outputs.append(out)
        single_outputs = np.stack(single_outputs)

        # Process as batch
        flax_processor.set_input_size(400, 300)
        batch = jnp.array(np.stack(images), dtype=jnp.float32)
        batch_outputs = np.array(flax_processor(batch)["pixel_values"])

        np.testing.assert_allclose(
            batch_outputs,
            single_outputs,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Batch output doesn't match single image outputs",
        )


class TestEdgeCases:
    """Test edge cases."""

    def test_minimum_size(self, hf_processor, flax_processor):
        """Test with minimum viable size (224x224)."""
        image = create_random_image(224, 224)

        hf_out = process_with_hf(hf_processor, image)
        flax_out = process_with_flax(flax_processor, image)

        assert hf_out.shape == (3, 224, 224)
        assert flax_out.shape == (3, 224, 224)

    def test_exact_resize_size(self, hf_processor, flax_processor):
        """Test when input matches resize target (256x256)."""
        image = create_random_image(256, 256)

        hf_out = process_with_hf(hf_processor, image)
        flax_out = process_with_flax(flax_processor, image)

        np.testing.assert_allclose(
            flax_out,
            hf_out,
            rtol=0.05,
            atol=0.05,
        )

    def test_very_wide_image(self, hf_processor, flax_processor):
        """Test very wide aspect ratio."""
        image = create_random_image(256, 1024)

        hf_out = process_with_hf(hf_processor, image)
        flax_out = process_with_flax(flax_processor, image)

        assert hf_out.shape == (3, 224, 224)
        assert flax_out.shape == (3, 224, 224)

    def test_very_tall_image(self, hf_processor, flax_processor):
        """Test very tall aspect ratio."""
        image = create_random_image(1024, 256)

        hf_out = process_with_hf(hf_processor, image)
        flax_out = process_with_flax(flax_processor, image)

        assert hf_out.shape == (3, 224, 224)
        assert flax_out.shape == (3, 224, 224)


class TestNumericalStability:
    """Test numerical stability."""

    def test_all_zeros(self, hf_processor, flax_processor):
        """All zero image should normalize correctly."""
        image = np.zeros((256, 256, 3), dtype=np.uint8)

        hf_out = process_with_hf(hf_processor, image)
        flax_out = process_with_flax(flax_processor, image)

        # Should be -mean/std for each channel
        expected = np.array([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])

        np.testing.assert_allclose(flax_out[:, 112, 112], expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(hf_out[:, 112, 112], expected, rtol=1e-4, atol=1e-4)

    def test_all_max(self, hf_processor, flax_processor):
        """All 255 image should normalize correctly."""
        image = np.full((256, 256, 3), 255, dtype=np.uint8)

        hf_out = process_with_hf(hf_processor, image)
        flax_out = process_with_flax(flax_processor, image)

        # Should be (1-mean)/std for each channel
        expected = np.array([(1.0 - 0.485) / 0.229, (1.0 - 0.456) / 0.224, (1.0 - 0.406) / 0.225])

        np.testing.assert_allclose(flax_out[:, 112, 112], expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(hf_out[:, 112, 112], expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
