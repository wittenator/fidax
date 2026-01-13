from flax import nnx
from transformers import AutoImageProcessor

from .dinov2 import DinoV2FeatureExtractor
from .inception import InceptionV3FeatureExtractor


def get_fid_network(
    model_name: str, dtype: str = "float32", ckpt_dir: str | None = "data"
) -> tuple[nnx.Module, nnx.Module]:
    try:
        if model_name == "inception_v3":
            model = InceptionV3FeatureExtractor(dtype=dtype, ckpt_dir=ckpt_dir)

            def image_processor(self, images):
                return (images * 2.0) - 1.0
        else:
            image_processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=ckpt_dir)
            model = DinoV2FeatureExtractor(model_name=model_name, dtype=dtype, ckpt_dir=ckpt_dir)
    except ValueError as err:
        raise ValueError(f"Model {model} not supported for FID computation.") from err

    return image_processor, model
