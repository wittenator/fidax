from flax import nnx

from ..preprocessor.image_preprocessor import FlaxImageProcessor
from .dinov2 import DinoV2FeatureExtractor
from .inception import InceptionV3FeatureExtractor, InceptionV3Preprocessor


def get_fid_network(
    model_name: str, dtype: str = "float32", ckpt_dir: str | None = "data"
) -> tuple[nnx.Module, nnx.Module]:
    try:
        if model_name == "inception_v3":
            model = InceptionV3FeatureExtractor(dtype=dtype, ckpt_dir=ckpt_dir)
            image_processor = InceptionV3Preprocessor()
        else:
            image_processor = FlaxImageProcessor.from_pretrained(model_name)
            model = DinoV2FeatureExtractor(model_name=model_name, dtype=dtype, ckpt_dir=ckpt_dir)
    except ValueError as err:
        raise ValueError(f"Model {model} not supported for FID computation.") from err

    return image_processor, model
