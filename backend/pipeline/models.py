from dataclasses import dataclass

from backend.pipeline import gen_blur, gen_depth, gen_saliency


@dataclass
class Models:
    saliency_model: object
    depth_processor: object
    depth_model: object
    blur_model: object


def load_models() -> Models:
    saliency_model = gen_saliency.load_model()
    depth_processor, depth_model = gen_depth.load_model()
    blur_model = gen_blur.load_model()
    return Models(
        saliency_model=saliency_model,
        depth_processor=depth_processor,
        depth_model=depth_model,
        blur_model=blur_model,
    )
