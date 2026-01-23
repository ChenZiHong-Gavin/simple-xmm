from .image.clip_image_encoder import ImageModalEncoder
from .image.clip_image_processor import ImageModalProcessor

MODALITY_PROCESSORS = {
    "image": {
        "clip": ImageModalProcessor,
    }
}

MODALITY_ENCODERS = {
    "image": {
        "clip": ImageModalEncoder,
    }
}
