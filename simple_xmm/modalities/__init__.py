from .image.clip_image_encoder import ImageModalEncoder
from .image.clip_image_processor import ImageModalProcessor
from .audio.clip_audio_encoder import AudioModalEncoder
from .audio.clip_audio_processor import AudioModalProcessor

MODALITY_PROCESSORS = {
    "image": {
        "clip": ImageModalProcessor,
    },
    "audio": {
        "whisper": AudioModalProcessor,
    },
}

MODALITY_ENCODERS = {
    "image": {
        "clip": ImageModalEncoder,
    },
    "audio": {
        "whisper": AudioModalEncoder,
    },
}
