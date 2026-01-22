from .audio.audio_processor import AudioModalProcessor
from .image.image_processor import ImageModalProcessor
from .protein.protein_processor import ProteinModalProcessor

MODALITY_PROCESSORS = {
    "audio": AudioModalProcessor,
    "protein": ProteinModalProcessor,
    "image": ImageModalProcessor,
}
