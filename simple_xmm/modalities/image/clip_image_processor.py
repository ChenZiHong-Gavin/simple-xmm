from typing import Dict, Any, List, Tuple
from PIL import Image
from transformers import AutoImageProcessor, AutoConfig
import torch
from simple_xmm.modalities.base_processor import BaseModalProcessor


class ImageModalProcessor(BaseModalProcessor):
    def __init__(
        self,
        tag: str = "image",
        model_path: str = None,
        trust_remote_code: bool = False,
    ):
        super().__init__(tag)
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

        # Determine feature length
        vision_config = getattr(self.config, "vision_config", self.config)
        patch_size = getattr(vision_config, "patch_size", 32)

        # Get image size from processor
        if hasattr(self.image_processor, "crop_size"):
            size = self.image_processor.crop_size
        elif hasattr(self.image_processor, "size"):
            size = self.image_processor.size
        else:
            size = getattr(vision_config, "image_size", 224)

        if isinstance(size, dict):
            image_size = size.get("height", size.get("shortest_edge", 224))
        else:
            image_size = size

        grid_size = image_size // patch_size
        self.feature_length = grid_size * grid_size + 1

    def process(self, content: str) -> Dict[str, Any]:
        """
        content: 图片路径，例如 "/data/coco/train2017/000000.jpg"
        """
        image_path = content.strip()

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image path: {image_path}, error: {e}")
            return {"pixel_values": None}

        inputs = self.image_processor(images=image, return_tensors="pt")

        pixel_values = inputs["pixel_values"].squeeze(0)  # (C, H, W)

        return {"image_values": pixel_values}

    def get_feature_length(self, features: Dict[str, Any]) -> int:
        return self.feature_length

    def pad(self, features: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        image_values = [f["image_values"] for f in features]

        if not image_values:
            return torch.empty(0), torch.empty(0)

        # 固定大小的特征，直接stack
        padded_features = torch.stack(image_values)
        # 图像通常不需要attention mask，返回全1
        attention_mask = torch.ones(len(image_values), dtype=torch.long)

        return padded_features, attention_mask
