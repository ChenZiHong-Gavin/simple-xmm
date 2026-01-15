from PIL import Image
from typing import Dict, Any
from transformers import AutoImageProcessor
from .base import BaseModalProcessor


class ImageModalProcessor(BaseModalProcessor):
    def __init__(self, tag: str = "image", image_processor: AutoImageProcessor = None):
        """
        Args:
            tag: 标签名称，默认 'image'
            image_processor: 预训练模型的 Image Processor，
                        用于加载对应的预处理配置（均值、方差、尺寸）。
        """
        super().__init__(tag)
        self.image_processor = image_processor

    def process(self, content: str) -> Dict[str, Any]:
        """
        content: 正则提取出的图片路径，例如 "/data/coco/train2017/000000.jpg"
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
