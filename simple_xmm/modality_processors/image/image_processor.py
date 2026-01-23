from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch
from simple_xmm.modality_processors.base_processor import BaseModalProcessor


class ImageModalProcessor(BaseModalProcessor):
    def __init__(self, tag: str = "image", model_path: str = None, trust_remote_code: bool = False):
        """
        Args:
            tag: 标签名称，默认 'image'
            image_processor: 预训练模型的 Image Processor，
                        用于加载对应的预处理配置（均值、方差、尺寸）。
        """
        super().__init__(tag)
        self.image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)

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

    def pad(self, features: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        image_values = [f["image_values"] for f in features]

        if not image_values:
            return torch.empty(0), torch.empty(0)

        # 固定大小的特征，直接stack
        padded_features = torch.stack(image_values)
        # 图像通常不需要attention mask，返回全1
        attention_mask = torch.ones(len(image_values), dtype=torch.long)

        return padded_features, attention_mask

    def get_encoder(self):
        return AutoModel.from_pretrained(self.model_path)

    def encode(
        self,
        encoder: torch.nn.Module,
        projector: torch.nn.Module,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """图像编码：调用pixel_values参数，无需mask"""
        outputs = encoder(pixel_values=values)
        features = projector(outputs.last_hidden_state)
        # 图像通常没有padding，直接返回所有样本
        return [f for f in features]
    
    def get_hidden_size(
        self,
        encoder: torch.nn.Module
    ) -> List[torch.Tensor]:
        return encoder.config.text_config.hidden_size
