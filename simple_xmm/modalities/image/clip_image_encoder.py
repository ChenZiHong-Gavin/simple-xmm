from typing import Optional, List
from transformers import CLIPVisionModel
import torch
from simple_xmm.modalities.base_encoder import BaseModalEncoder


class ImageModalEncoder(BaseModalEncoder):
    def __init__(
        self,
        tag: str = "image",
        model_path: str = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        super().__init__(tag)
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.image_encoder = CLIPVisionModel.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code
        )

    def forward(
        self, values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        outputs = self.image_encoder(pixel_values=values)
        features = outputs.last_hidden_state
        return features

    def post_process(
        self,
        features: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        # Image features are usually fixed length (patches + cls)
        # If attention_mask is provided (which is all 1s usually), we can use it
        if attention_mask is not None:
            return [
                features[i, : int(attention_mask[i].sum().item())]
                for i in range(len(features))
            ]
        return [f for f in features]

    @property
    def hidden_size(self) -> int:
        return self.image_encoder.config.hidden_size
