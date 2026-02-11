from typing import List, Optional
from transformers import AutoModel
import torch
from simple_xmm.modalities.base_encoder import BaseModalEncoder


class RNAModalEncoder(BaseModalEncoder):
    def __init__(
        self,
        tag: str = "rna",
        model_path: str = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        super().__init__(tag)

        self.encoder = AutoModel.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

    def forward(
        self,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """RNA编码：调用input_ids参数"""
        outputs = self.encoder(input_ids=values, attention_mask=attention_mask)
        # return last_hidden_state
        return outputs.last_hidden_state

    def post_process(
        self,
        features: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        if attention_mask is not None:
            return [
                features[i, : attention_mask[i].sum()] for i in range(len(features))
            ]
        return [f for f in features]

    @property
    def hidden_size(self) -> int:
        return self.encoder.config.hidden_size
