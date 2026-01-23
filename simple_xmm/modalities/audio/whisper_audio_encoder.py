from typing import List, Optional
import torch
from transformers import AutoModel
from simple_xmm.modalities.base_encoder import BaseModalEncoder


class AudioModalEncoder(BaseModalEncoder):
    def __init__(
        self,
        tag: str = "audio",
        model_path: str = None,
        trust_remote_code: bool = False,
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
        kwargs = {}
        if "input_features" in self.encoder.forward.__code__.co_varnames:
            kwargs["input_features"] = values
        else:
            kwargs["input_values"] = values

        if attention_mask:
            kwargs["attention_mask"] = attention_mask

        outputs = self.encoder(**kwargs)
        return outputs.last_hidden_state

    def post_process(
        self,
        features: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        if attention_mask is not None:
            input_lens = attention_mask.sum(dim=1)
            # values shape: (B, Seq) or (B, Seq, Freq)
            # input_lens based on dim 1
            scale = features.shape[1] / values.shape[1]
            valid_out_lens = (
                (input_lens * scale).long().clamp(min=1, max=features.shape[1])
            )
            return [features[i, : valid_out_lens[i]] for i in range(len(features))]

        return [f for f in features]

    @property
    def hidden_size(self) -> int:
        return self.encoder.config.hidden_size
