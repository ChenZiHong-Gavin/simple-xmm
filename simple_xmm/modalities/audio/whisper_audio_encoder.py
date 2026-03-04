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
        kwargs = {}
        if "input_features" in self.encoder.forward.__code__.co_varnames:
            kwargs["input_features"] = values
        else:
            kwargs["input_values"] = values

        if attention_mask is not None:
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
            # values shape: (B, Seq, Freq)
            # attention_mask shape: (B, Seq)
            input_lens = attention_mask.sum(dim=1)

            # Calculate output lengths based on downsampling factor
            # Whisper encoder has 2 conv layers with stride 1 (stem) but let's check ratio dynamically
            in_seq_len = values.shape[1]
            out_seq_len = features.shape[1]

            if in_seq_len > 0:
                scale = out_seq_len / in_seq_len
                valid_out_lens = (input_lens * scale).long()
                # Clamp to ensure we don't exceed actual output length
                valid_out_lens = valid_out_lens.clamp(max=out_seq_len)

                return [features[i, : valid_out_lens[i]] for i in range(len(features))]

        return [f for f in features]

    @property
    def hidden_size(self) -> int:
        return self.encoder.config.hidden_size
