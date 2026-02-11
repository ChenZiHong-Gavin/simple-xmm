from typing import List, Optional
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from simple_xmm.modalities import MODALITY_ENCODERS
from simple_xmm.models.model_base import XMMModelBase


class XMMMlpProjectorModel(XMMModelBase):
    def __init__(self, llm: PreTrainedModel, modal_configs: dict):
        super().__init__(llm)
        self.modal_projectors = nn.ModuleDict()

        for name, kwargs in modal_configs.items():
            kwargs = kwargs.copy()
            model_type = kwargs.pop("model_type")
            cls = MODALITY_ENCODERS[name][model_type]
            encoder = cls(tag=name, **kwargs)
            self.modal_encoders[name] = encoder
            enc_dim = encoder.hidden_size
            self.modal_projectors[name] = nn.Sequential(
                nn.Linear(enc_dim, llm.config.hidden_size),
                nn.GELU(),
                nn.Linear(llm.config.hidden_size, llm.config.hidden_size),
            )

    def encode_modality(
        self,
        modal_type: str,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        encoder = self.modal_encoders[modal_type]
        projector = self.modal_projectors[modal_type]

        encoded = encoder.forward(values, attention_mask)  # (B, Seq, EncDim)
        projected = projector(encoded)  # (B, Seq, HiddenSize)

        return encoder.post_process(projected, values, attention_mask)
