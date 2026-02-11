from typing import List, Optional
import torch
import torch.nn as nn
from transformers import PreTrainedModel, Blip2QFormerModel, Blip2QFormerConfig
from simple_xmm.modalities import MODALITY_ENCODERS
from simple_xmm.models.model_base import XMMModelBase


class XMMQFormerProjectorModel(XMMModelBase):
    def __init__(self, llm: PreTrainedModel, modal_configs: dict):
        super().__init__(llm)
        self.modal_qformers = nn.ModuleDict()
        self.modal_queries = nn.ParameterDict()
        self.modal_projectors = nn.ModuleDict()

        for name, kwargs in modal_configs.items():
            kwargs = kwargs.copy()
            model_type = kwargs.pop("model_type")

            # Extract Q-Former args
            num_query_tokens = kwargs.pop("num_query_tokens", 32)
            qformer_hidden_size = kwargs.pop("qformer_hidden_size", 768)
            num_hidden_layers = kwargs.pop("num_hidden_layers", 2)

            cls = MODALITY_ENCODERS[name][model_type]
            encoder = cls(tag=name, **kwargs)
            self.modal_encoders[name] = encoder
            enc_dim = encoder.hidden_size

            # Q-Former Config
            config = Blip2QFormerConfig(
                hidden_size=qformer_hidden_size,
                encoder_hidden_size=enc_dim,
                num_hidden_layers=num_hidden_layers,
                vocab_size=1,  # Not used for text generation, just encoder
            )
            # We use Blip2QFormerModel which is a BERT-like encoder
            self.modal_qformers[name] = Blip2QFormerModel(config)

            self.modal_queries[name] = nn.Parameter(
                torch.zeros(1, num_query_tokens, qformer_hidden_size)
            )
            self.modal_queries[name].data.normal_(
                mean=0.0, std=config.initializer_range
            )

            self.modal_projectors[name] = nn.Linear(
                qformer_hidden_size, llm.config.hidden_size
            )

    def encode_modality(
        self,
        modal_type: str,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        encoder = self.modal_encoders[modal_type]
        qformer = self.modal_qformers[modal_type]
        queries = self.modal_queries[modal_type]  # (1, num_query_tokens, hidden_size)
        projector = self.modal_projectors[modal_type]

        # Encoder forward
        # values: (B, Seq, ...)
        encoded = encoder.forward(values, attention_mask)  # (B, Seq, EncDim)

        # Q-Former forward
        # query_embeds: (B, num_query_tokens, hidden_size)
        batch_size = encoded.shape[0]
        query_embeds = queries.expand(batch_size, -1, -1)

        qformer_output = qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=encoded,
            encoder_attention_mask=None,  # We might need this if we have padding in encoded
        )

        query_output = (
            qformer_output.last_hidden_state
        )  # (B, num_query_tokens, hidden_size)

        projected = projector(query_output)  # (B, num_query_tokens, llm_hidden_size)

        return encoder.post_process(projected, values, attention_mask)
