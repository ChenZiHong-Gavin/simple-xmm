# model.py
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel
from dataclasses import dataclass


@dataclass
class ModalProjectorConfig:
    model_path: str
    projector_type: str = "linear"
    num_tokens: int = 64


class SimpleCrossModalModel(nn.Module):
    def __init__(
        self, llm: PreTrainedModel, modal_configs: Dict[str, ModalProjectorConfig]
    ):
        super().__init__()
        self.llm = llm
        self.modal_encoders = nn.ModuleDict()
        self.modal_projectors = nn.ModuleDict()
        self.modal_configs = modal_configs

        # 为每个模态加载编码器和投影器
        for modal_name, config in modal_configs.items():
            print(f"Loading {modal_name} encoder from {config.model_path}")
            encoder = AutoModel.from_pretrained(config.model_path)
            self.modal_encoders[modal_name] = encoder

            projector = self._create_projector(
                encoder.config.hidden_size,
                llm.config.hidden_size,
                config.projector_type,
            )
            self.modal_projectors[modal_name] = projector

    def _create_projector(self, input_dim: int, output_dim: int, proj_type: str):
        if proj_type == "linear":
            return nn.Linear(input_dim, output_dim)
        elif proj_type == "mlp":
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
            )
        raise ValueError(f"Unknown projector type: {proj_type}")

    def inject_modal_representations(
        self,
        hidden_states: torch.Tensor,
        modal_ids: Dict[str, torch.Tensor],
        modal_info: List[Dict],
        device: torch.device,
    ) -> torch.Tensor:
        """核心注入逻辑"""
        for info in modal_info:
            if info["start"] == -1:  # padding
                continue

            modal_type = info["type"]
            start_pos = info["start"]
            batch_idx = info["batch_idx"]

            # 编码+投影
            encoder = self.modal_encoders[modal_type]
            projector = self.modal_projectors[modal_type]

            modal_repr = encoder(modal_ids[modal_type][batch_idx : batch_idx + 1])
            modal_repr = projector(modal_repr.last_hidden_state)

            # 替换预留tokens
            num_tokens = self.modal_configs[modal_type].num_tokens
            k = min(num_tokens, modal_repr.size(1))
            hidden_states[batch_idx, start_pos + 1 : start_pos + 1 + k] = modal_repr[
                0, :k
            ]

        return hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        modal_ids: Optional[Dict[str, torch.Tensor]] = None,
        modal_info: Optional[List[List[Dict]]] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        if modal_ids is not None and modal_info is not None:
            for b in range(inputs_embeds.shape[0]):
                inputs_embeds = self.inject_modal_representations(
                    inputs_embeds, modal_ids, modal_info[b], input_ids.device
                )

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def save_modal_projectors(self, output_dir: str):
        import os

        os.makedirs(output_dir, exist_ok=True)
        for name, projector in self.modal_projectors.items():
            torch.save(projector.state_dict(), f"{output_dir}/{name}_projector.bin")
            print(f"Saved {name} projector")


def create_cross_modal_model(llm_path: str, modal_configs: Dict[str, dict]):
    from transformers import AutoModelForCausalLM

    llm = AutoModelForCausalLM.from_pretrained(
        llm_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )

    modal_obj_configs = {
        name: ModalProjectorConfig(**cfg) for name, cfg in modal_configs.items()
    }

    return SimpleCrossModalModel(llm, modal_obj_configs)
