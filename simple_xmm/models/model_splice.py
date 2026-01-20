from typing import Dict, List, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel
from torch.nn.utils.rnn import pad_sequence


@dataclass
class ModalProjectorConfig:
    model_path: str
    projector_type: str = "mlp"


class XMMModel(nn.Module):
    def __init__(
        self, llm: PreTrainedModel, modal_configs: Dict[str, ModalProjectorConfig]
    ):
        super().__init__()
        self.llm = llm
        self.modal_encoders = nn.ModuleDict()
        self.modal_projectors = nn.ModuleDict()
        self.modal_configs = modal_configs

        for modal_name, config in modal_configs.items():
            encoder = AutoModel.from_pretrained(config.model_path)
            self.modal_encoders[modal_name] = encoder

            enc_dim = getattr(
                encoder.config, "hidden_size", getattr(encoder.config, "d_model")
            )
            self.modal_projectors[modal_name] = nn.Sequential(
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
        """
        输入: (B, Input_Seq, ...) 和 (B, Input_Seq) 的 mask
        输出: List[Tensor]，每个 Tensor 为 (Valid_Output_Seq, Dim)，去除了 Padding
        """
        encoder = self.modal_encoders[modal_type]
        projector = self.modal_projectors[modal_type]

        if modal_type == "audio":
            outputs = encoder(input_features=values, attention_mask=attention_mask)
        elif modal_type == "protein":
            outputs = encoder(input_ids=values, attention_mask=attention_mask)
        elif modal_type == "image":
            outputs = encoder(pixel_values=values)
        else:
            raise ValueError(f"Unsupported modal type: {modal_type}")

        last_hidden_state = outputs.last_hidden_state
        # 投影到 LLM 维度
        features = projector(last_hidden_state)  # (B, Out_Seq, Dim)

        results = []
        batch_size = features.shape[0]

        # 计算 Input Mask 到 Output Mask 的映射
        if attention_mask is not None:
            input_lens = attention_mask.sum(dim=1)  # (B,)

            scale = features.shape[1] / values.shape[1]
            valid_out_lens = (input_lens * scale).long()
            # 修正边界，防止算出 0
            valid_out_lens = torch.clamp(valid_out_lens, min=1, max=features.shape[1])

            for i in range(batch_size):
                length = valid_out_lens[i]
                results.append(features[i, :length, :])
        else:
            # 如果没有 mask (如图片)，直接全取
            for i in range(batch_size):
                results.append(features[i])

        return results

    def prepare_multimodal_inputs(
        self, input_ids, labels, attention_mask, modal_info, modal_features
    ):
        new_inputs_embeds = []
        new_labels = []
        new_attention_masks = []

        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        modal_counters = {k: 0 for k in modal_features.keys()}

        for i in range(len(input_ids)):
            # 1. 确定当前样本的有效文本长度（去除 Padding）
            # 假设 tokenizer padding side 是 right，且 pad_token_id 对应的 mask 是 0
            valid_len = attention_mask[i].sum().item()

            cur_embeds = inputs_embeds[i, :valid_len]  # 只取有效部分
            cur_labels = labels[i, :valid_len]
            cur_mask = attention_mask[i, :valid_len]  # 全是 1

            cur_info = sorted(modal_info[i], key=lambda x: x["start"])

            parts_embeds = []
            parts_labels = []
            parts_masks = []

            cur_pos = 0

            for info in cur_info:
                m_type = info["type"]
                m_start = info["start"]

                # 过滤掉超出 valid_len 的异常 info (防止 dataset 处理出错)
                if m_start >= valid_len:
                    continue

                # 拼接之前的文本
                if m_start > cur_pos:
                    parts_embeds.append(cur_embeds[cur_pos:m_start])
                    parts_labels.append(cur_labels[cur_pos:m_start])
                    parts_masks.append(cur_mask[cur_pos:m_start])

                # 拼接模态特征
                if m_type in modal_features:
                    idx = modal_counters[m_type]
                    feature = modal_features[m_type][idx]
                    modal_counters[m_type] += 1

                    parts_embeds.append(feature)

                    # 标签设为 IGNORE_INDEX
                    feat_len = feature.shape[0]
                    parts_labels.append(
                        torch.full(
                            (feat_len,), -100, device=feature.device, dtype=torch.long
                        )
                    )
                    # Mask 设为 1
                    parts_masks.append(
                        torch.full(
                            (feat_len,), 1, device=feature.device, dtype=torch.long
                        )
                    )

                # 跳过原文本中的占位符 token (假设占位符长度为1)
                cur_pos = m_start + 1

            # 拼接剩余的有效文本
            if cur_pos < len(cur_embeds):
                parts_embeds.append(cur_embeds[cur_pos:])
                parts_labels.append(cur_labels[cur_pos:])
                parts_masks.append(cur_mask[cur_pos:])

            new_inputs_embeds.append(torch.cat(parts_embeds, dim=0))
            new_labels.append(torch.cat(parts_labels, dim=0))
            new_attention_masks.append(torch.cat(parts_masks, dim=0))

        # 重新 Padding 成 Batch
        batch_embeds = pad_sequence(
            new_inputs_embeds, batch_first=True, padding_value=0.0
        )
        batch_labels = pad_sequence(new_labels, batch_first=True, padding_value=-100)
        batch_masks = pad_sequence(
            new_attention_masks, batch_first=True, padding_value=0
        )

        return batch_embeds, batch_labels, batch_masks

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        image_values: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        protein_values: Optional[torch.Tensor] = None,
        protein_attention_mask: Optional[torch.Tensor] = None,
        modal_info: Optional[List] = None,
        **kwargs,
    ):
        modal_features = {}

        if image_values:
            modal_features["image"] = self.encode_modality("image", image_values)

        if audio_values:
            modal_features["audio"] = self.encode_modality(
                "audio", audio_values, attention_mask=audio_attention_mask
            )

        if protein_values:
            modal_features["protein"] = self.encode_modality(
                "protein", protein_values, attention_mask=protein_attention_mask
            )

        # splice
        if modal_features and modal_info:
            inputs_embeds, labels, attention_mask = self.prepare_multimodal_inputs(
                input_ids, labels, attention_mask, modal_info, modal_features
            )
        else:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # 防止梯度断流
        if (
            self.training
            and self.llm.is_gradient_checkpointing
            and inputs_embeds.requires_grad is False
        ):
            inputs_embeds.requires_grad_(True)

        return self.llm(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )
