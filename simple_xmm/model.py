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

        # 1. Encoder Forward (利用 Mask)
        if modal_type == "audio":
            # Whisper/Wav2Vec2 等通常支持 attention_mask
            outputs = encoder(input_features=values, attention_mask=attention_mask)
        elif modal_type == "protein":
            outputs = encoder(input_ids=values, attention_mask=attention_mask)
        else:  # Image
            outputs = encoder(pixel_values=values)

        last_hidden_state = outputs.last_hidden_state  # (B, Out_Seq, Dim)

        # 2. 投影到 LLM 维度
        features = projector(last_hidden_state)

        # 3. 动态切片 (关键步骤)
        results = []
        batch_size = features.shape[0]

        if attention_mask is not None:
            # 计算 Input Mask 到 Output Mask 的映射
            input_lens = attention_mask.sum(dim=1)  # (B,)

            # 尝试使用 Encoder 自带的长度计算器 (Wav2Vec2, Whisper 有此方法)
            if hasattr(encoder, "_get_feat_extract_output_lengths"):
                valid_out_lens = encoder._get_feat_extract_output_lengths(input_lens)
            else:
                # Fallback: 根据 shape 比例估算 (Output / Input)
                # 注意：这里用 Max length 计算比例
                scale = features.shape[1] / values.shape[1]
                valid_out_lens = (input_lens * scale).long()
                # 修正边界，防止算出 0
                valid_out_lens = torch.clamp(
                    valid_out_lens, min=1, max=features.shape[1]
                )

            for i in range(batch_size):
                length = valid_out_lens[i]
                # 只取有效长度
                results.append(features[i, :length, :])
        else:
            # 如果没有 mask (如图片)，直接全取
            for i in range(batch_size):
                results.append(features[i])

        return results

    def prepare_multimodal_inputs(
        self, input_ids, labels, attention_mask, modal_info, modal_features
    ):
        """拼接文本和动态长度的模态特征"""
        new_inputs_embeds = []
        new_labels = []
        new_attention_masks = []

        # 获取基础文本 Embedding
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # 模态计数器
        modal_counters = {k: 0 for k in modal_features.keys()}

        for i in range(len(input_ids)):
            cur_embeds = inputs_embeds[i]
            cur_labels = labels[i]
            cur_mask = attention_mask[i]
            cur_info = sorted(modal_info[i], key=lambda x: x["start"])

            parts_embeds = []
            parts_labels = []
            parts_masks = []

            cur_pos = 0

            for info in cur_info:
                m_type = info["type"]
                m_start = info["start"]

                # 1. 放入之前的文本
                if m_start > cur_pos:
                    parts_embeds.append(cur_embeds[cur_pos:m_start])
                    parts_labels.append(cur_labels[cur_pos:m_start])
                    parts_masks.append(cur_mask[cur_pos:m_start])

                # 2. 放入模态特征 (动态长度)
                if m_type in modal_features:
                    idx = modal_counters[m_type]
                    feature = modal_features[m_type][idx]  # 这是一个不定长的 Tensor
                    modal_counters[m_type] += 1

                    parts_embeds.append(feature)

                    # 扩展 Label (-100) 和 Mask (1)
                    feat_len = feature.shape[0]
                    parts_labels.append(
                        torch.full(
                            (feat_len,), -100, device=feature.device, dtype=torch.long
                        )
                    )
                    parts_masks.append(
                        torch.full(
                            (feat_len,), 1, device=feature.device, dtype=torch.long
                        )
                    )

                cur_pos = m_start + 1  # 跳过原来的占位符

            # 3. 放入剩余文本
            if cur_pos < len(cur_embeds):
                parts_embeds.append(cur_embeds[cur_pos:])
                parts_labels.append(cur_labels[cur_pos:])
                parts_masks.append(cur_mask[cur_pos:])

            # 拼接单个样本
            new_inputs_embeds.append(torch.cat(parts_embeds, dim=0))
            new_labels.append(torch.cat(parts_labels, dim=0))
            new_attention_masks.append(torch.cat(parts_masks, dim=0))

        # 再次 Padding 组成 Batch (因为插入后长度不齐了)
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
        **kwargs
    ):
        modal_features = {}

        # 编码模态特征
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

        # 混合特征
        if modal_features and modal_info:
            inputs_embeds, labels, attention_mask = self.prepare_multimodal_inputs(
                input_ids, labels, attention_mask, modal_info, modal_features
            )
        else:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # LLM Forward
        return self.llm(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
