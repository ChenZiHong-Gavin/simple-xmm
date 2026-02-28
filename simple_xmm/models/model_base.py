from typing import List, Optional
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from torch.nn.utils.rnn import pad_sequence


class XMMModelBase(nn.Module):
    """
    XMM 模型的基类，实现了通用的多模态输入处理和前向传播逻辑。
    子类需要实现 `encode_modality` 方法以及初始化特定的 Encoder 和 Projector。
    """

    def __init__(self, llm: PreTrainedModel):
        super().__init__()
        self.llm = llm
        self.modal_encoders = nn.ModuleDict()
        # 子类可以根据需要添加 self.modal_projectors 或其他组件

    def encode_modality(
        self,
        modal_type: str,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        对单个模态进行编码和投影。
        子类必须实现此方法。
        """
        raise NotImplementedError

    def prepare_multimodal_inputs(
        self, input_ids, labels, attention_mask, modal_info, modal_features
    ):
        """
        将文本 token 和模态特征进行拼接，构造最终送入 LLM 的输入。
        """
        new_inputs_embeds = []
        new_labels = []
        new_attention_masks = []

        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        modal_counters = {k: 0 for k in modal_features.keys()}

        for i in range(len(input_ids)):
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

                # 跳过原文本中的占位符 token
                cur_pos = m_start + 1

            # 拼接剩余的有效文本
            if cur_pos < len(cur_embeds):
                parts_embeds.append(cur_embeds[cur_pos:])
                parts_labels.append(cur_labels[cur_pos:])
                parts_masks.append(cur_mask[cur_pos:])

            new_inputs_embeds.append(torch.cat(parts_embeds, dim=0))
            new_labels.append(torch.cat(parts_labels, dim=0))
            new_attention_masks.append(torch.cat(parts_masks, dim=0))

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
        modal_info: Optional[List] = None,
        **modal_inputs,
    ):
        modal_features = {}
        # 遍历已注册的 modal_encoders
        for modal_name in self.modal_encoders.keys():
            values_key = f"{modal_name}_values"
            mask_key = f"{modal_name}_attention_mask"

            if values_key in modal_inputs and modal_inputs[values_key] is not None:
                values = modal_inputs[values_key]
                modal_attention_mask = modal_inputs.get(mask_key)

                modal_features[modal_name] = self.encode_modality(
                    modal_name, values, modal_attention_mask
                )

        # splice
        if modal_features and modal_info:
            inputs_embeds, labels, attention_mask = self.prepare_multimodal_inputs(
                input_ids, labels, attention_mask, modal_info, modal_features
            )
        else:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        return self.llm(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            return_dict=True,
        )
