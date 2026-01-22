from transformers import AutoTokenizer
from typing import Dict, Any, List, Tuple, Optional
import torch
from torch.nn.utils.rnn import pad_sequence
from simple_xmm.modality_processors.base_processor import BaseModalProcessor


class ProteinModalProcessor(BaseModalProcessor):
    def __init__(self, tag: str = "protein", protein_processor: AutoTokenizer = None):
        """
        Args:
            tag: 标签，默认 'protein'
            model_path: ESM 模型路径，用于加载对应的 Tokenizer
        """
        super().__init__(tag)
        self.tokenizer = protein_processor
        self.pad_value = self.tokenizer.pad_token_id

    def process(self, content: str) -> Dict[str, Any]:
        """
        content: 蛋白质氨基酸序列，例如 "MALWMRLLPLLALLALWGPDPAAAFVN..."
        这里默认 content 就是序列字符串。
        """
        sequence = content.strip()
        sequence = "".join(sequence.split())

        # ESM tokenizer 会处理字符切分，并添加 <cls> (开头) 和 <eos> (结尾)
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,  # 自动截断超过 max_length 的序列
            max_length=1024,
        )  # shape: (1, seq_len)

        return {
            "protein_values": inputs["input_ids"].squeeze(0),
        }

    def pad(self, features: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """离散Token的padding，可自动生成mask"""
        protein_values = [f["protein_values"] for f in features]

        if not protein_values:
            return torch.empty(0), torch.empty(0)

        padded_features = pad_sequence(
            protein_values, batch_first=True, padding_value=self.pad_value
        )
        # 根据padding值自动生成mask
        attention_mask = padded_features.ne(self.pad_value).long()

        return padded_features, attention_mask

    def encode(
        self,
        encoder: torch.nn.Module,
        projector: torch.nn.Module,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """蛋白质编码：调用input_ids参数"""
        outputs = encoder(input_ids=values, attention_mask=attention_mask)
        features = projector(outputs.last_hidden_state)

        # 对于离散token，可以直接用attention_mask裁剪
        if attention_mask is not None:
            return [
                features[i, : attention_mask[i].sum()] for i in range(len(features))
            ]

        return [f for f in features]
