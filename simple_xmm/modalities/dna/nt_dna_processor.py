from transformers import AutoTokenizer
from typing import Dict, Any, List, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence
from simple_xmm.modalities.base_processor import BaseModalProcessor


class DNAModalProcessor(BaseModalProcessor):
    def __init__(
        self,
        tag: str = "dna",
        model_path: str = None,
        trust_remote_code: bool = False,
        max_length: int = 1024,
    ):
        """
        Args:
            tag: 标签，默认 'dna'
            model_path: Nucleotide Transformer 模型路径，用于加载对应的 Tokenizer
            max_length: 最大序列长度，默认 1024
        """
        super().__init__(tag)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        # 某些Tokenizer可能没有 pad_token, 需要检查
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pad_value = self.tokenizer.pad_token_id
        self.max_length = max_length

    def process(self, content: str) -> Dict[str, Any]:
        """
        content: DNA序列，例如 "ATCG..."
        """
        sequence = content.strip()
        sequence = "".join(sequence.split())

        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "dna_values": inputs["input_ids"].squeeze(0),
        }

    def get_feature_length(self, features: Dict[str, Any]) -> int:
        return features["dna_values"].shape[0]

    def pad(self, features: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """离散Token的padding，可自动生成mask"""
        dna_values = [f["dna_values"] for f in features]

        if not dna_values:
            return torch.empty(0), torch.empty(0)

        padded_features = pad_sequence(
            dna_values, batch_first=True, padding_value=self.pad_value
        )
        # 根据padding值自动生成mask
        attention_mask = padded_features.ne(self.pad_value).long()

        return padded_features, attention_mask
