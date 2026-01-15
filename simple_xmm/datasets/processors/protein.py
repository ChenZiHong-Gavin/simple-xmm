from transformers import AutoTokenizer
from typing import Dict, Any
from .base import BaseModalProcessor


class ProteinModalProcessor(BaseModalProcessor):
    def __init__(self, tag: str = "protein", protein_processor: AutoTokenizer = None):
        """
        Args:
            tag: 标签，默认 'protein'
            model_path: ESM 模型路径，用于加载对应的 Tokenizer
        """
        super().__init__(tag)
        self.tokenizer = protein_processor

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
