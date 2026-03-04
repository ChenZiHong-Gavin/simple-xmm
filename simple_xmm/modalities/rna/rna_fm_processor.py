import os
import torch
from typing import Dict, Any, List, Tuple
from simple_xmm.modalities.base_processor import BaseModalProcessor


class RNAModalProcessor(BaseModalProcessor):
    def __init__(
        self,
        tag: str = "rna",
        model_name: str = "rna-fm",
        max_length: int = 1024,
        **kwargs
    ):
        super().__init__(tag)
        self.max_length = max_length

        if os.path.isfile(model_name):
            model_path = model_name
            model_type = kwargs.get("model_type", "rna-fm")
        else:
            model_type = model_name
            model_path = kwargs.get("model_path", None)

        _, self.alphabet = self._load_from_local(model_path, model_type)

        # 使用官方的 batch_converter（正确处理 <cls> 和 <eos>）
        self.batch_converter = self.alphabet.get_batch_converter()
        self.pad_idx = self.alphabet.padding_idx
        print(f"RNA-FM pad_idx: {self.pad_idx}, vocab size: {len(self.alphabet)}")

    def _load_from_local(self, model_path: str, model_type: str):
        """使用 fm.pretrained 加载本地模型（与你参考代码一致）"""
        import fm
        from torch.serialization import add_safe_globals
        import argparse
        
        # 关键：注册 argparse.Namespace 为安全类型，允许 PyTorch 2.6+ 加载
        add_safe_globals([argparse.Namespace])
        
        if model_type == "rna-fm":
            model, alphabet = fm.pretrained.rna_fm_t12(model_path)
        elif model_type == "mrna-fm":
            model, alphabet = fm.pretrained.mrna_fm_t12(model_path)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        return model, alphabet

    def process(self, content: str) -> Dict[str, Any]:
        """使用 batch_converter 编码（与官方推荐方式一致）"""
        # 清洗序列
        seq = content.strip().upper().replace("T", "U")
        seq = "".join(c for c in seq if c in "AUCG")
        seq = "".join(seq.split())[:self.max_length]
        
        # 使用官方 batch_converter: [(name, seq)] -> labels, strs, tokens
        _, _, tokens = self.batch_converter([("seq", seq)])
        
        # tokens 是 [1, seq_len] 的 LongTensor，squeeze 成 [seq_len]
        return {"rna_values": tokens.squeeze(0)}
    
    def get_feature_length(self, features: Dict[str, Any]) -> int:
        return features["rna_values"].shape[0]
    
    def pad(self, features: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用 RNA-FM 的 pad_idx 进行 padding"""
        values = [f["rna_values"] for f in features]
        
        padded = torch.nn.utils.rnn.pad_sequence(
            values, batch_first=True, padding_value=self.pad_idx
        )
        attention_mask = padded.ne(self.pad_idx).long()
        
        return padded, attention_mask