import torch
import os
from typing import Optional
from simple_xmm.modalities.base_encoder import BaseModalEncoder


class RNAModalEncoder(BaseModalEncoder):
    def __init__(
        self,
        tag: str = "rna",
        model_name: str = "rna-fm",
        **kwargs,
    ):
        super().__init__(tag)
        
        # 判断是路径还是模型类型
        if os.path.isfile(model_name):
            model_path = model_name
            model_type = kwargs.get("model_type", "rna-fm")
        else:
            model_type = model_name
            model_path = kwargs.get("model_path", None)
        
        # 优先本地加载（集群环境必须提供本地路径）
        if model_path and os.path.exists(model_path):
            self.model, self.alphabet = self._load_from_local(model_path, model_type)
        else:
            # 尝试官方 API（需要网络，集群会失败）
            try:
                import fm
                if model_type == "rna-fm":
                    self.model, self.alphabet = fm.pretrained.rna_fm_t12()
                elif model_type == "mrna-fm":
                    self.model, self.alphabet = fm.pretrained.mrna_fm_t12()
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")
            except Exception as e:
                raise RuntimeError(
                    f"集群环境无法自动下载权重。请手动下载并指定本地路径：\n"
                    f"  1. 下载 RNA-FM: https://github.com/ml4bio/RNA-FM/releases \n"
                    f"  2. 配置中设置: model_name='/path/to/RNA-FM_pretrained.pth'\n"
                    f"原始错误: {e}"
                )
        
        self.model.eval()
        self._hidden_size = 640 if model_type == "rna-fm" else 1280

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

    def forward(
        self,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """与官方相同：input_ids -> representations[12]"""
        with torch.no_grad():
            results = self.model(values, repr_layers=[12])
        return results["representations"][12]

    @property
    def hidden_size(self) -> int:
        return self._hidden_size