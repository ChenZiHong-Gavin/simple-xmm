from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import torch.nn as nn


class BaseModalEncoder(nn.Module, ABC):
    def __init__(self, tag: str):
        super().__init__()
        self.tag = tag

    @abstractmethod
    def forward(
        self,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            values: 输入特征
            attention_mask: 注意力掩码
        Returns:
            Features (batch_size, seq_len, hidden_size)
        """
        pass

    def post_process(
        self,
        features: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        将 batch 特征转换为样本列表，并根据 mask 去除 padding。
        默认实现：假设没有 padding 或不需要去除（如 Image）。
        """
        return [f for f in features]

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Return the hidden size of the encoder output."""
        pass
