from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import re
import torch


class BaseModalProcessor(ABC):
    def __init__(self, tag: str):
        """
        tag: 模态标签，如 'image', 'protein'
        """
        self.tag = tag
        self.pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL)
        self.pad_token = f"<|{tag}_pad|>"
        self.start_token = f"<{tag}>"
        self.end_token = f"</{tag}>"

    @abstractmethod
    def process(self, content: str) -> Dict[str, Any]:
        """
        接收正则匹配到的内容（通常是文件路径或序列字符串），
        返回处理后的信息（pixel_values, 或者转换后的 input_ids）。
        """

    @abstractmethod
    def pad(self, features: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对特征进行填充，返回填充后的特征张量和注意力掩码。
        """

    def get_placeholder(self) -> str:
        return self.pad_token

    def get_special_tokens(self) -> List[str]:
        return [self.pad_token, self.start_token, self.end_token]
