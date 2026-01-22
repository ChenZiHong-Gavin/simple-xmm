from abc import ABC, abstractmethod
from typing import Dict, Any
import re


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

    def get_placeholder(self) -> str:
        return self.pad_token
