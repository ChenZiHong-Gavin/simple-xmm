from typing import Dict, Any, TypedDict, List
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from simple_xmm.utils.registry import get_template_class
from simple_xmm.modalities.base_processor import BaseModalProcessor
from simple_xmm.datasets.base_dataset import XMMBaseDataset


IGNORE_INDEX = -100


class XMMSeq2SeqSample(TypedDict, total=True):
    input_ids: torch.Tensor  # size=(L,)
    labels: torch.Tensor  # size=(L,)
    modal_info: List[Dict[str, Any]]


class XMMSeq2SeqBatch(TypedDict, total=True):
    input_ids: torch.Tensor  # shape: (B, L)
    labels: torch.Tensor  # shape: (B, L)
    attention_mask: torch.Tensor  # shape: (B, L)
    modal_info: List[List[Dict[str, Any]]] = []


class XMMSeq2SeqDataset(XMMBaseDataset):
    def __init__(
        self,
        path: str,
        template: str,
        tokenizer: AutoTokenizer,
        processors: Dict[str, BaseModalProcessor],
        dataset_name: str = None,
        split: str = "train",
        data_files: str = None,
        max_samples: int = None,
        cutoff_len: int = None,
    ):
        """
        Args:
            path: 数据集id
            template: 模板字符串
            tokenizer: 主文本tokenizer
            dataset_name: 数据集名称（如"xmm"）
            split: 数据集划分（如"train"）
            data_files: 指定具体的数据文件（如"train.parquet"）
            max_samples: 读取样本数量
            cutoff_len: 最大序列长度（包含模态特征展开后的长度）
        """
        assert path, "Path must be provided"
        assert template, "Template must be provided"

        super().__init__(
            path=path,
            tokenizer=tokenizer,
            processors=processors,
            dataset_name=dataset_name,
            split=split,
            data_files=data_files,
            max_samples=max_samples,
            cutoff_len=cutoff_len,
        )

        self.formatter = get_template_class(template)

    def preprocess(self, raw_sample: dict[str, Any]) -> XMMSeq2SeqSample:
        messages, _ = self.formatter.format_supervised_sample(raw_sample)

        prompt_msgs = messages[:-1]
        response_msg = messages[-1]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        response_text = response_msg["content"] + self.tokenizer.eos_token

        # 多模态处理
        prompt_ids, modal_info = self._process_prompt(prompt_text)
        resp_ids = self.tokenizer.encode(response_text, add_special_tokens=False)

        full_ids = prompt_ids + resp_ids

        # 截断逻辑
        full_ids, modal_info = self._truncate(full_ids, modal_info)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()

        # 计算新的 prompt 长度用于 mask labels
        prompt_len = len(prompt_ids)
        if len(input_ids) < prompt_len:
            prompt_len = len(input_ids)

        labels[:prompt_len] = IGNORE_INDEX

        return {"input_ids": input_ids, "labels": labels, "modal_info": modal_info}

    def __getitem__(self, index: int) -> XMMSeq2SeqSample:
        """Get a tokenized data sample by index."""
        return self.preprocess(self.raw_data[index])


@dataclass
class XMMDataCollator:
    tokenizer: Any
    processors: Dict[str, BaseModalProcessor]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        input_ids_padded = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels_padded = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = input_ids_padded.ne(self.tokenizer.pad_token_id).long()

        batch = {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask,
            "modal_info": [f["modal_info"] for f in features],
        }

        modalities = {}
        for modal in self.processors.keys():
            modalities[modal] = []

        for sample in features:
            for modal in sample["modal_info"]:
                m_type = modal["type"]
                m_data = modal["content"]
                modalities[m_type].append(m_data)

        # 为模态数据生成 Mask + Padding
        for modal, modal_data in modalities.items():
            if modal_data:
                modal_features, modal_mask = self.processors[modal].pad(modal_data)
                batch[f"{modal}_values"] = modal_features
                batch[f"{modal}_attention_mask"] = modal_mask

        return batch
