from typing import Dict, Any, TypedDict, List
import re
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from datasets import load_dataset
from simple_xmm.utils.registry import get_template_class

IGNORE_INDEX = -100


class XMMSeq2SeqSample(TypedDict, total=True):
    input_ids: torch.Tensor  # size=(L,)
    labels: torch.Tensor  # size=(L,)
    modal_info: List[Dict[str, Any]]


class XMMSeq2SeqBatch(TypedDict, total=True):
    input_ids: torch.Tensor  # shape: (B, L)
    labels: torch.Tensor  # shape: (B, L)
    attention_mask: torch.Tensor  # shape: (B, L)
    modal_info: List[List[Dict[str, Any]]]


class XMMSeq2SeqDataset(Dataset):
    def __init__(
        self,
        path: str,
        template: str,
        tokenizer: AutoTokenizer,
        dataset_name: str = None,
        split: str = "train",
        data_files: str = None,
        max_samples: int = None,
        modals: List[str] = [],
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
            modals: 模态
        """
        super().__init__()
        assert path, "Path must be provided"
        assert template, "Template must be provided"
        self.path = path
        self.tokenizer = tokenizer
        ds_path = path
        ds_files = data_files
        if (
            ds_files is None
            and isinstance(path, str)
            and re.search(r"\.(json|parquet|csv)$", path)
        ):
            ds_files = path
            ds_path = path.rsplit(".", 1)[-1]
        self.raw_data = load_dataset(
            ds_path, name=dataset_name, split=split, data_files=ds_files
        )
        if max_samples:
            self.raw_data = self.raw_data.select(
                range(min(int(max_samples), len(self.raw_data)))
            )
        self.formatter = get_template_class(template)
        # <modal>...</modal>, re.DOTALL 允许匹配换行符
        self.regex_map = {
            k: re.compile(rf"<{k}>\s*(.*?)\s*</{k}>", re.DOTALL) for k in modals
        }
        self.modals = modals

    def _process_prompt(self, text: str):
        """分离模态标签并插入对应padding"""
        input_ids = []
        modal_info: List[Dict[str, Any]] = []

        matches = []
        for m_type, pattern in self.regex_map.items():
            for m in pattern.finditer(text):
                matches.append(
                    {
                        "type": m_type,
                        "start": m.start(),
                        "end": m.end(),
                        "content": m.group(1),
                    }
                )
        matches.sort(key=lambda x: x["start"])

        curr_pos = 0
        for m in matches:
            if m["start"] > curr_pos:
                text_part = text[curr_pos : m["start"]]
                input_ids.extend(
                    self.tokenizer.encode(text_part, add_special_tokens=False)
                )

            m_type = m["type"]

            pad_token_str = f"<|{m_type}_pad|>"
            pad_id = self.tokenizer.convert_tokens_to_ids(pad_token_str)
            if pad_id is None or pad_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Pad token <|{m_type}_pad|> not found in tokenizer.")

            start_idx = len(input_ids)
            input_ids.append(pad_id)
            modal_info.append(
                {"type": m_type, "start": start_idx, "content": m["content"]}
            )
            curr_pos = m["end"]

        if curr_pos < len(text):
            input_ids.extend(
                self.tokenizer.encode(text[curr_pos:], add_special_tokens=False)
            )

        return input_ids, modal_info

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
        input_ids = torch.tensor(prompt_ids + resp_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[: len(prompt_ids)] = IGNORE_INDEX

        return {"input_ids": input_ids, "labels": labels, "modal_info": modal_info}

    def __getitem__(self, index: int) -> XMMSeq2SeqSample:
        """Get a tokenized data sample by index."""
        return self.preprocess(self.raw_data[index])

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.raw_data)


class XMMSeq2SeqCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[XMMSeq2SeqSample]) -> XMMSeq2SeqBatch:
        input_ids = [x["input_ids"] for x in batch]
        labels = [x["labels"] for x in batch]
        modal_infos = [x.get("modal_info", []) for x in batch]

        # Pad 文本
        input_ids_padded = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels_padded = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = input_ids_padded.ne(self.pad_token_id)

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask,
            "modal_info": modal_infos,
        }


if __name__ == "__main__":
    import json
    import os
    import tempfile

    modals = ["protein"]
    tokenizer = AutoTokenizer.from_pretrained(
        r"D:\Project\work\shanghai ai lab\GraphGen\models", trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    special_tokens = {"additional_special_tokens": [f"<|{m}_pad|>" for m in modals]}
    tokenizer.add_special_tokens(special_tokens)

    sample = {
        "instruction": "<protein>ACDEFGHIKLMNPQRSTVWY</protein> 请分析该序列",
        "input": "",
        "output": "这是一个示例回复",
    }
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    try:
        json.dump([sample], tmp)
        tmp.flush()
        tmp.close()
        dataset = XMMSeq2SeqDataset(
            path=tmp.name,
            template="Alpaca",
            tokenizer=tokenizer,
            split="train",
            max_samples=1,
            modals=modals,
        )
        collator = XMMSeq2SeqCollator(tokenizer)
        batch = collator([dataset[0], dataset[0]])
        print("input_ids.shape:", batch["input_ids"].shape)
        print("labels.shape:", batch["labels"].shape)
        print("attention_mask.sum:", int(batch["attention_mask"].sum()))
        print("modal_info:", batch["modal_info"])
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
