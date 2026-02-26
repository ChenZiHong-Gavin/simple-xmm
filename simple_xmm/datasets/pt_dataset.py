from typing import Dict, Any, List
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import re
from simple_xmm.datasets.sft_dataset import IGNORE_INDEX, XMMSeq2SeqSample
from simple_xmm.modalities.base_processor import BaseModalProcessor


class XMMPtDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        processors: Dict[str, BaseModalProcessor],
        dataset_name: str = None,
        split: str = "train",
        data_files: str = None,
        max_samples: int = None,
        cutoff_len: int = None,
    ):
        """
        Args:
            path: Dataset path or ID
            tokenizer: Main text tokenizer
            processors: Modality processors
            dataset_name: Dataset name (e.g., "xmm")
            split: Dataset split (e.g., "train")
            data_files: Specific data files (e.g., "train.parquet")
            max_samples: Number of samples to load
            cutoff_len: Maximum sequence length
        """
        super().__init__()
        self.path = path
        self.cutoff_len = cutoff_len
        self.tokenizer = tokenizer
        self.processors = processors

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

        for proc in self.processors.values():
            if self.tokenizer.convert_tokens_to_ids(proc.pad_token) is None:
                raise ValueError(f"Pad token {proc.pad_token} not found in tokenizer.")

    def _process_prompt(self, text: str):
        """Separates modality tags and inserts corresponding padding."""
        input_ids = []
        modal_info: List[Dict[str, Any]] = []

        matches = []
        for proc in self.processors.values():
            for m in proc.pattern.finditer(text):
                matches.append(
                    {
                        "processor": proc,
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
            proc = m["processor"]
            processed_content = proc.process(m["content"])

            pad_id = self.tokenizer.convert_tokens_to_ids(proc.pad_token)

            modal_info.append(
                {
                    "type": proc.tag,
                    "start": len(input_ids),
                    "raw": m["content"],
                    "content": processed_content,
                }
            )

            # Insert placeholder
            input_ids.append(pad_id)
            curr_pos = m["end"]

        if curr_pos < len(text):
            input_ids.extend(
                self.tokenizer.encode(text[curr_pos:], add_special_tokens=False)
            )

        return input_ids, modal_info

    def preprocess(self, raw_sample: dict[str, Any]) -> XMMSeq2SeqSample:
        # Try to find text content in common fields
        text = raw_sample.get("text", raw_sample.get("content", ""))

        # Append EOS token
        text += self.tokenizer.eos_token

        input_ids_list, modal_info = self._process_prompt(text)

        # Truncation logic
        if self.cutoff_len:
            token_lengths = [1] * len(input_ids_list)
            for m in modal_info:
                idx = m["start"]
                if idx < len(token_lengths):
                    proc = self.processors[m["type"]]
                    token_lengths[idx] = proc.get_feature_length(m["content"])

            cur_len = 0
            trunc_idx = len(input_ids_list)
            for i, length in enumerate(token_lengths):
                if cur_len + length > self.cutoff_len:
                    trunc_idx = i
                    break
                cur_len += length

            input_ids_list = input_ids_list[:trunc_idx]
            modal_info = [m for m in modal_info if m["start"] < trunc_idx]

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        labels = input_ids.clone()

        # Mask modality tokens in labels so we don't compute loss on them
        for m in modal_info:
            idx = m["start"]
            if idx < len(labels):
                labels[idx] = IGNORE_INDEX

        return {"input_ids": input_ids, "labels": labels, "modal_info": modal_info}

    def __getitem__(self, index: int) -> XMMSeq2SeqSample:
        return self.preprocess(self.raw_data[index])

    def __len__(self) -> int:
        return len(self.raw_data)
