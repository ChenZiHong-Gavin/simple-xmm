from typing import Dict, Any
import torch
from simple_xmm.datasets.sft_dataset import IGNORE_INDEX, XMMSeq2SeqSample
from simple_xmm.modalities.base_processor import BaseModalProcessor
from simple_xmm.datasets.base_dataset import XMMBaseDataset


class XMMPtDataset(XMMBaseDataset):
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

    def preprocess(self, raw_sample: dict[str, Any]) -> XMMSeq2SeqSample:
        # Try to find text content in common fields
        text = raw_sample.get("text", raw_sample.get("content", ""))

        # Append EOS token
        text += self.tokenizer.eos_token

        input_ids_list, modal_info = self._process_prompt(text)

        # Truncation logic
        input_ids_list, modal_info = self._truncate(input_ids_list, modal_info)

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
