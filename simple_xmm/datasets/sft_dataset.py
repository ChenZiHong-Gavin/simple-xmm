from typing import Dict, Any, TypedDict, List
from dataclasses import dataclass
import re
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from datasets import load_dataset
from simple_xmm.utils.registry import get_template_class
from .processors.base import BaseModalProcessor


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


class XMMSeq2SeqDataset(Dataset):
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

        self.processors = processors
        for proc in self.processors.values():
            if self.tokenizer.convert_tokens_to_ids(proc.pad_token) is None:
                raise ValueError(f"Pad token {proc.pad_token} not found in tokenizer.")

    def _process_prompt(self, text: str):
        """分离模态标签并插入对应padding"""
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

            # 插入占位符
            input_ids.append(pad_id)
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


@dataclass
class XMMDataCollator:
    tokenizer: Any
    audio_pad_value: float = 0.0
    protein_pad_value: int = 1  # ESM pad_token_id
    image_pad_value: float = 0.0

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
        }

        # 收集各模态数据
        all_audio = []
        all_protein = []
        all_image = []

        for sample in features:
            for modal in sample["modal_info"]:
                m_type = modal["type"]
                m_data = modal["content"]

                if m_type == "audio":
                    all_audio.append(m_data["audio_values"])
                elif m_type == "protein":
                    all_protein.append(m_data["protein_values"])
                elif m_type == "image":
                    all_image.append(m_data["image_values"])

        # 模态数据处理 (生成 Mask + Padding)
        # --- Audio: 连续信号，必须手动生成 Mask ---
        if all_audio:
            # 假设输入可能是 (Seq,) 或 (Freq, Seq)
            # 我们统一转为 (Seq, ...) 进行 pad_sequence
            processed_audio = []
            audio_masks = []

            for wav in all_audio:
                # 确保是 (Seq, ...) 格式
                if wav.dim() == 2 and wav.shape[0] < wav.shape[1]:
                    # 假设是 (Freq, Seq)，转置为 (Seq, Freq)
                    wav = wav.transpose(0, 1)

                processed_audio.append(wav)

                # 【核心】生成 Mask: 在 Pad 之前，创建一个全 1 的 tensor
                # 长度等于当前样本的时间步长 (Seq 维度)
                seq_len = wav.shape[0]
                audio_masks.append(torch.ones(seq_len, dtype=torch.long))

            # shape: (B, Seq, Freq) 或 (B, Seq)
            batch["audio_values"] = pad_sequence(
                processed_audio, batch_first=True, padding_value=self.audio_pad_value
            )
            batch["audio_attention_mask"] = pad_sequence(
                audio_masks, batch_first=True, padding_value=0
            )

        # --- Protein: 离散 Token，可以根据 pad_value 生成 Mask ---
        if all_protein:
            padded_protein = pad_sequence(
                all_protein, batch_first=True, padding_value=self.protein_pad_value
            )
            batch["protein_values"] = padded_protein
            batch["protein_attention_mask"] = padded_protein.ne(
                self.protein_pad_value
            ).long()

        # --- Image: 固定大小，通常不需要 Mask ---
        if all_image:
            batch["image_values"] = torch.stack(all_image)

        return batch
