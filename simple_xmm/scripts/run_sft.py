import logging
from typing import Dict
import torch
from omegaconf import OmegaConf

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)
from torch.utils.data import random_split
from simple_xmm.datasets.sft_dataset import XMMSeq2SeqDataset, XMMDataCollator
from simple_xmm.models.model_splice import XMMSpliceModel
from simple_xmm.modalities import MODALITY_PROCESSORS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_processors(modal_configs: Dict):
    processors = {}

    for name, kwargs in modal_configs.items():
        if name not in MODALITY_PROCESSORS:
            logger.warning(
                "Modality %s not implemented in processor map, skipping.", name
            )
            continue
        kwargs = kwargs.copy()
        model_type = kwargs.pop("model_type")
        cls = MODALITY_PROCESSORS[name][model_type]
        processors[name] = cls(tag=name, **kwargs)

    return processors


def set_special_tokens(tokenizer, processors):
    special_tokens = []
    for p in processors.values():
        special_tokens += p.get_special_tokens()
    tokenizer.add_tokens(special_tokens, special_tokens=True)


def build_datasets(data_config: Dict, tokenizer, processors):
    """
    Builds training and validation datasets based on the configuration.
    """
    logger.info(f"Loading data from {data_config['path']}...")

    full_dataset = XMMSeq2SeqDataset(
        path=data_config["path"],
        template=data_config["template"],
        tokenizer=tokenizer,
        processors=processors,
        max_samples=data_config.get("max_samples", None),
        # cutoff_len=data_config.get("cutoff_len", 2048),
    )

    val_size = data_config.get("val_size", 0.0)

    if val_size > 0:
        dataset_len = len(full_dataset)
        val_len = int(dataset_len * val_size)
        train_len = dataset_len - val_len

        generator = torch.Generator()
        train_dataset, eval_dataset = random_split(
            full_dataset, [train_len, val_len], generator=generator
        )
        logger.info(
            f"Data split: {train_len} training samples, {val_len} validation samples."
        )
    else:
        train_dataset = full_dataset
        eval_dataset = None
        logger.info(
            f"No validation split defined. Using all {len(full_dataset)} samples for training."
        )

    return train_dataset, eval_dataset


def run_sft(config_path):
    cfg = OmegaConf.load(config_path)

    set_seed(42)

    # Text Tokenizer
    llm_path = cfg["model"]["llm_name_or_path"]
    trust_remote_code = cfg["model"]["trust_remote_code"]
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path, trust_remote_code=trust_remote_code
    )

    modal_configs = cfg["model"]["modal_configs"] if "modal_configs" in cfg["model"] else {}

    # Processors
    logger.info("Setting up processors...")
    processors = build_processors(modal_configs)

    # Special Tokens: pad_token, start_token, end_token
    set_special_tokens(tokenizer, processors)

    # Dataset & Collator
    logger.info("Loading dataset...")
    data_config = cfg["data"]
    train_dataset, eval_dataset = build_datasets(data_config, tokenizer, processors)

    collator = XMMDataCollator(
        tokenizer=tokenizer,
        processors=processors,
    )

    # Build Model
    logger.info("Initializing XMM Model...")
    llm = AutoModelForCausalLM.from_pretrained(
        llm_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16 if cfg["train"]["bf16"] else torch.float16,
    )
    # Resize embedding
    llm.resize_token_embeddings(len(tokenizer))

    model = XMMSpliceModel(llm=llm, modal_configs=modal_configs)

    # 打印可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")

    training_args_dict = {**cfg["train"], **cfg["output"]}
    training_args = TrainingArguments(**training_args_dict, save_safetensors=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model()
