import logging
from typing import Dict
import torch
from omegaconf import OmegaConf

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    AutoFeatureExtractor,
    Trainer,
    set_seed,
)

from simple_xmm.datasets.sft_dataset import XMMSeq2SeqDataset, XMMDataCollator
from simple_xmm.datasets.processors import (
    AudioModalProcessor,
    ProteinModalProcessor,
    ImageModalProcessor,
)
from simple_xmm.models.model_splice import XMMModel, ModalProjectorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_processors(modal_configs: Dict, processor_args: Dict):
    processors = {}
    PROCESSOR_MAP = {
        "audio": (AudioModalProcessor, "audio"),
        "protein": (ProteinModalProcessor, "protein"),
        "image": (ImageModalProcessor, "image"),
    }

    for name, _ in modal_configs.items():
        if name not in PROCESSOR_MAP:
            logger.warning(
                "Modality %s not implemented in processor map, skipping.", name
            )
            continue

        cls, default_tag = PROCESSOR_MAP[name]
        # 获取该模态的特定参数，如 max_length
        kwargs = processor_args.get(name, {})

        # 实例化: 这里需要根据你的 Processor __init__ 签名适配
        # 假设 Processor __init__ 接收 tag 和其他 kwargs
        # 如果需要传入 encoder/tokenizer，可以在这里初始化并传入
        if name == "protein":
            pt_tok = AutoTokenizer.from_pretrained(modal_configs[name]["model_path"])
            processors[name] = cls(tag=default_tag, protein_processor=pt_tok, **kwargs)
        elif name == "audio":
            au_feat = AutoFeatureExtractor.from_pretrained(
                modal_configs[name]["model_path"]
            )
            processors[name] = cls(tag=default_tag, audio_processor=au_feat, **kwargs)
        else:
            processors[name] = cls(tag=default_tag, **kwargs)

    return processors


def run_sft(
    config_path,
    output_dir,
    local_rank,
):
    cfg = OmegaConf.load(config_path)

    set_seed(42)

    # Text Tokenizer
    llm_path = cfg["model"]["llm_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)

    modal_configs = cfg["model"]["modal_configs"] or {}

    # Processors
    logger.info("Building processors...")
    processors = build_processors(
        modal_configs, cfg["data"].get("processor_args", {})
    )

    # Special Tokens: pad_token, start_token, end_token
    special_tokens = [p.pad_token for p in processors.values()]
    special_tokens += [p.start_token for p in processors.values()]
    special_tokens += [p.end_token for p in processors.values()]
    tokenizer.add_tokens(special_tokens, special_tokens=True)

    # --- 3. Dataset & Collator ---
    logger.info("Loading dataset...")
    train_dataset = XMMSeq2SeqDataset(
        path=cfg["data"]["path"],
        template=cfg["data"]["template"],
        tokenizer=tokenizer,
        processors=processors,
        max_samples=cfg["data"].get("max_samples", None),
    )

    collator = XMMDataCollator(
        tokenizer=tokenizer,
        audio_pad_value=0.0,
        protein_pad_value=1,  # ESM pad id
        image_pad_value=0.0,
    )

    # Build Model
    logger.info("Initializing XMM Model...")
    llm = AutoModelForCausalLM.from_pretrained(
        llm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg["train"]["bf16"] else torch.float16,
    )

    # Resize embedding 因为添加了 special tokens
    llm.resize_token_embeddings(len(tokenizer))

    # 构建 Modal Configs 对象
    modal_projector_configs = {}
    for name, conf in modal_configs.items():
        modal_projector_configs[name] = ModalProjectorConfig(
            model_path=conf["model_path"],
            projector_type=conf.get("projector_type", "mlp"),
        )

    model = XMMModel(llm=llm, modal_configs=modal_projector_configs)

    # --- 5. Freeze Strategy ---
    # 冻结 Modal Encoders (通常都冻结)
    for name, conf in modal_configs.items():
        if not conf.get("trainable", False):
            logger.info(f"Freezing encoder: {name}")
            for param in model.modal_encoders[name].parameters():
                param.requires_grad = False
                logger.debug("Frozen parameter: %s", name)

    # 冻结 LLM (如果配置要求)
    if cfg["train"].get("freeze_llm", False):
        logger.info("Freezing LLM backbone...")
        for param in model.llm.parameters():
            param.requires_grad = False
        # 确保 Projectors 是可训练的
        for param in model.modal_projectors.parameters():
            param.requires_grad = True

    # 打印可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")

    training_args_dict = {**cfg["train"], **cfg["output"]}
    training_args = TrainingArguments(**training_args_dict, save_safetensors=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model()
