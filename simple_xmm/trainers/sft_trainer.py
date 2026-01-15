import os
import yaml
import torch
import logging
from typing import Dict, Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

from dataset import XMMSeq2SeqDataset, XMMDataCollator
from dataset.processors import (
    AudioModalProcessor,
    ProteinModalProcessor,
    ImageModalProcessor,
)
from model import XMMModel, ModalProjectorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_processors(modal_configs: Dict, processor_args: Dict):
    """根据配置动态实例化 Processors"""
    processors = {}

    # 映射表：Tag -> Class
    # 实际项目中可以使用 registry 模式注册，这里为了简单用字典
    PROCESSOR_MAP = {
        "audio": (AudioModalProcessor, "audio"),
        "protein": (ProteinModalProcessor, "protein"),
        "image": (ImageModalProcessor, "image"),
    }

    for name, _ in modal_configs.items():
        if name not in PROCESSOR_MAP:
            logger.warning(
                f"Modality {name} not implemented in processor map, skipping."
            )
            continue

        cls, default_tag = PROCESSOR_MAP[name]
        # 获取该模态的特定参数，如 max_length
        kwargs = processor_args.get(name, {})

        # 实例化: 这里需要根据你的 Processor __init__ 签名适配
        # 假设 Processor __init__ 接收 tag 和其他 kwargs
        # 如果需要传入 encoder/tokenizer，可以在这里初始化并传入
        if name == "protein":
            # ProteinProcessor 可能需要 tokenizer
            from transformers import AutoTokenizer

            pt_tok = AutoTokenizer.from_pretrained(modal_configs[name]["model_path"])
            processors[name] = cls(tag=default_tag, protein_processor=pt_tok, **kwargs)
        elif name == "audio":
            from transformers import AutoFeatureExtractor

            au_feat = AutoFeatureExtractor.from_pretrained(
                modal_configs[name]["model_path"]
            )
            processors[name] = cls(tag=default_tag, audio_processor=au_feat, **kwargs)
        else:
            processors[name] = cls(tag=default_tag, **kwargs)

    return processors


# =================================================================
# 2. 自定义 Trainer (优雅保存)
# =================================================================
class XMMTrainer(Trainer):
    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        """
        自定义保存逻辑：
        1. 总是保存 Modal Projectors
        2. 如果 LLM 没冻结，保存 LLM
        3. 如果使用了 LoRA，保存 Adapters
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存 Projectors (这是最重要的部分)
        # 建议在 XMMModel 里实现一个 save_projectors 方法
        projectors_path = os.path.join(output_dir, "modal_projectors.pt")
        torch.save(self.model.modal_projectors.state_dict(), projectors_path)

        # 2. 保存 Config
        # 将原始 modal_configs 保存下来，推理时需要用它重建模型结构
        # 这里假设 self.model.modal_configs 存在
        # with open(os.path.join(output_dir, "modal_config.yaml"), 'w') as f: ...

        # 3. 保存 LLM (Standard Trainer logic)
        # 如果是 Full Finetuning，这里会保存巨大的模型
        # 如果是 LoRA，Standard Trainer 会只保存 Adapter (非常好)
        super().save_model(output_dir, _internal_call)

        # 保存 Tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)


# =================================================================
# 3. Main Logic
# =================================================================
def main():
    # --- Load Config ---
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(42)

    # --- 1. Prepare Text Tokenizer ---
    llm_path = cfg["model"]["llm_path"]
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Build Processors ---
    logger.info("Building processors...")
    processors = build_processors(
        cfg["model"]["modalities"], cfg["data"].get("processor_args", {})
    )

    # 注册 Special Tokens (例如 <|audio_pad|>)
    # 这一步至关重要，否则 Dataset 会报错
    special_tokens = [p.pad_token for p in processors.values()]
    tokenizer.add_tokens(special_tokens, special_tokens=True)

    # --- 3. Dataset & Collator ---
    logger.info("Loading dataset...")
    train_dataset = XMMSeq2SeqDataset(
        path=cfg["data"]["train_file"],
        template=cfg["data"]["template"],
        tokenizer=tokenizer,
        processors=processors,
        max_samples=cfg["data"].get("max_samples", None),
    )

    # Collator 需要各模态的 Pad Value
    # 这里需要根据实际使用的 Processor/Tokenizer 获取 padding value
    # 简化起见，这里硬编码了，实际工程中应从 processors 实例中获取
    collator = XMMDataCollator(
        tokenizer=tokenizer,
        audio_pad_value=0.0,
        protein_pad_value=1,  # ESM pad id
        image_pad_value=0.0,
    )

    # --- 4. Build Model ---
    logger.info("Initializing XMM Model...")
    llm = AutoModelForCausalLM.from_pretrained(
        llm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg["training"]["bf16"] else torch.float16,
    )

    # Resize embedding 因为添加了 special tokens
    llm.resize_token_embeddings(len(tokenizer))

    # 构建 Modal Configs 对象
    modal_projector_configs = {}
    for name, conf in cfg["model"]["modalities"].items():
        modal_projector_configs[name] = ModalProjectorConfig(
            model_path=conf["model_path"],
            projector_type=conf.get("projector_type", "mlp"),
        )

    model = XMMModel(llm=llm, modal_configs=modal_projector_configs)

    # --- 5. Freeze Strategy ---
    # 冻结 Modal Encoders (通常都冻结)
    for name, conf in cfg["model"]["modalities"].items():
        if not conf.get("trainable", False):
            logger.info(f"Freezing encoder: {name}")
            for param in model.modal_encoders[name].parameters():
                param.requires_grad = False

    # 冻结 LLM (如果配置要求)
    if cfg["training"].get("freeze_llm", False):
        logger.info("Freezing LLM backbone...")
        for param in model.llm.parameters():
            param.requires_grad = False
        # 确保 Projectors 是可训练的
        for param in model.modal_projectors.parameters():
            param.requires_grad = True

    # 打印可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")

    # --- 6. Trainer ---
    training_args = TrainingArguments(**cfg["training"])

    trainer = XMMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # --- 7. Start Training ---
    logger.info("Starting training...")
    trainer.train()

    # 最终保存
    trainer.save_model()


if __name__ == "__main__":
    main()
