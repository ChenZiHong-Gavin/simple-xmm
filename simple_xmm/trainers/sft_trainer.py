# train_sft.py
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer
from model import create_cross_modal_model
from dataset import CrossModalDataset, cross_modal_collate_fn


def main():
    # 1. 配置
    modal_configs = {
        "protein": {
            "model_path": "facebook/esm2_t12_35M_UR50D",
            "num_tokens": 64,
        },
    }

    # 2. 创建模型
    model = create_cross_modal_model("Qwen/Qwen2-7B-Instruct", modal_configs)

    # 3. Tokenizers
    text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    modal_tokenizers = {
        "protein": AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D"),
    }

    # 4. 数据集
    train_dataset = CrossModalDataset(
        "data/train.parquet",
        text_tokenizer,
        modal_tokenizers,
        modal_token_nums={"protein": 64},
    )

    # 5. 训练参数
    training_args = SFTConfig(
        output_dir="./sft_outputs",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        max_seq_length=2048,
        dataset_text_field=None,  # 我们自定义forward
    )

    # 6. 自定义Trainer
    class CustomSFTTrainer(SFTTrainer):
        def save_model(self, output_dir, _internal_call=False):
            super().save_model(output_dir, _internal_call)
            if hasattr(self.model, "save_modal_projectors"):
                self.model.save_modal_projectors(output_dir)

    # 7. 训练
    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=cross_modal_collate_fn,
    )

    trainer.train()
    model.llm.save_pretrained(f"{training_args.output_dir}/llm")


if __name__ == "__main__":
    main()
