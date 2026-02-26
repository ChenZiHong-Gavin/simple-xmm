<div align="center">
<a href="https://github.com/your-repo/simple-xmm">
<img height=350 alt="Simple-XMM" src="https://capsule-render.vercel.app/api?type=waving&color=ffdd7a&height=300&section=header&text=Simple-XMM&fontSize=70&fontColor=0f0000&animation=fadeIn&fontAlignY=38&desc=A%20Simple%20Multimodal%20LLM%20Framework&descAlignY=60&descAlign=50"></img></a>
</div>

Simple-XMM 是一个简易且可扩展的多模态大语言模型（MLLM）框架，旨在简化多模态模型的训练和微调流程。它采用了模块化的设计，将模态处理器（Processor）和编码器（Encoder）解耦，支持图像、音频、蛋白质等多种模态。

## ✨ 特性

- **多模态支持**：

  | 模态 (Modality) | 模型 (Model) | 描述 (Description) |
  | :--- | :--- | :--- |
  | 🖼️ **图像 (Image)** | CLIP | 集成 CLIP 模型 |
  | 🎙️ **音频 (Audio)** | Whisper | 集成 Whisper 模型 |
  | 🧬 **蛋白质 (Protein)** | ESM | 集成 ESM 模型 |
  | 🧬 **DNA** | Nucleotide Transformer (NT) | 集成 Nucleotide Transformer (NT) 模型 |
  | 🧪 **RNA** | RNA-FM | 集成 RNA-FM 模型 |
- **模块化架构**：轻松扩展新的模态，只需实现对应的 `Encoder` 和 `Processor`。
- **SFT 训练**：内置 Supervised Fine-Tuning (SFT) 流程，支持多模态数据的混合训练。
- **配置灵活**：基于 OmegaConf 的 YAML 配置管理。

## 🛠️ 安装

```bash
git clone https://github.com/your-repo/simple-xmm.git
cd simple-xmm
pip install -r requirements.txt
```

## 🚀 快速开始

### 准备数据

数据格式遵循 Alpaca 风格，并支持多模态标签。例如：

```json
[
  {
    "instruction": "Describe this image.",
    "input": "<image>/path/to/image.jpg</image>",
    "output": "A cute cat sitting on the sofa."
  }
]
```

### 运行 SFT 训练

项目提供了多种模态的训练配置示例，位于 `examples/` 目录下。

```bash
# 图像模态训练示例
python simple_xmm/train.py --config examples/train_full/ti2t/clip/sft.yaml --stage sft

# 音频模态训练示例
python simple_xmm/train.py --config examples/train_full/ta2t/sft.yaml --stage sft

# DNA 模态训练示例
python simple_xmm/train.py --config examples/train_full/td2t/nt/sft.yaml --stage sft

# RNA 模态训练示例
python simple_xmm/train.py --config examples/train_full/tr2t/fm/sft.yaml --stage sft
```

## 📂 项目结构

```
simple_xmm/
├── datasets/       # 数据集处理
├── modalities/     # 模态实现 (Audio, Image, Protein, DNA, RNA)
├── models/         # 模型架构 (Splice Model)
├── scripts/        # 核心逻辑脚本 (run_sft)
├── configs/        # 配置模板
└── train.py        # 训练入口
```

## 🤝 贡献

欢迎提交 PR 扩展更多的模态或优化模型架构！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feat/new-modality`)
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

MIT License
