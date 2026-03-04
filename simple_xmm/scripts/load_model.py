import os
import logging
from typing import Dict, Union
import torch
from safetensors.torch import load_file as load_safetensors

logger = logging.getLogger(__name__)


def load_pretrained_weights(model, pretrained_path: Union[str, None], device: str = 'cpu'):
    """
    加载预训练权重到模型中。支持：
    - HuggingFace 目录（含 pytorch_model.bin / model.safetensors）
    - DeepSpeed ZeRO checkpoint（含 zero_to_fp32.py）
    - 单个 .bin/.pt/.pth/.safetensors 文件
    - 分片权重（pytorch_model-00001-of-00002.bin 等）
    
    Args:
        model: 目标模型
        pretrained_path: 权重路径（目录或文件），为 None 或不存在时跳过
        device: 加载设备，默认 'cpu' 避免显存占用
    """
    if not pretrained_path or not os.path.exists(pretrained_path):
        logger.info("No pretrained weights specified or path does not exist, training from scratch.")
        return

    logger.info(f"Loading pretrained weights from: {pretrained_path}")
    
    # 1. 根据路径类型加载原始 state_dict
    state_dict = _load_checkpoint_state_dict(pretrained_path, device)
    if state_dict is None:
        return
    
    # 2. 清理 state_dict（移除 optimizer 状态、DDP 前缀等）
    state_dict = _clean_state_dict(state_dict)
    
    # 3. 对齐并加载到模型
    _load_state_dict_into_model(model, state_dict)


def _load_checkpoint_state_dict(path: str, device: str) -> Union[Dict, None]:
    """根据路径类型加载 checkpoint"""
    if os.path.isdir(path):
        return _load_from_directory(path, device)
    else:
        return _load_from_file(path, device)


def _load_from_directory(checkpoint_dir: str, device: str) -> Union[Dict, None]:
    """从目录加载，支持 HF 格式和 DeepSpeed 格式"""
    # 优先级：safetensors > pytorch_model.bin > 分片 bin > DeepSpeed
    
    safetensors_file = os.path.join(checkpoint_dir, "model.safetensors")
    pytorch_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
    
    # 1. 尝试加载 safetensors
    if os.path.exists(safetensors_file):
        logger.info(f"Loading from safetensors: {safetensors_file}")
        return load_safetensors(safetensors_file, device=device)
    
    # 2. 尝试加载 pytorch_model.bin
    if os.path.exists(pytorch_file):
        logger.info(f"Loading from pytorch_model.bin")
        return torch.load(pytorch_file, map_location=device)
    
    # 3. 尝试加载分片权重
    shard_files = sorted([f for f in os.listdir(checkpoint_dir) 
                         if f.startswith("pytorch_model-") and f.endswith(".bin")])
    if shard_files:
        logger.info(f"Loading {len(shard_files)} sharded checkpoint files")
        state_dict = {}
        for shard in shard_files:
            shard_path = os.path.join(checkpoint_dir, shard)
            state_dict.update(torch.load(shard_path, map_location=device))
        return state_dict
    
    # 4. 尝试 DeepSpeed ZeRO checkpoint
    zero_file = os.path.join(checkpoint_dir, "zero_to_fp32.py")
    if os.path.exists(zero_file):
        logger.info("Detected DeepSpeed ZeRO checkpoint, converting...")
        return _load_deepspeed_checkpoint(checkpoint_dir, device)
    
    logger.error(f"No valid checkpoint found in directory: {checkpoint_dir}")
    return None


def _load_from_file(file_path: str, device: str) -> Dict:
    """从单个文件加载"""
    logger.info(f"Loading from file: {file_path}")
    
    if file_path.endswith('.safetensors'):
        return load_safetensors(file_path, device=device)
    else:
        # 支持 .bin, .pt, .pth
        return torch.load(file_path, map_location=device)


def _load_deepspeed_checkpoint(checkpoint_dir: str, device: str) -> Union[Dict, None]:
    """加载 DeepSpeed ZeRO checkpoint"""
    try:
        # 尝试自动转换
        import subprocess
        result = subprocess.run(
            ["python", "zero_to_fp32.py", ".", "pytorch_model.bin"],
            cwd=checkpoint_dir,
            capture_output=True,
            text=True,
            check=True
        )
        
        converted_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(converted_file):
            logger.info("Successfully converted DeepSpeed checkpoint")
            return torch.load(converted_file, map_location=device)
            
    except Exception as e:
        logger.error(f"DeepSpeed checkpoint conversion failed: {e}")
        logger.error("Please manually run: python zero_to_fp32.py . pytorch_model.bin")
        
    return None


def _clean_state_dict(state_dict: Dict) -> Dict:
    """清理 state_dict，移除优化器状态和分布式训练前缀"""
    # 如果是 HF Trainer 保存的 checkpoint，可能包含 'model' 键
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    
    # 移除 'module.' 前缀（来自 DDP）和 '_orig_mod.'（来自 torch.compile）
    new_state_dict = {}
    for k, v in state_dict.items():
        # 跳过优化器状态（如果存在）
        if k in ["optimizer_states", "lr_scheduler_states", "global_step"]:
            continue
            
        # 移除前缀
        original_key = k
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("_orig_mod."):
            k = k[10:]
            
        new_state_dict[k] = v
        
    return new_state_dict


def _load_state_dict_into_model(model, state_dict: Dict):
    """将 state_dict 加载到模型，处理 shape 不匹配和 missing keys"""
    model_state_dict = model.state_dict()
    
    # 过滤 shape 不匹配的参数
    filtered_state_dict = {}
    shape_mismatch_keys = []
    
    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                shape_mismatch_keys.append(
                    f"{k}: checkpoint {list(v.shape)} vs model {list(model_state_dict[k].shape)}"
                )
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    # 日志统计
    loaded_params = sum(p.numel() for p in filtered_state_dict.values())
    total_params = sum(p.numel() for p in model.parameters())
    all_checkpoint_params = sum(p.numel() for p in state_dict.values())
    
    logger.info(f"Weight loading summary:")
    logger.info(f"  - Loaded parameters: {loaded_params:,} ({loaded_params/1e6:.2f}M)")
    logger.info(f"  - Model total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info(f"  - Coverage: {loaded_params/total_params*100:.2f}%")
    logger.info(f"  - Checkpoint total: {all_checkpoint_params:,}")
    
    if shape_mismatch_keys:
        logger.warning(f"Shape mismatch ({len(shape_mismatch_keys)} keys), skipped:")
        for msg in shape_mismatch_keys[:5]:  # 只显示前5个
            logger.warning(f"    {msg}")
        if len(shape_mismatch_keys) > 5:
            logger.warning(f"    ... and {len(shape_mismatch_keys)-5} more")
    
    if missing_keys:
        # 通常 missing_keys 是预期中的（比如新添加的 LoRA 或随机初始化的头部）
        trainable_missing = [k for k in missing_keys if model.state_dict()[k].requires_grad]
        if trainable_missing:
            logger.info(f"Missing trainable keys ({len(trainable_missing)}): {trainable_missing[:5]}...")
        else:
            logger.debug(f"Missing frozen keys ({len(missing_keys)}): {missing_keys[:3]}...")
    
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
    
    logger.info("Successfully loaded pretrained weights.")


# 可选：单独加载特定组件的辅助函数
def load_component_weights(model, component_path: str, component_name: str = "projector"):
    """
    仅加载特定组件（如 projector 或 encoder）的权重
    
    Args:
        model: 完整模型
        component_path: 权重文件路径
        component_name: 组件名称（用于日志）
    """
    if not os.path.exists(component_path):
        logger.warning(f"{component_name} weights not found at {component_path}")
        return
    
    logger.info(f"Loading {component_name} weights from {component_path}")
    state_dict = torch.load(component_path, map_location='cpu')
    state_dict = _clean_state_dict(state_dict)
    
    # 过滤出该组件的键（假设键名包含组件名，如 "modal_projectors.protein"）
    component_keys = {k: v for k, v in state_dict.items() if component_name.replace('.', '_') in k or component_name in k}
    
    if not component_keys:
        logger.warning(f"No keys matching {component_name} found in checkpoint")
        return
        
    missing, unexpected = model.load_state_dict(component_keys, strict=False)
    logger.info(f"Loaded {len(component_keys)} parameters for {component_name}")