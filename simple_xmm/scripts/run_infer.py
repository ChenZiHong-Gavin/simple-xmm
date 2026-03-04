import argparse
import logging
import torch
from typing import Dict
from omegaconf import OmegaConf
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)

from simple_xmm.models.model_linear import XMMLinearProjectorModel
from simple_xmm.models.model_mlp import XMMMlpProjectorModel
from simple_xmm.models.model_qformer import XMMQFormerProjectorModel
from simple_xmm.scripts.run_train import build_processors, set_special_tokens
from simple_xmm.scripts.load_model import load_pretrained_weights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_inference_input(text: str, tokenizer, processors: Dict):
    """
    Process input text and modalities for inference.
    """
    input_ids = []
    modal_info = []
    modal_features_list = {name: [] for name in processors.keys()}

    # Find all modality matches
    matches = []
    for name, proc in processors.items():
        for m in proc.pattern.finditer(text):
            matches.append(
                {
                    "name": name,
                    "processor": proc,
                    "start": m.start(),
                    "end": m.end(),
                    "content": m.group(1),
                }
            )

    # Sort by position
    matches.sort(key=lambda x: x["start"])

    curr_pos = 0
    for m in matches:
        # Append text before this modality
        if m["start"] > curr_pos:
            text_part = text[curr_pos : m["start"]]
            input_ids.extend(tokenizer.encode(text_part, add_special_tokens=False))

        # Process modality
        proc = m["processor"]
        name = m["name"]
        content = m["content"]

        # Load/Process feature
        try:
            # Note: process might return different structures depending on processor
            # But usually it returns a dict or tensor.
            # We assume it returns something we can put in a list and then pad.
            processed_feature = proc.process(content)
            modal_features_list[name].append(processed_feature)

            pad_id = tokenizer.convert_tokens_to_ids(proc.pad_token)

            modal_info.append(
                {
                    "type": name,
                    "start": len(input_ids),  # position of placeholder
                    "raw": content,
                    # "content": processed_feature # We don't need to store big tensors in info
                }
            )

            # Insert placeholder
            input_ids.append(pad_id)

        except Exception as e:
            logger.error(f"Failed to process modality {name} content '{content}': {e}")
            # If failed, maybe skip or treat as text?
            # For now, let's just skip this modality tag but keep text?
            # Or raise error.
            raise e

        curr_pos = m["end"]

    # Append remaining text
    if curr_pos < len(text):
        input_ids.extend(tokenizer.encode(text[curr_pos:], add_special_tokens=False))

    # Convert input_ids to tensor (Batch size 1)
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids_tensor)

    # Prepare modal_inputs
    modal_inputs = {}
    for name, features in modal_features_list.items():
        if features:
            proc = processors[name]
            # pad expects a list of features
            values, masks = proc.pad(features)
            modal_inputs[f"{name}_values"] = values
            modal_inputs[f"{name}_attention_mask"] = masks

    # Wrap modal_info in a list (batch size 1)
    modal_info_batch = [modal_info]

    return input_ids_tensor, attention_mask, modal_info_batch, modal_inputs


def run_infer(
    config_path,
    checkpoint_path,
    text=None,
    input_file=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    use_chat_template=False,
    **kwargs,
):
    if not text and not input_file:
        raise ValueError("Either text or input_file must be provided.")

    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

    cfg = OmegaConf.load(config_path)
    set_seed(42)

    # 1. Tokenizer
    llm_path = cfg["model"]["llm_name_or_path"]
    trust_remote_code = cfg["model"].get("trust_remote_code", False)
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path, trust_remote_code=trust_remote_code
    )

    modal_configs = (
        cfg["model"]["modal_configs"] if "modal_configs" in cfg["model"] else {}
    )

    # Apply chat template if requested
    if use_chat_template:
        messages = [{"role": "user", "content": text}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # 2. Processors
    logger.info("Setting up processors...")
    processors = build_processors(modal_configs)
    set_special_tokens(tokenizer, processors)

    # 3. Model
    logger.info("Initializing XMM Model...")
    llm = AutoModelForCausalLM.from_pretrained(
        llm_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16,  # Use float16 for inference
        device_map=device,
    )
    llm.resize_token_embeddings(len(tokenizer))

    projector_type = cfg["model"].get("projector_type", "mlp")
    if projector_type == "linear":
        model_cls = XMMLinearProjectorModel
    elif projector_type == "qformer":
        model_cls = XMMQFormerProjectorModel
    else:
        model_cls = XMMMlpProjectorModel

    model = model_cls(llm=llm, modal_configs=modal_configs)

    # 4. Load Weights
    logger.info(f"Loading weights from {checkpoint_path}...")
    load_pretrained_weights(model, checkpoint_path, device=device)

    model.to(device)
    model.eval()

    # 5. Process Input
    logger.info(f"Processing input: {text}")
    input_ids, attention_mask, modal_info, modal_inputs = process_inference_input(
        text, tokenizer, processors
    )

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    # Move modal inputs to device
    for k, v in modal_inputs.items():
        if isinstance(v, torch.Tensor):
            modal_inputs[k] = v.to(device)

    # 6. Generate
    logger.info(
        f"Generating with params: max_new_tokens={max_new_tokens}, do_sample={do_sample}, temp={temperature}, top_p={top_p}"
    )
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            modal_info=modal_info,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            **modal_inputs,
        )

    # 7. Decode
    generated_ids = outputs[0]
    # Skip input tokens
    # generated_ids = generated_ids[input_ids.shape[1]:]
    # Usually generate returns input + new tokens

    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("-" * 20 + " Output " + "-" * 20)
    print(decoded_text)
    print("-" * 50)

    return decoded_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (folder or file)",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        type=str,
        help="Input text with modal tags, e.g. 'Describe <image>path.jpg</image>'",
    )
    input_group.add_argument(
        "--input_file", type=str, help="Path to text file containing input text"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    # Generation args
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Max new tokens to generate"
    )
    parser.add_argument("--do_sample", action="store_true", help="Whether to sample")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top p sampling")

    args = parser.parse_args()

    run_infer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        text=args.text,
        input_file=args.input_file,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
