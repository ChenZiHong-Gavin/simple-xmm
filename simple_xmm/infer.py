import argparse
import logging
import torch
from simple_xmm.scripts.run_infer import run_infer


def main():
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

    # Setup Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

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


if __name__ == "__main__":
    main()
