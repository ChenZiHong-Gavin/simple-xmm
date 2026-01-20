import argparse
import logging
from simple_xmm.scripts.run_sft import run_sft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # Setup Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    run_sft(
        config_path=args.config, output_dir=args.output_dir, local_rank=args.local_rank
    )


if __name__ == "__main__":
    main()
