import argparse
import logging
from simple_xmm.scripts.run_train import run_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["sft", "pt"],
        help="Stage: sft or pt",
    )
    args = parser.parse_args()

    # Setup Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    run_train(config_path=args.config, stage=args.stage)


if __name__ == "__main__":
    main()
