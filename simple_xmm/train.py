import argparse
import logging
from simple_xmm.scripts.run_sft import run_sft
from simple_xmm.scripts.run_pt import run_pt


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

    if args.stage == "sft":
        run_sft(config_path=args.config)
    elif args.stage == "pt":
        run_pt(config_path=args.config)


if __name__ == "__main__":
    main()
