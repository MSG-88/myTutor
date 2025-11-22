"""Entry point for running the fine-tuning pipeline via YAML config."""
import argparse
from pathlib import Path
import yaml
from slm_finetune_framework.pipeline.orchestrator import FinetunePipeline


DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "example_config.yaml"


def main(config_path: str | Path = DEFAULT_CONFIG_PATH):
    """Load configuration and execute the pipeline."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    pipeline = FinetunePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SLM finetuning pipeline.")
    parser.add_argument(
        "--config",
        dest="config_path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config_path)
