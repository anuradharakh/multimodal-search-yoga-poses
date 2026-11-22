"""Run the pipeline stages that are switched on in ``config.yaml``.

Examples:
  python run_pipeline.py                          # every stage enabled in config.yaml
  python run_pipeline.py --only embedding tuning  # just these stages (names without "run_")
"""

import argparse

from src.utils.config_loader import load_config
from src.utils.device import seed_everything
from src.utils.logger import log_step, log_success, log_warning

# Stage functions are registered here in execution order as the pipeline is built up.
STAGES: dict = {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--only", nargs="*", help="run only these stages, e.g. --only data_prep embedding")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["project"]["random_seed"])

    selected = [
        (flag, fn)
        for flag, fn in STAGES.items()
        if (flag.removeprefix("run_") in args.only if args.only else config["pipeline"].get(flag, False))
    ]
    if not selected:
        log_warning("No stages selected — enable some under `pipeline:` in config.yaml or pass --only.")
    for flag, fn in selected:
        log_step(flag.removeprefix("run_").replace("_", " ").upper())
        fn(config)
    if selected:
        log_success("Pipeline finished.")


if __name__ == "__main__":
    main()
