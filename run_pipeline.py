"""Run the pipeline stages that are switched on in ``config.yaml``.

Examples:
  python run_pipeline.py                          # every stage enabled in config.yaml
  python run_pipeline.py --only embedding tuning  # just these stages (names without "run_")
"""

import argparse

from src.utils.config_loader import get_path, load_config
from src.utils.device import seed_everything
from src.utils.io import write_json
from src.utils.logger import log_info, log_step, log_success, log_warning


def run_data_prep(config: dict) -> None:
    """Download → clean → stratified split → data/processed/manifest.csv."""
    from src.data.cleaning import clean_images
    from src.data.download import get_raw_dataset_dir
    from src.data.manifest import image_dir, manifest_path
    from src.data.pose_names import display_name
    from src.data.splitting import stratified_split

    raw_dir = get_raw_dataset_dir(config)
    df, report = clean_images(raw_dir, image_dir(config), config["data"]["min_side"], config["data"]["max_side"])
    df["split"] = stratified_split(df, config["data"]["split"], config["project"]["random_seed"])
    df["pose_display"] = df["pose"].map(display_name)
    df.to_csv(manifest_path(config), index=False)

    write_json(report.to_dict(), get_path(config, "output_dir") / "metrics" / "data_cleaning.json")
    log_info(f"Cleaning report: {report.to_dict()}")
    log_success(f"Kept {len(df):,} images of {df.pose.nunique()} poses — {df.split.value_counts().to_dict()}")


def run_embedding(config: dict) -> None:
    """Cache raw features for every backbone × preprocessing × {original, mirrored} view."""
    from src.data.manifest import load_manifest
    from src.embedding.extractor import extract_features
    from src.utils.device import get_device

    df = load_manifest(config)
    device = get_device(config["project"]["device"])
    log_info(f"Device: {device}")
    for prep in config["embedding"]["preprocess"]:
        done = extract_features(config, df["abs_path"].tolist(), config["embedding"]["backbones"], prep, device)
        log_info(f"[{prep}] extracted: {done or 'all cached'}")
    log_success(f"{len(list(get_path(config, 'embeddings_dir').glob('*.npy')))} feature matrices in data/embeddings/")


# Stage functions in execution order; each is switched on by `pipeline.<name>` in config.yaml.
STAGES: dict = {
    "run_data_prep": run_data_prep,
    "run_embedding": run_embedding,
}


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
