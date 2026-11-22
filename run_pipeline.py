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


def run_tuning(config: dict) -> None:
    """Grid-search every configuration on the validation split and save the winner."""
    from src.embedding.postprocess import SearchConfig, save_selection
    from src.evaluation.tuning import ablation_table, best_per_backbone, grid_search
    from src.utils.io import write_csv

    metrics_dir = get_path(config, "output_dir") / "metrics"
    grid = grid_search(config)
    write_csv(grid, metrics_dir / "tuning_grid_val.csv")
    write_csv(ablation_table(grid), metrics_dir / "tuning_ablation_val.csv")
    write_csv(best_per_backbone(grid), metrics_dir / "tuning_best_per_backbone_val.csv")

    best = grid.iloc[0]
    cfg = SearchConfig.from_dict({**best.to_dict(), "tta": bool(best.tta)})
    val = {k: float(best[k]) for k in ("P@1", "P@5", "P@10", "mAP@10", "Hit@5")}
    save_selection(metrics_dir / "selected_config.json", cfg, val)
    log_info(f"{len(grid)} configurations scored on val")
    log_success(f"Selected {cfg.key} — val P@1 {val['P@1']:.3f} · mAP@10 {val['mAP@10']:.3f}")


def run_indexing(config: dict) -> None:
    """Embed every image with the selected pipeline and load it into Qdrant."""
    from src.indexing.index_builder import build_index

    meta = build_index(config)
    write_json(meta, get_path(config, "output_dir") / "metrics" / "index_meta.json")
    log_success(f"Indexed {meta['points']:,} points · vectors {meta['dims']} · {meta['build_seconds']} s")


def run_retrieval_eval(config: dict) -> None:
    """Tune the reranker on val, compare alternatives, then score the full pipeline on test via Qdrant."""
    from dataclasses import asdict

    from src.evaluation import retrieval_eval as R
    from src.evaluation.text_search import text_search_eval
    from src.retrieval.reranker import RerankerParams, load_neighbor_table
    from src.retrieval.search_service import neighbor_table_path, reranker_selection_path
    from src.utils.device import get_device
    from src.utils.io import write_csv

    metrics_dir = get_path(config, "output_dir") / "metrics"
    table = load_neighbor_table(neighbor_table_path(config))

    grid = R.tune_reranker(config, table)
    write_csv(grid, metrics_dir / "reranker_grid_val.csv")
    best_row = R.select_reranker(grid, config["reranker"]["selection"])
    best = RerankerParams(int(best_row.fetch_k), int(best_row.k), float(best_row.weight))
    val_metrics = {k: float(best_row[k]) for k in ("P@1", "P@5", "P@10", "mAP@10", "Hit@5")}
    write_json({"params": asdict(best), "val_metrics": val_metrics}, reranker_selection_path(config))
    log_info(f"Reranker selected on val (with Hit@5 guardrail): {best}")

    alternatives = R.compare_alternatives(config, table, best)
    write_csv(alternatives, metrics_dir / "reranker_alternatives_val.csv")
    log_info("Alternatives (val):\n" + alternatives[["method", "P@1", "P@5", "mAP@10"]].round(3).to_string(index=False))

    results = {split: R.evaluate_through_qdrant(config, table, best, split) for split in ("val", "test")}
    test = results["test"]
    df = R.load_manifest(config)
    write_csv(
        R.per_pose_report(df["pose"].to_numpy(), test.pop("_qi"), test.pop("_ids")), metrics_dir / "per_pose_test.csv"
    )
    results["val"].pop("_ids"), results["val"].pop("_qi")
    write_json({"reranker": asdict(best), **results}, metrics_dir / "retrieval_final.json")

    text = text_search_eval(config, get_device(config["project"]["device"]))
    write_csv(text, metrics_dir / "text_search.csv")

    for name in ("dense", "dense+rerank"):
        m = test[name]
        summary = " · ".join(f"{k} {m[k]:.3f}" for k in ("P@1", "P@5", "mAP@10", "Hit@5"))
        log_success(f"TEST {name:13s} {summary}")


def run_style_transfer(config: dict) -> None:
    """Gatys style transfer on yoga photos: style grid, weight sweep, layer/init ablations, figures."""
    from src.style_transfer.figures import make_style_figures
    from src.style_transfer.study import Study, summarize_by_style

    runs = Study(config).run()
    runs.to_csv(get_path(config, "output_dir") / "metrics" / "style_runs.csv", index=False)
    make_style_figures(config, runs)
    summary = summarize_by_style(config, runs)[["pose_oks", "ssim", "pose_retrieved"]].round(2)
    log_info("Per artwork (mean over poses):\n" + summary.to_string())
    n_images = int((runs.experiment != "original").sum())
    log_success(f"{n_images} stylised images → data/style_transfer/ · figures → outputs/figures/")


# Stage functions in execution order; each is switched on by `pipeline.<name>` in config.yaml.
STAGES: dict = {
    "run_data_prep": run_data_prep,
    "run_embedding": run_embedding,
    "run_tuning": run_tuning,
    "run_indexing": run_indexing,
    "run_retrieval_eval": run_retrieval_eval,
    "run_style_transfer": run_style_transfer,
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
