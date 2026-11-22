"""Figures for the search part, built from saved metrics and cached vectors (no vector database needed).

Exact NumPy search over the cached vectors returns the same neighbours as the local Qdrant index,
so the figures can be rebuilt while the app holds the database.
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.manifest import load_manifest, split_masks
from src.data.pose_names import display_name
from src.embedding.backbones import load_backbone
from src.embedding.postprocess import build_vectors, load_selection
from src.indexing.index_builder import CLIP_CONFIG, selection_path
from src.retrieval.reranker import NeighborhoodReranker, load_neighbor_table
from src.retrieval.search_service import load_reranker_params, neighbor_table_path
from src.utils.config_loader import get_path
from src.utils.device import get_device
from src.utils.plotting import BACKBONE_LABELS, BAD, GOOD, GRID, SERIES, TEXT, TEXT_2, save_figure, thumb


def plot_backbones(best: pd.DataFrame, path: Path) -> Path:
    """Grouped horizontal bars: best val mAP@10 per backbone, unsupervised vs + LDA."""
    pivot = best.pivot_table(index="backbone", columns="setting", values="mAP@10")
    pivot = pivot.reindex(pivot["+ LDA"].sort_values().index)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = np.arange(len(pivot))
    h = 0.38
    for i, (setting, color) in enumerate((("unsupervised", SERIES[0]), ("+ LDA", SERIES[1]))):
        vals = pivot[setting].to_numpy()
        ax.barh(y + (0.5 - i) * h, vals, height=h - 0.04, color=color, label=setting)
        for yy, v in zip(y + (0.5 - i) * h, vals, strict=True):
            ax.text(v + 0.006, yy, f"{v:.3f}", va="center", fontsize=8, color=TEXT_2)
    ax.set_yticks(y, [BACKBONE_LABELS.get(b, b) for b in pivot.index])
    ax.set_xlabel("validation mAP@10 (best setting per backbone)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(0.65, pivot.to_numpy().max() + 0.08))
    ax.legend(loc="lower right")
    ax.set_title("Backbone comparison — frozen features vs. + supervised LDA projection", loc="left")
    return save_figure(fig, path)


def plot_ablation(abl: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = np.arange(len(abl))[::-1]
    ax.barh(y, abl["mAP@10"], color=SERIES[0], height=0.62)
    for yy, v, p1 in zip(y, abl["mAP@10"], abl["P@1"], strict=True):
        ax.text(v + 0.006, yy, f"{v:.3f}  (P@1 {p1:.0%})", va="center", fontsize=8, color=TEXT_2)
    ax.set_yticks(y, abl["step"])
    ax.set_xlabel("validation mAP@10")
    ax.set_xlim(0, abl["mAP@10"].max() + 0.17)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Ablation — one change at a time from the textbook baseline", loc="left")
    return save_figure(fig, path)


def plot_hardest_poses(per_pose: pd.DataFrame, path: Path, n: int = 12) -> Path:
    worst = per_pose.nsmallest(n, "P@5").iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * n + 0.8))
    y = np.arange(len(worst))
    ax.barh(y, worst["P@5"], color=SERIES[1], height=0.62)
    for yy, (_, r) in zip(y, worst.iterrows(), strict=True):
        conf = display_name(r.most_confused_with).split(" (")[0] if r.most_confused_with else "—"
        ax.text(r["P@5"] + 0.01, yy, f"{r['P@5']:.0%}  · confused with {conf}", va="center", fontsize=7.5, color=TEXT_2)
    ax.set_yticks(y, [display_name(p).split(" (")[0] for p in worst.pose], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("test Precision@5")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"{n} hardest poses for the selected pipeline", loc="left")
    return save_figure(fig, path)


def plot_query_results(
    queries: list[tuple[str | Path | None, str, list[tuple[str, str, float]]]],
    path: Path,
    title: str,
) -> Path:
    """Rows of ``(query image or None, query label, [(result path, result pose, score), ...])``."""
    k = max(len(r) for _, _, r in queries)
    fig, axes = plt.subplots(len(queries), k + 1, figsize=(1.75 * (k + 1), 2.3 * len(queries)))
    axes = np.atleast_2d(axes)
    for i, (qimg, qlabel, results) in enumerate(queries):
        ax = axes[i, 0]
        if qimg is not None:
            ax.imshow(thumb(qimg, 200, square=True))
        else:
            ax.text(0.5, 0.5, "“" + textwrap.fill(qlabel, 20) + "”", ha="center", va="center", fontsize=8.5, color=TEXT)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        if qimg is not None:
            ax.set_xlabel("QUERY\n" + display_name(qlabel).split(" (")[1].rstrip(")"), fontsize=7.5, color=TEXT)
        else:
            ax.set_xlabel("TEXT QUERY", fontsize=7.5, color=TEXT_2)
        for j in range(k):
            ax = axes[i, j + 1]
            if j >= len(results):
                ax.axis("off")
                continue
            rpath, rpose, score = results[j]
            ax.imshow(thumb(rpath, 200, square=True))
            correct = qimg is None or rpose == qlabel
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(2.5)
                s.set_edgecolor(GRID if qimg is None else (GOOD if correct else BAD))
            mark = "" if qimg is None else ("✓ " if correct else "✗ ")
            ax.set_xlabel(
                f"{mark}{display_name(rpose).split(' (')[1].rstrip(')')[:22]}\n{score:.3f}", fontsize=7, color=TEXT
            )
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
        for s in axes[i, 0].spines.values():
            s.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=TEXT)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_reranker_alternatives(alt: pd.DataFrame, path: Path) -> Path:
    """Second-stage options on val: mAP@10 bars, Hit@5 in the label."""
    alt = alt.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 2.9))
    y = np.arange(len(alt))
    colors = [SERIES[0] if "baseline" not in m else GRID for m in alt.method]
    colors = [SERIES[1] if m.startswith("neighbourhood") else c for m, c in zip(alt.method, colors, strict=True)]
    ax.barh(y, alt["mAP@10"], color=colors, height=0.62)
    for yy, (_, r) in zip(y, alt.iterrows(), strict=True):
        ax.text(
            r["mAP@10"] + 0.006,
            yy,
            f"{r['mAP@10']:.3f}  (Hit@5 {r['Hit@5']:.2f})",
            va="center",
            fontsize=8,
            color=TEXT_2,
        )
    ax.set_yticks(y, [m.split(" (")[0] if m.startswith("neighbourhood") else m for m in alt.method])
    ax.set_xlabel("validation mAP@10")
    ax.set_xlim(0, alt["mAP@10"].max() + 0.16)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Second stage — reranker vs. alternatives (dense baseline in grey)", loc="left")
    return save_figure(fig, path)


def plot_reranker_tradeoff(grid: pd.DataFrame, selected: dict, max_drop: float, path: Path) -> Path:
    """Every reranker setting on val: precision (mAP@10) vs. hit rate (Hit@5), with the guardrail."""
    base = grid[grid.fetch_k == 0].iloc[0]
    rest = grid[grid.fetch_k > 0]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for (w, g), color in zip(rest.groupby("weight"), [*SERIES, TEXT_2], strict=False):
        ax.scatter(
            g["Hit@5"], g["mAP@10"], s=34, color=color, label=f"λ = {w}", edgecolor="white", linewidth=0.8, zorder=3
        )
    ax.scatter([base["Hit@5"]], [base["mAP@10"]], s=90, marker="s", color=TEXT, label="no reranker", zorder=4)
    floor = base["Hit@5"] - max_drop
    ax.axvline(floor, color=BAD, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(floor, ax.get_ylim()[0] + 0.005, f"  guardrail: Hit@5 ≥ {floor:.3f}", color=BAD, fontsize=8, va="bottom")
    ax.annotate(
        "selected",
        (selected["Hit@5"], selected["mAP@10"]),
        xytext=(18, 10),
        textcoords="offset points",
        fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": TEXT_2},
    )
    ax.set_xlabel("validation Hit@5 (≥ 1 correct pose in the top 5)")
    ax.set_ylabel("validation mAP@10")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.set_title("Reranker settings: higher λ buys precision at the cost of hit rate", loc="left")
    return save_figure(fig, path)


TEXT_QUERIES = (
    "a headstand",
    "Warrior II pose with arms stretched out",
    "a person standing on one leg holding their foot behind them",
)


def make_search_figures(config: dict, n_examples: int = 4, k: int = 6, seed: int = 7) -> None:
    metrics = get_path(config, "output_dir") / "metrics"
    figures = get_path(config, "output_dir") / "figures"
    plot_backbones(pd.read_csv(metrics / "tuning_best_per_backbone_val.csv"), figures / "search_backbones.png")
    plot_ablation(pd.read_csv(metrics / "tuning_ablation_val.csv"), figures / "search_ablation.png")
    plot_hardest_poses(pd.read_csv(metrics / "per_pose_test.csv"), path=figures / "search_hardest_poses.png")
    plot_reranker_alternatives(
        pd.read_csv(metrics / "reranker_alternatives_val.csv"), figures / "reranker_alternatives.png"
    )
    selected = pd.read_json(metrics / "selected_reranker.json", typ="series")["val_metrics"]
    plot_reranker_tradeoff(
        pd.read_csv(metrics / "reranker_grid_val.csv"),
        selected,
        config["reranker"]["selection"]["max_guardrail_drop"],
        figures / "reranker_tradeoff.png",
    )

    # Example queries: held-out test photos → final (dense + reranker) results.
    df = load_manifest(config)
    labels, g, _, t = split_masks(df)
    vecs, _ = build_vectors(config, load_selection(selection_path(config)), g, labels)
    gi, ti = np.flatnonzero(g), np.flatnonzero(t)
    params = load_reranker_params(config)
    reranker = NeighborhoodReranker(load_neighbor_table(neighbor_table_path(config)), params)
    sims = vecs[ti] @ vecs[gi].T
    cand = np.argsort(-sims, axis=1)[:, : params.fetch_k]
    results = []
    for qi in range(len(ti)):
        ids, scores = reranker.rerank(gi[cand[qi]], sims[qi, cand[qi]])
        results.append((ids[:k], scores[:k]))
    p_at_k = np.array([(labels[ids] == labels[ti[qi]]).mean() for qi, (ids, _) in enumerate(results)])
    top_sim = sims[np.arange(len(ti)), cand[:, 0]]
    rng = np.random.default_rng(seed)
    good = rng.choice(
        np.flatnonzero((p_at_k >= 5 / 6) & (top_sim < 0.85)), n_examples, replace=False
    )  # skip near-duplicates
    bad = rng.choice(np.flatnonzero(p_at_k <= 1 / 6), 2, replace=False)

    def rows(selection):
        return [
            (
                df.abs_path.iloc[ti[q]],
                labels[ti[q]],
                [(df.abs_path.iloc[i], labels[i], float(s)) for i, s in zip(*results[q], strict=True)],
            )
            for q in selection
        ]

    plot_query_results(
        rows(good),
        figures / "search_examples.png",
        "Held-out test queries → top-6 results after reranking (green = same pose)",
    )
    plot_query_results(
        rows(bad), figures / "search_failures.png", "Failure cases — visually similar but different poses"
    )

    device = get_device(config["project"]["device"])
    clip_vecs, _ = build_vectors(config, CLIP_CONFIG, g)
    text = load_backbone("clip_rn50", device).encode_text(list(TEXT_QUERIES), device).cpu().numpy()
    tsims = text @ clip_vecs[gi].T
    trows = []
    for i, q in enumerate(TEXT_QUERIES):
        order = np.argsort(-tsims[i])[:k]
        trows.append((None, q, [(df.abs_path.iloc[gi[j]], labels[gi[j]], float(tsims[i, j])) for j in order]))
    plot_query_results(trows, figures / "search_text_examples.png", "Text → image search with CLIP-RN50")
