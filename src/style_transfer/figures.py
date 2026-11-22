"""Figures for the style-transfer study, redrawn from the saved run table (no models needed)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.pose_names import display_name
from src.style_transfer.study import load_content_table, summarize_by_style
from src.utils.config_loader import get_path
from src.utils.plotting import GRID, SERIES, save_figure, show, thumb


def plot_style_grid(config: dict, df: pd.DataFrame, contents: pd.DataFrame, path: Path) -> None:
    STYLES, OUT = config["style_transfer"]["styles"], get_path(config, "style_dir")
    grid = df[df.experiment == "grid"]
    styles = list(STYLES)
    fig, axes = plt.subplots(
        len(contents) + 1, len(styles) + 1, figsize=(2.3 * (len(styles) + 1), 2.3 * (len(contents) + 1))
    )
    axes[0, 0].axis("off")
    for j, s in enumerate(styles, start=1):
        show(axes[0, j], thumb(get_path(config, "styles_dir") / f"{s}.jpg"), STYLES[s].split(" — ")[1].split(" (")[0])
    for i, (_, c) in enumerate(contents.iterrows(), start=1):
        show(axes[i, 0], thumb(c.abs_path), display_name(c.pose).split(" (")[1].rstrip(")"))
        for j, s in enumerate(styles, start=1):
            r = grid[(grid.pose == c.pose) & (grid["style"] == s)].iloc[0]
            mark = "✓" if r.pose_retrieved else "✗"
            show(axes[i, j], thumb(OUT / r.image), f"OKS {r.pose_oks:.2f} · search {mark}", fontsize=8)
    fig.suptitle("Content photo (left) × artwork (top) — OKS = keypoint similarity to the original pose", y=0.995)
    fig.tight_layout()
    save_figure(fig, path, dpi=110)


def plot_weight_sweep(config: dict, df: pd.DataFrame, contents: pd.DataFrame, path: Path) -> None:
    s, OUT = config["style_transfer"], get_path(config, "style_dir")
    SWEEP_PAIRS, WEIGHT_RATIOS = s["sweep_pairs"], s["weight_ratios"]
    sw = df[df.experiment == "weights"].copy()
    sw["ratio"] = sw.tag.str[2:].astype(float)
    path_of = dict(zip(contents.pose, contents.abs_path, strict=True))
    fig, axes = plt.subplots(
        len(SWEEP_PAIRS), len(WEIGHT_RATIOS) + 1, figsize=(2.4 * (len(WEIGHT_RATIOS) + 1), 2.6 * len(SWEEP_PAIRS))
    )
    for i, (pose, style) in enumerate(SWEEP_PAIRS):
        show(axes[i, 0], thumb(path_of[pose]), "content")
        sub = sw[(sw.pose == pose) & (sw["style"] == style)].sort_values("ratio")
        for j, (_, r) in enumerate(sub.iterrows(), start=1):
            show(
                axes[i, j],
                thumb(OUT / r.image),
                f"β/α = 1e{int(np.log10(r.ratio))}\nOKS {r.pose_oks:.2f} · SSIM {r.ssim:.2f}",
                fontsize=8,
            )
    fig.tight_layout()
    save_figure(fig, path, dpi=110)


def plot_weight_tradeoff(df: pd.DataFrame, path: Path) -> None:
    sw = df[df.experiment == "weights"].copy()
    sw["ratio"] = sw.tag.str[2:].astype(float)
    sw["pair"] = sw.pose.map(lambda p: display_name(p).split(" (")[1].rstrip(")")) + " × " + sw["style"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, col, title in zip(
        axes,
        ("style_dist_log10", "content_dist", "pose_oks"),
        (
            "Style distance to artwork (log₁₀ Gram MSE) ↓",
            "Content distance to photo (rel. conv4_2) ↓",
            "Pose preservation (OKS) ↑",
        ),
        strict=True,
    ):
        for (pair, g), color in zip(sw.groupby("pair"), SERIES, strict=False):
            g = g.sort_values("ratio")
            ax.plot(g.ratio, g[col], marker="o", markersize=7, linewidth=2, color=color, label=pair)
        ax.set_xscale("log")
        ax.set_xlabel("style / content weight β/α")
        ax.set_title(title, fontsize=10)
        ax.grid(color=GRID, linewidth=0.8)
    axes[2].set_ylim(0, 1.02)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, path, dpi=120)


def plot_layer_ablation(config: dict, df: pd.DataFrame, contents: pd.DataFrame, path: Path) -> None:
    ABLATION_PAIRS, OUT = config["style_transfer"]["ablation_pairs"], get_path(config, "style_dir")
    sub = df[df.experiment.isin(["layers", "init"])]
    path_of = dict(zip(contents.pose, contents.abs_path, strict=True))
    cols = [("layers", "shallow"), ("layers", "all"), ("layers", "deep"), ("init", "noise")]
    labels = {
        "shallow": "style: conv1–3",
        "all": "style: conv1–5 (default)",
        "deep": "style: conv3–5",
        "noise": "noise init",
    }
    fig, axes = plt.subplots(
        len(ABLATION_PAIRS), len(cols) + 1, figsize=(2.6 * (len(cols) + 1), 2.8 * len(ABLATION_PAIRS))
    )
    for i, (pose, style) in enumerate(ABLATION_PAIRS):
        show(axes[i, 0], thumb(path_of[pose]), "content")
        for j, (exp, tag) in enumerate(cols, start=1):
            r = sub[(sub.experiment == exp) & (sub.pose == pose) & (sub["style"] == style) & (sub.tag == tag)].iloc[0]
            show(
                axes[i, j],
                thumb(OUT / r.image),
                f"{labels[tag]}\nOKS {r.pose_oks:.2f} · SSIM {r.ssim:.2f}",
                fontsize=8,
            )
    fig.tight_layout()
    save_figure(fig, path, dpi=110)


def plot_loss_curve(config: dict, path: Path) -> None:
    """Loss trajectories of the weight-sweep runs for the first sweep pair."""
    s, OUT = config["style_transfer"], get_path(config, "style_dir")
    pose, style = s["sweep_pairs"][0]
    WEIGHT_RATIOS = [float(r) for r in s["weight_ratios"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    ramp = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")  # ordinal blue ramp, light → dark
    for r, color in zip(WEIGHT_RATIOS, ramp, strict=True):
        meta = OUT / "weights" / f"{pose.replace(' ', '_')}__{style}__sw{r:.0e}.json"
        if not meta.exists():
            continue
        h = pd.DataFrame(json.loads(meta.read_text())["loss_history"])
        x = np.linspace(0, 1, len(h))
        axes[0].plot(x, h.content, color=color, linewidth=2, label=f"β/α=1e{int(np.log10(r))}")
        axes[1].plot(x, h["style"], color=color, linewidth=2, label=f"β/α=1e{int(np.log10(r))}")
    for ax, t in zip(axes, ("content loss", "style loss (Gram MSE)"), strict=True):
        ax.set_yscale("log")
        ax.set_xlabel("fraction of L-BFGS evaluations")
        ax.set_title(t, fontsize=10)
        ax.grid(color=GRID, linewidth=0.8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, path, dpi=120)


def make_style_figures(config: dict, runs: pd.DataFrame | None = None) -> None:
    metrics = get_path(config, "output_dir") / "metrics"
    figures = get_path(config, "output_dir") / "figures"
    runs = runs if runs is not None else pd.read_csv(metrics / "style_runs.csv")
    contents = load_content_table(config)
    summarize_by_style(config, runs).to_csv(metrics / "style_summary_by_style.csv")
    plot_style_grid(config, runs, contents, figures / "style_grid.jpg")
    plot_weight_sweep(config, runs, contents, figures / "style_weight_sweep.jpg")
    plot_weight_tradeoff(runs, figures / "style_weight_tradeoff.png")
    plot_layer_ablation(config, runs, contents, figures / "style_layer_ablation.jpg")
    plot_loss_curve(config, figures / "style_loss_curves.png")
