"""
Generate comparison charts from results/ JSON files.

Usage (from code/):
  python plot_results.py
  python plot_results.py --run 200000
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR
from utils import load_json, setup_logger

logger = setup_logger("PlotResults")

METRICS = ("MAP", "Recall", "Precision@10", "nDCG@10")
MODEL_LABELS = {
    "bm25": "BM25",
    "tfidf": "TF-IDF",
    "embedding": "Dense",
    "hybrid_serial": "Hybrid Serial",
    "hybrid_parallel": "Hybrid Parallel",
    "before_refinement": "Before Refinement",
    "after_refinement": "After Refinement",
}

COLORS = {
    "bm25": "#4C78A8",
    "tfidf": "#72B7B2",
    "embedding": "#F58518",
    "hybrid_serial": "#E45756",
    "hybrid_parallel": "#54A24B",
    "before_refinement": "#4C78A8",
    "after_refinement": "#E45756",
}


def _run_label(run_dir: Path) -> str:
    name = run_dir.name
    return "full" if name == "run_full" else name.replace("run_", "")


def _find_run_dirs(run_filter=None):
    bases = [
        RESULTS_DIR / "dense",
        RESULTS_DIR / "baseline",
        RESULTS_DIR / "refinement",
        RESULTS_DIR / "bm25_tuning",
    ]
    runs = {}
    for base in bases:
        if not base.exists():
            continue
        for run_dir in sorted(base.glob("run_*")):
            label = _run_label(run_dir)
            if run_filter and label != str(run_filter) and run_filter not in run_dir.name:
                continue
            runs.setdefault(label, {})[base.name] = run_dir
    return runs


def _extract_metrics_from_comparison(data):
    models = {}
    for name, entry in data.get("models", {}).items():
        metrics = entry.get("metrics", entry)
        models[name] = {
            m: float(metrics.get(m, metrics.get(m.replace("@10", ""), 0)))
            for m in METRICS
        }
        if "retrieval_seconds" in entry:
            models[name]["time_s"] = entry["retrieval_seconds"]
    return models


def _normalize_baseline(data):
    models = {}
    for name, metrics in data.get("models", {}).items():
        models[name] = {
            "MAP": float(metrics.get("MAP", 0)),
            "Recall": float(metrics.get("Mean Recall", metrics.get("Recall", 0))),
            "Precision@10": float(metrics.get("Mean P@10", metrics.get("Precision@10", 0))),
            "nDCG@10": float(metrics.get("nDCG@10", 0)),
        }
    return models


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_all_models_metrics(models: dict, out_dir: Path, title_suffix: str):
    names = [n for n in models if n in MODEL_LABELS or True]
    order = ["bm25", "tfidf", "embedding", "hybrid_serial", "hybrid_parallel"]
    names = [n for n in order if n in models] or list(models.keys())
    labels = [MODEL_LABELS.get(n, n) for n in names]
    x = np.arange(len(METRICS))
    width = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(names):
        vals = [models[name].get(m, 0) for m in METRICS]
        offset = (i - len(names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=MODEL_LABELS.get(name, name),
                      color=COLORS.get(name, None), alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(max(models[n].get(m, 0) for m in METRICS) for n in names) * 1.15)
    ax.set_title(f"All Retrieval Models — Metrics Comparison ({title_suffix})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "01_all_models_metrics.png")


def plot_map_ranking(models: dict, out_dir: Path, title_suffix: str):
    ranked = sorted(models.items(), key=lambda x: x[1].get("MAP", 0), reverse=True)
    names = [MODEL_LABELS.get(n, n) for n, _ in ranked]
    maps = [m.get("MAP", 0) for _, m in ranked]
    colors = [COLORS.get(n, "#888") for n, _ in ranked]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(names))
    bars = ax.barh(y, maps, color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("MAP")
    ax.set_title(f"Model Ranking by MAP ({title_suffix})")
    for bar, val in zip(bars, maps):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
                va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out_dir, "02_map_ranking.png")


def plot_retrieval_time(models: dict, out_dir: Path, title_suffix: str):
    timed = {k: v for k, v in models.items() if "time_s" in v}
    if not timed:
        return
    order = ["bm25", "tfidf", "embedding", "hybrid_serial", "hybrid_parallel"]
    names = [n for n in order if n in timed] or list(timed.keys())
    labels = [MODEL_LABELS.get(n, n) for n in names]
    times = [timed[n]["time_s"] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, times, color=[COLORS.get(n, "#888") for n in names], alpha=0.9)
    ax.set_ylabel("Retrieval time (seconds)")
    ax.set_title(f"Retrieval Speed by Model ({title_suffix})")
    plt.xticks(rotation=15, ha="right")
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f}s", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "03_retrieval_time.png")


def plot_refinement(before_after: dict, out_dir: Path, title_suffix: str):
    before = before_after.get("before", {})
    after = before_after.get("after", {})
    if not before or not after:
        return

    x = np.arange(len(METRICS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b_vals = [before.get(m, 0) for m in METRICS]
    a_vals = [after.get(m, 0) for m in METRICS]
    ax.bar(x - width / 2, b_vals, width, label="Before Refinement", color="#4C78A8", alpha=0.9)
    ax.bar(x + width / 2, a_vals, width, label="After Refinement", color="#E45756", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_ylabel("Score")
    ax.set_title(f"Query Refinement — Before vs After ({title_suffix})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "04_refinement_before_after.png")


def plot_baseline_vs_advanced(baseline_models: dict, dense_models: dict, out_dir: Path, title_suffix: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (title, models) in zip(axes, [("Baseline (Lexical)", baseline_models),
                                           ("All Models", dense_models)]):
        names = list(models.keys())
        labels = [MODEL_LABELS.get(n, n) for n in names]
        maps = [models[n].get("MAP", 0) for n in names]
        ax.bar(labels, maps, color=[COLORS.get(n, "#888") for n in names], alpha=0.9)
        ax.set_title(title)
        ax.set_ylabel("MAP")
        ax.set_ylim(0, max(maps) * 1.2 if maps else 1)
        plt.sca(ax)
        plt.xticks(rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(maps):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle(f"Baseline vs Advanced Models ({title_suffix})", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "05_baseline_vs_advanced.png")


def plot_bm25_heatmap(grid_results: list, out_dir: Path, title_suffix: str):
    if not grid_results:
        return
    k1_vals = sorted({r["k1"] for r in grid_results})
    b_vals = sorted({r["b"] for r in grid_results})
    matrix = np.zeros((len(b_vals), len(k1_vals)))
    for r in grid_results:
        i = b_vals.index(r["b"])
        j = k1_vals.index(r["k1"])
        matrix[i, j] = r["metrics"]["MAP"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(k1_vals)))
    ax.set_xticklabels([str(k) for k in k1_vals])
    ax.set_yticks(range(len(b_vals)))
    ax.set_yticklabels([str(b) for b in b_vals])
    ax.set_xlabel("k1")
    ax.set_ylabel("b")
    ax.set_title(f"BM25 Tuning — MAP Heatmap ({title_suffix})")
    for i in range(len(b_vals)):
        for j in range(len(k1_vals)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="MAP")
    _save(fig, out_dir, "06_bm25_tuning_heatmap.png")


def plot_radar(models: dict, out_dir: Path, title_suffix: str):
    order = ["bm25", "tfidf", "embedding", "hybrid_serial", "hybrid_parallel"]
    names = [n for n in order if n in models]
    if len(names) < 2:
        return

    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name in names:
        vals = [models[name].get(m, 0) for m in METRICS]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=MODEL_LABELS.get(name, name),
                color=COLORS.get(name))
        ax.fill(angles, vals, alpha=0.08, color=COLORS.get(name))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRICS)
    ax.set_title(f"Model Comparison Radar ({title_suffix})", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    _save(fig, out_dir, "07_model_radar.png")


def plot_summary_table(models: dict, refinement: dict, tuning_best: dict, out_dir: Path, title_suffix: str):
    rows = []
    order = ["bm25", "tfidf", "embedding", "hybrid_serial", "hybrid_parallel"]
    for name in order:
        if name in models:
            m = models[name]
            rows.append([MODEL_LABELS.get(name, name)] + [f"{m.get(k, 0):.4f}" for k in METRICS])

    if refinement:
        rows.append(["Refinement (before)"] + [f"{refinement['before'].get(k, 0):.4f}" for k in METRICS])
        rows.append(["Refinement (after)"] + [f"{refinement['after'].get(k, 0):.4f}" for k in METRICS])

    if tuning_best:
        rows.append([f"BM25 tuned k1={tuning_best.get('k1')}, b={tuning_best.get('b')}"] +
                    [f"{tuning_best['metrics'].get(k, 0):.4f}" for k in METRICS])

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(12, max(3, 0.4 * len(rows) + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Model"] + list(METRICS),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title(f"All Results Summary ({title_suffix})", fontsize=12, pad=20)
    _save(fig, out_dir, "08_summary_table.png")


def generate_charts(run_filter=None):
    runs = _find_run_dirs(run_filter)
    if not runs:
        logger.error("No result runs found under results/")
        return []

    generated = []
    for run_label, paths in runs.items():
        out_dir = RESULTS_DIR / "charts" / f"run_{run_label}"
        title_suffix = f"{run_label} docs, 10K queries"

        dense_path = paths.get("dense", Path()) / "full_comparison.json"
        baseline_path = paths.get("baseline", Path()) / "baseline_comparison.json"
        refinement_path = paths.get("refinement", Path()) / "before_after_analysis.json"
        tuning_path = paths.get("bm25_tuning", Path()) / "bm25_tuning_report.json"

        dense_models = {}
        baseline_models = {}
        refinement_data = {}
        tuning_best = {}

        if dense_path.exists():
            data = load_json(dense_path)
            dense_models = _extract_metrics_from_comparison(data)
            plot_all_models_metrics(dense_models, out_dir, title_suffix)
            plot_map_ranking(dense_models, out_dir, title_suffix)
            plot_retrieval_time(dense_models, out_dir, title_suffix)
            plot_radar(dense_models, out_dir, title_suffix)

        if baseline_path.exists():
            baseline_models = _normalize_baseline(load_json(baseline_path))
            if dense_models:
                plot_baseline_vs_advanced(baseline_models, dense_models, out_dir, title_suffix)

        if refinement_path.exists():
            refinement_data = load_json(refinement_path)
            plot_refinement(refinement_data, out_dir, title_suffix)

        if tuning_path.exists():
            report = load_json(tuning_path)
            plot_bm25_heatmap(report.get("grid_results", []), out_dir, title_suffix)
            tuning_best = report.get("best_params", {})

        if dense_models or refinement_data:
            plot_summary_table(dense_models, refinement_data, tuning_best, out_dir, title_suffix)

        generated.append(str(out_dir))
        logger.info(f"Charts for run_{run_label} → {out_dir}")

    return generated


def main():
    parser = argparse.ArgumentParser(description="Plot IR evaluation charts")
    parser.add_argument("--run", default=None, help="Run label e.g. 200000 or full")
    args = parser.parse_args()

    paths = generate_charts(args.run)
    if paths:
        print("\nGenerated charts in:")
        for p in paths:
            print(f"  {p}")
    else:
        print("No charts generated. Run evaluations first or check results/ folder.")


if __name__ == "__main__":
    main()
