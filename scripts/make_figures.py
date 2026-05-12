from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

from refract.datasets.constants import ALL_DATASETS, BINARY_DATASETS

warnings.filterwarnings("ignore")

app = typer.Typer(pretty_exceptions_enable=False)

_MODEL_ORDER = ["refract", "mlp", "gcn", "graphsage", "gat", "appnp"]
_MODEL_LABELS = {
    "refract": "Refract", "mlp": "MLP", "gcn": "GCN",
    "graphsage": "GraphSAGE", "gat": "GAT", "appnp": "APPNP",
}
_ABLATION_VARIANTS = ["full", "no_align", "no_smooth", "no_entropy", "no_aux"]
_ABLATION_LABELS = {
    "full": "Full", "no_align": "−align", "no_smooth": "−smooth",
    "no_entropy": "−entropy", "no_aux": "−aux",
}
_REGIME_ORDER = ["attribute_dominant", "topology_dominant", "concordant", "conflict", "uncertain"]
_REGIME_LABELS_SHORT = {
    "attribute_dominant": "Attr.", "topology_dominant": "Topo.",
    "concordant": "Conc.", "conflict": "Conf.", "uncertain": "Unc.",
}


def _neurips_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
    })


def _palette(n: int) -> list:
    return list(sns.color_palette("colorblind", n))


def _load_results(results_dir: Path, datasets: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(results_dir / f"{ds}.csv") for ds in datasets if (results_dir / f"{ds}.csv").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_history(results_dir: Path, datasets: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(results_dir / f"{ds}_history.parquet") for ds in datasets if (results_dir / f"{ds}_history.parquet").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_gate(results_dir: Path, datasets: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(results_dir / f"{ds}_gate.parquet") for ds in datasets if (results_dir / f"{ds}_gate.parquet").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_ablations(results_dir: Path, datasets: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(results_dir / f"{ds}_ablations.csv") for ds in datasets if (results_dir / f"{ds}_ablations.csv").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _import_legacy(results_dir: Path, legacy_dir: Path) -> None:
    for csv_path in legacy_dir.glob("*.csv"):
        dest = results_dir / csv_path.name
        if dest.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            if "seed" not in df.columns:
                df["seed"] = 0
            df.to_csv(dest, index=False)
            typer.echo(f"[legacy] Imported {csv_path.name} → {dest}")
        except Exception as e:
            typer.echo(f"[legacy] Failed {csv_path.name}: {e}", err=True)


def _split_mean_std(df: pd.DataFrame, metric_col: str) -> tuple[float, float]:
    per_split = df.groupby("split")[metric_col].mean()
    return float(per_split.mean()), float(per_split.std())


def _subplot_grid(n: int, ncols: int) -> tuple[plt.Figure, np.ndarray]:
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 2.4))
    if nrows * ncols == 1:
        return fig, np.array([[axes]])
    if nrows == 1:
        return fig, axes.reshape(1, -1)
    if ncols == 1:
        return fig, axes.reshape(-1, 1)
    return fig, axes


def fig_training_curves(history_df: pd.DataFrame, out_fig: Path, datasets: list[str]) -> None:
    if history_df.empty or "dataset" not in history_df.columns:
        typer.echo("[figures] No history data — skipping training curves")
        return
    avail = [d for d in datasets if d in history_df["dataset"].values]
    if not avail:
        return

    palette = _palette(len(_MODEL_ORDER))
    model_colors = dict(zip(_MODEL_ORDER, palette))
    fig, axes = _subplot_grid(len(avail), ncols=min(4, len(avail)))

    for idx, ds in enumerate(avail):
        ax = axes[idx // axes.shape[1]][idx % axes.shape[1]]
        sub = history_df[history_df["dataset"] == ds]
        use_roc = ds in BINARY_DATASETS and "val_roc" in sub.columns and sub["val_roc"].notna().any()
        metric_col = "val_roc" if use_roc else "val_acc"
        ylabel = "Val ROC-AUC (%)" if use_roc else "Val Accuracy (%)"
        max_epoch = int(sub["epoch"].max())

        for model in _MODEL_ORDER:
            msub = sub[sub["model"] == model]
            if msub.empty:
                continue
            msub_metric = msub.dropna(subset=[metric_col])
            if msub_metric.empty:
                continue
            msub_metric = msub_metric.copy()
            msub_metric[metric_col] = pd.to_numeric(msub_metric[metric_col], errors="coerce")
            msub_metric = msub_metric.dropna(subset=[metric_col])
            if msub_metric.empty:
                continue
            grouped = msub_metric.groupby("epoch")[metric_col]
            epochs = np.array(sorted(grouped.groups.keys()))
            means = grouped.mean().values.astype(float) * 100
            stds = grouped.std().fillna(0).values.astype(float) * 100
            lw = 1.6 if model == "refract" else 0.9
            zorder = 3 if model == "refract" else 2
            ax.plot(epochs, means, color=model_colors[model], lw=lw, zorder=zorder,
                    label=_MODEL_LABELS[model])
            ax.fill_between(epochs, means - stds, means + stds,
                            color=model_colors[model], alpha=0.12, zorder=zorder - 1)

        ax.set_title(ds.replace("-", "‑"), pad=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel if idx % axes.shape[1] == 0 else "")
        ax.set_xlim(0, max_epoch)

    for idx in range(len(avail), axes.size):
        axes.flat[idx].set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(_MODEL_ORDER),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"[figures] Training curves → {out_fig}")


def fig_main_table(results_df: pd.DataFrame, out_tab: Path, datasets: list[str]) -> None:
    if results_df.empty or "dataset" not in results_df.columns:
        typer.echo("[figures] No results data — skipping main table")
        return
    avail = [d for d in datasets if d in results_df["dataset"].values]
    if not avail:
        return

    rows_tex = []
    for ds in avail:
        sub = results_df[results_df["dataset"] == ds]
        use_roc = ds in BINARY_DATASETS
        metric_col = "roc_auc" if use_roc else "accuracy"

        best_mean = -1.0
        best_model = ""
        model_stats: dict[str, tuple[float, float]] = {}
        for model in _MODEL_ORDER:
            msub = sub[sub["model"] == model][[metric_col, "split"]].dropna()
            if msub.empty:
                model_stats[model] = (float("nan"), float("nan"))
                continue
            m, s = _split_mean_std(msub, metric_col)
            model_stats[model] = (m, s)
            if m > best_mean:
                best_mean = m
                best_model = model

        cells = []
        for model in _MODEL_ORDER:
            m, s = model_stats[model]
            if np.isnan(m):
                cells.append("—")
            else:
                cell = f"{m*100:.1f}$\\pm${s*100:.1f}"
                cells.append(f"\\textbf{{{cell}}}" if model == best_model else cell)

        dagger = "$^\\dagger$" if use_roc else ""
        rows_tex.append(f"  {ds}{dagger} & " + " & ".join(cells) + " \\\\")

    col_headers = " & ".join(f"\\textbf{{{_MODEL_LABELS[m]}}}" for m in _MODEL_ORDER)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Test performance (mean$\\pm$std \\% across 10 splits). $^\\dagger$ROC-AUC; others: accuracy.}",
        "\\label{tab:main_results}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l" + "c" * len(_MODEL_ORDER) + "}",
        "\\toprule",
        f"  Dataset & {col_headers} \\\\",
        "\\midrule",
    ] + rows_tex + [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]

    out_tab.parent.mkdir(parents=True, exist_ok=True)
    out_tab.write_text("\n".join(lines) + "\n")
    typer.echo(f"[figures] Main table   → {out_tab}")


def fig_ablation(ablation_df: pd.DataFrame, out_fig: Path) -> None:
    if ablation_df.empty or "dataset" not in ablation_df.columns:
        typer.echo("[figures] No ablation data — skipping ablation figure")
        return
    avail_ds = ablation_df["dataset"].unique().tolist()
    if not avail_ds:
        return

    n_ds = len(avail_ds)
    n_vars = len(_ABLATION_VARIANTS)
    palette = _palette(n_vars)
    group_width = 0.8
    bar_w = group_width / n_vars
    x = np.arange(n_ds)

    full_means: dict[str, float] = {}
    for ds in avail_ds:
        metric_col = "roc_auc" if ds in BINARY_DATASETS else "accuracy"
        vals = ablation_df[(ablation_df["dataset"] == ds) & (ablation_df["variant"] == "full")][metric_col].dropna().astype(float)
        full_means[ds] = float(vals.mean()) if not vals.empty else float("nan")

    fig, ax = plt.subplots(figsize=(max(4.0, n_ds * 1.8), 3.2))
    for v_idx, variant in enumerate(_ABLATION_VARIANTS):
        means = []
        for ds in avail_ds:
            metric_col = "roc_auc" if ds in BINARY_DATASETS else "accuracy"
            vals = ablation_df[(ablation_df["dataset"] == ds) & (ablation_df["variant"] == variant)][metric_col].dropna().astype(float)
            means.append(float(vals.mean()) * 100 if not vals.empty else float("nan"))
        offsets = x - group_width / 2 + (v_idx + 0.5) * bar_w
        bars = ax.bar(offsets, means, width=bar_w * 0.9, color=palette[v_idx],
                      label=_ABLATION_LABELS[variant], zorder=2)
        if variant != "full":
            for bar, ds, m in zip(bars, avail_ds, means):
                full = full_means[ds] * 100
                if not (np.isnan(m) or np.isnan(full)):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                            f"{m - full:+.1f}", ha="center", va="bottom", fontsize=5.5, color="0.3")

    ax.set_xticks(x)
    ax.set_xticklabels(avail_ds, rotation=20, ha="right")
    ax.set_ylabel("Primary metric (%)")
    ax.set_title("Loss-component ablation")
    ax.legend(frameon=False, ncol=n_vars, loc="lower right")
    ax.yaxis.grid(True, lw=0.4, color="0.85", zorder=0)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"[figures] Ablation     → {out_fig}")


def fig_main_bar(results_df: pd.DataFrame, out_fig: Path, datasets: list[str]) -> None:
    if results_df.empty or "dataset" not in results_df.columns:
        typer.echo("[figures] No results data — skipping main bar chart")
        return
    avail = [d for d in datasets if d in results_df["dataset"].values]
    if not avail:
        return

    n_ds = len(avail)
    n_models = len(_MODEL_ORDER)
    palette = _palette(n_models)
    model_colors = dict(zip(_MODEL_ORDER, palette))
    group_width = 0.8
    bar_w = group_width / n_models
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(max(5.0, n_ds * 1.4), 3.4))

    for m_idx, model in enumerate(_MODEL_ORDER):
        means, errs = [], []
        for ds in avail:
            metric_col = "roc_auc" if ds in BINARY_DATASETS else "accuracy"
            msub = results_df[(results_df["dataset"] == ds) & (results_df["model"] == model)][[metric_col, "split"]].dropna()
            if msub.empty:
                means.append(float("nan"))
                errs.append(0.0)
            else:
                m, s = _split_mean_std(msub, metric_col)
                means.append(m * 100)
                errs.append(s * 100)
        offsets = x - group_width / 2 + (m_idx + 0.5) * bar_w
        hatch = "//" if model == "refract" else None
        lw = 0.8 if model == "refract" else 0.4
        ax.bar(offsets, means, width=bar_w * 0.9, color=model_colors[model],
               label=_MODEL_LABELS[model], hatch=hatch, linewidth=lw,
               edgecolor="white" if model != "refract" else "k", zorder=2)
        ax.errorbar(offsets, means, yerr=errs, fmt="none", color="0.3",
                    capsize=1.5, linewidth=0.6, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("-", "‑") for d in avail], rotation=20, ha="right")
    ax.set_ylabel("Primary metric (%)")
    ax.yaxis.grid(True, lw=0.4, color="0.85", zorder=0)
    ax.legend(frameon=False, ncol=n_models, loc="upper center",
              bbox_to_anchor=(0.5, 1.12), columnspacing=0.8)

    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"[figures] Main bar     → {out_fig}")


def fig_routing_heatmap(gate_df: pd.DataFrame, out_fig: Path, datasets: list[str]) -> None:
    if gate_df.empty or "dataset" not in gate_df.columns:
        typer.echo("[figures] No gate data — skipping routing heatmap")
        return
    avail = [d for d in datasets if d in gate_df["dataset"].values]
    if not avail:
        return

    pivot = (
        gate_df[gate_df["dataset"].isin(avail)]
        .groupby(["dataset", "regime_label"])["gate_attr"]
        .median()
        .unstack("regime_label")
        .reindex(index=avail, columns=_REGIME_ORDER)
    )

    fig, ax = plt.subplots(figsize=(len(_REGIME_ORDER) * 1.2, len(avail) * 0.65 + 0.6))
    im = ax.imshow(pivot.values.astype(float), aspect="auto", vmin=0.0, vmax=1.0,
                   cmap="RdBu_r", origin="upper")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val - 0.5) > 0.25 else "k"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=text_color)

    ax.set_xticks(range(len(_REGIME_ORDER)))
    ax.set_xticklabels([_REGIME_LABELS_SHORT[r] for r in _REGIME_ORDER])
    ax.set_yticks(range(len(avail)))
    ax.set_yticklabels([d.replace("-", "‑") for d in avail])
    ax.set_xlabel("Node regime type")
    ax.set_title("Median gate weight (attr.) per regime", pad=4)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("← topology      attribute →", fontsize=6)
    cbar.set_ticks([0, 0.5, 1])
    cbar.ax.tick_params(labelsize=6)

    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"[figures] Routing heatmap → {out_fig}")


def fig_regime_distribution(gate_df: pd.DataFrame, out_fig: Path, datasets: list[str]) -> None:
    if gate_df.empty or "dataset" not in gate_df.columns:
        typer.echo("[figures] No gate data — skipping regime distribution")
        return
    avail = [d for d in datasets if d in gate_df["dataset"].values]
    if not avail:
        return

    regime_palette = dict(zip(_REGIME_ORDER, _palette(len(_REGIME_ORDER))))
    counts = (
        gate_df[gate_df["dataset"].isin(avail)]
        .groupby(["dataset", "regime_label"])["node_idx"]
        .count()
        .unstack("regime_label")
        .reindex(index=avail, columns=_REGIME_ORDER)
        .fillna(0)
    )
    fracs = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(max(4.0, len(avail) * 1.0), 3.0))
    bottom = np.zeros(len(avail))
    x = np.arange(len(avail))
    for regime in _REGIME_ORDER:
        vals = fracs[regime].values if regime in fracs.columns else np.zeros(len(avail))
        ax.bar(x, vals, bottom=bottom, color=regime_palette[regime],
               label=_REGIME_LABELS_SHORT[regime], width=0.7, zorder=2)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("-", "‑") for d in avail], rotation=20, ha="right")
    ax.set_ylabel("Fraction of nodes")
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, lw=0.4, color="0.85", zorder=0)
    ax.set_title("Node regime composition per dataset", pad=4)
    fig.legend(frameon=False, ncol=len(_REGIME_ORDER), loc="lower center",
               bbox_to_anchor=(0.5, -0.04), columnspacing=0.8)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"[figures] Regime dist. → {out_fig}")


def fig_routing_overview(gate_df: pd.DataFrame, out_fig: Path) -> None:
    if gate_df.empty or "dataset" not in gate_df.columns:
        typer.echo("[figures] No gate data — skipping routing overview")
        return
    avail_ds = [ds for ds in gate_df["dataset"].unique() if not gate_df[gate_df["dataset"] == ds]["gate_attr"].empty]
    if not avail_ds:
        return

    palette = _palette(len(avail_ds))
    fig, ax = plt.subplots(figsize=(max(4.0, len(avail_ds) * 1.1), 3.0))

    data = [gate_df[gate_df["dataset"] == ds]["gate_attr"].dropna().values for ds in avail_ds]
    vp = ax.violinplot(data, positions=range(len(avail_ds)), showmedians=True, showextrema=False)
    for body, color in zip(vp["bodies"], palette):
        body.set_facecolor(color)
        body.set_alpha(0.75)
    vp["cmedians"].set_color("k")
    vp["cmedians"].set_linewidth(1.2)

    ax.set_xticks(range(len(avail_ds)))
    ax.set_xticklabels([ds.replace("-", "‑") for ds in avail_ds], rotation=20, ha="right")
    ax.set_ylabel("Gate weight (attr.)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="0.7", lw=0.7, ls="--", zorder=0)
    ax.set_title("Per-dataset gate routing (all nodes)", pad=4)
    ax.yaxis.grid(True, lw=0.4, color="0.85", zorder=0)

    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"[figures] Routing overview → {out_fig}")


def fig_routing_analysis(gate_df: pd.DataFrame, out_fig: Path) -> None:
    if gate_df.empty or "dataset" not in gate_df.columns:
        typer.echo("[figures] No gate data — skipping routing analysis")
        return
    avail_ds = gate_df["dataset"].unique().tolist()
    if not avail_ds:
        return

    regime_palette = dict(zip(_REGIME_ORDER, _palette(len(_REGIME_ORDER))))
    fig, axes = _subplot_grid(len(avail_ds), ncols=min(3, len(avail_ds)))

    for idx, ds in enumerate(avail_ds):
        ax = axes[idx // axes.shape[1]][idx % axes.shape[1]]
        sub = gate_df[gate_df["dataset"] == ds]
        present = [r for r in _REGIME_ORDER if r in sub["regime_label"].values]
        data = [sub[sub["regime_label"] == r]["gate_attr"].values for r in present]
        colors = [regime_palette[r] for r in present]

        vp = ax.violinplot(data, positions=range(len(present)), showmedians=True, showextrema=False)
        for body, color in zip(vp["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.75)
        vp["cmedians"].set_color("k")
        vp["cmedians"].set_linewidth(1.2)

        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([_REGIME_LABELS_SHORT[r] for r in present], fontsize=7)
        ax.set_ylabel("Gate weight (attr.)" if idx % axes.shape[1] == 0 else "")
        ax.set_title(ds.replace("-", "‑"), pad=3)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="0.7", lw=0.7, ls="--", zorder=0)

    for idx in range(len(avail_ds), axes.size):
        axes.flat[idx].set_visible(False)

    fig.suptitle("Gate routing by regime type", y=1.01, fontsize=9)
    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"[figures] Routing      → {out_fig}")


@app.command()
def run(
    results_dir: str = typer.Option("results/benchmarks", help="Directory containing eval outputs"),
    out_fig: str = typer.Option("paper/figures", help="Output directory for PDF figures"),
    out_tab: str = typer.Option("paper/tables", help="Output directory for LaTeX tables"),
    datasets: str = typer.Option("all", help="all | comma-separated dataset names"),
    legacy_import: bool = typer.Option(False, "--legacy-import", help="Import artifacts/benchmark_v2/ CSVs"),
    legacy_dir: str = typer.Option("artifacts/benchmark_v2", help="Path to legacy benchmark CSVs"),
) -> None:
    _neurips_style()

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    if legacy_import:
        _import_legacy(results_path, Path(legacy_dir))

    ds_list = list(ALL_DATASETS) if datasets == "all" else [d.strip() for d in datasets.split(",")]

    typer.echo(f"[figures] Loading data from {results_path} …")
    results_df = _load_results(results_path, ds_list)
    history_df = _load_history(results_path, ds_list)
    gate_df = _load_gate(results_path, ds_list)
    ablation_df = _load_ablations(results_path, ds_list)

    fig_path = Path(out_fig)
    tab_path = Path(out_tab)

    fig_training_curves(history_df, fig_path / "training_curves.pdf", ds_list)
    fig_main_table(results_df, tab_path / "main_results.tex", ds_list)
    fig_main_bar(results_df, fig_path / "main_bar.pdf", ds_list)
    fig_routing_heatmap(gate_df, fig_path / "routing_heatmap.pdf", ds_list)
    fig_regime_distribution(gate_df, fig_path / "regime_distribution.pdf", ds_list)
    fig_routing_overview(gate_df, fig_path / "routing_overview.pdf")
    fig_routing_analysis(gate_df, fig_path / "routing_analysis.pdf")


if __name__ == "__main__":
    app()
