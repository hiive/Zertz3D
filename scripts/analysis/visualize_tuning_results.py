#!/usr/bin/env python3
"""Visualize MCTS tuning results with both static and interactive outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_MPL_CACHE = Path("analysis_output/mpl_cache")
if "MPLCONFIGDIR" not in os.environ:
    DEFAULT_MPL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(DEFAULT_MPL_CACHE)

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from pandas.api import types as ptypes
import seaborn as sns
from matplotlib import pyplot as plt

try:
    import plotly.express as px
except ImportError:  # pragma: no cover - optional dependency
    px = None


def load_results(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        payload: Any = json.load(handle)

    if isinstance(payload, dict) and "results" in payload:
        records = payload["results"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"Unrecognised JSON structure in {path}")

    df = pd.json_normalize(records)
    if "hyperparams.exploration_constant" not in df.columns:
        raise ValueError("Expected flattened hyperparameters in the JSON input.")

    # Promote hyperparameters to dedicated columns for convenience.
    df = df.rename(
        columns=lambda name: name.replace("hyperparams.", "")
        if name.startswith("hyperparams.")
        else name
    )
    df["fpu_label"] = df["fpu_reduction"].apply(
        lambda value: "None" if pd.isna(value) else value
    )
    df["widening_label"] = df["widening_constant"].apply(
        lambda value: "None" if pd.isna(value) else value
    )
    df["rave_label"] = df["rave_constant"].apply(
        lambda value: "None" if pd.isna(value) else value
    )
    df["fpu_label"] = df["fpu_label"].astype(str)
    df["widening_label"] = df["widening_label"].astype(str)
    df["rave_label"] = df["rave_label"].astype(str)

    # Convenience numeric columns for plotting/colour mapping.
    df["fpu_numeric"] = df["fpu_reduction"].fillna(0.0)
    df["widening_numeric"] = df["widening_constant"].fillna(0.0)

    # Sort for stable plotting.
    df = df.sort_values(
        ["exploration_constant", "fpu_label", "widening_label", "rave_label"]
    ).reset_index(drop=True)
    return df


def _bucket_series(
    series: pd.Series, max_bins: int = 8
) -> tuple[pd.Series, list[str]]:
    """Coerce a series to coarse bins so sparse random sweeps still form a grid."""
    if ptypes.is_numeric_dtype(series) and series.nunique() > max_bins:
        binned = pd.cut(series, bins=max_bins, duplicates="drop")
        labels = binned.astype(str)
        order = [str(cat) for cat in binned.cat.categories]
        return labels, order

    labels = series.astype(str)
    order = sorted(labels.unique(), key=lambda value: labels[labels == value].index[0])
    return labels, order


def plot_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    pivot = (
        df[df["rave_constant"].isna()]
        .pivot_table(
            index="exploration_constant",
            columns="widening_constant",
            values="win_rate",
            aggfunc="mean",
        )
        .sort_index()
    )

    plt.figure(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "Mean win rate"},
    )
    plt.title("Mean win rate (RAVE disabled)")
    plt.xlabel("Progressive widening constant")
    plt.ylabel("Exploration constant")
    plt.tight_layout()
    path = output_dir / "heatmap_win_rate.png"
    plt.savefig(path, dpi=200)
    plt.close()


def plot_rave_box(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="rave_label", y="win_rate")
    plt.title("Win rate distribution by RAVE constant")
    plt.xlabel("RAVE constant (None = disabled)")
    plt.ylabel("Win rate")
    plt.tight_layout()
    path = output_dir / "rave_boxplot.png"
    plt.savefig(path, dpi=200)
    plt.close()


def plot_pareto(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="mean_time_per_game",
        y="win_rate",
        hue="exploration_constant",
        style="rave_label",
        palette="viridis",
    )
    plt.title("Win rate vs average game time")
    plt.xlabel("Mean time per game (s)")
    plt.ylabel("Win rate")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    path = output_dir / "pareto_scatter.png"
    plt.savefig(path, dpi=200)
    plt.close()


def normalise(series: pd.Series) -> pd.Series:
    minimum = series.min()
    maximum = series.max()
    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - minimum) / (maximum - minimum)


def plot_pareto_dark(df: pd.DataFrame, output_dir: Path) -> None:
    r = normalise(df["exploration_constant"])
    g = normalise(df["widening_numeric"])
    b = normalise(df["fpu_numeric"])
    colors = list(zip(r, g, b))
    symbols = df["rave_label"].astype(str)
    unique_symbols = symbols.unique()
    marker_cycle = ["o", "s", "^", "D", "P", "X", "*", "v", ">", "<", "h"]
    symbol_map = {
        value: marker_cycle[idx % len(marker_cycle)]
        for idx, value in enumerate(unique_symbols)
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    for value, marker in symbol_map.items():
        mask = symbols == value
        ax.scatter(
            df.loc[mask, "mean_time_per_game"],
            df.loc[mask, "win_rate"],
            c=[colors[i] for i in df.index[mask]],
            edgecolors="white",
            linewidths=0.6,
            s=50,
            marker=marker,
            label=f"RAVE {value}",
        )

    ax.set_title("Win rate vs average game time (RGB by hyperparameter)", color="white")
    ax.set_xlabel("Mean time per game (s)", color="white")
    ax.set_ylabel("Win rate", color="white")

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    ax.grid(False)

    ax.text(
        0.02,
        0.02,
        "Color key:\nR ↑ exploration constant\nG ↑ widening constant\nB ↑ FPU reduction",
        transform=ax.transAxes,
        color="white",
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="black", edgecolor="white", linewidth=0.3, alpha=0.7, boxstyle="round,pad=0.3"),
    )

    legend = ax.legend(
        title="RAVE constant",
        loc="lower right",
        facecolor="black",
        edgecolor="white",
        framealpha=0.8,
        fontsize=8,
        title_fontsize=10,
    )
    for text in legend.get_texts():
        text.set_color("white")
    legend.get_title().set_color("white")

    path = output_dir / "pareto_scatter_dark.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def _setup_dark_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor("black")
    ax.set_title(title, color="white")
    ax.set_xlabel(xlabel, color="white")
    ax.set_ylabel(ylabel, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")


def plot_dark_heatmaps(df: pd.DataFrame, output_dir: Path) -> None:
    base_df = df.copy()
    base_df["fpu_label"] = base_df["fpu_reduction"].apply(
        lambda value: "None" if pd.isna(value) else value
    )
    base_df["widening_label"] = base_df["widening_constant"].apply(
        lambda value: "None" if pd.isna(value) else value
    )
    base_df["rave_label"] = base_df["rave_constant"].apply(
        lambda value: "None" if pd.isna(value) else value
    )
    base_df["fpu_label"] = base_df["fpu_label"].astype(str)
    base_df["widening_label"] = base_df["widening_label"].astype(str)
    base_df["rave_label"] = base_df["rave_label"].astype(str)

    comparisons = [
        (
            ("exploration_constant", "widening_label"),
            base_df[base_df["rave_label"] == "None"],
            "Win rate (RAVE disabled)",
            "Progressive widening constant",
        ),
        (
            ("exploration_constant", "fpu_label"),
            base_df[base_df["rave_label"] == "None"],
            "Win rate vs FPU (RAVE disabled)",
            "FPU reduction",
        ),
        (
            ("fpu_label", "widening_label"),
            base_df[base_df["rave_label"] == "None"],
            "Win rate vs FPU/Widening (RAVE disabled)",
            "Progressive widening constant",
        ),
        (
            ("exploration_constant", "rave_label"),
            base_df,
            "Win rate vs RAVE constant",
            "RAVE constant",
        ),
    ]

    for (row_key, col_key), subset, title, col_label in comparisons:
        if subset.empty:
            continue

        row_labels, row_order = _bucket_series(subset[row_key])
        col_labels, col_order = _bucket_series(subset[col_key])
        working = subset.assign(__row=row_labels, __col=col_labels)

        pivot = working.pivot_table(
            index="__row",
            columns="__col",
            values="win_rate",
            aggfunc="mean",
        )
        pivot = pivot.reindex(index=row_order, columns=col_order)

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("black")
        _setup_dark_axes(
            ax,
            title,
            col_label,
            row_key.replace("_", " ").title(),
        )

        hm = sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="magma",
            cbar=True,
            cbar_kws={"label": "Mean win rate"},
            linewidths=0.2,
            linecolor="white",
            mask=pivot.isna(),
        )
        hm.figure.axes[-1].yaxis.label.set_color("white")
        hm.figure.axes[-1].tick_params(colors="white")

        plt.tight_layout()
        filename = f"heatmap_dark_{row_key}_vs_{col_key}.png"
        plt.savefig(output_dir / filename, dpi=200, facecolor="black")
        plt.close()


def plot_interactive(df: pd.DataFrame, output_dir: Path) -> Path | None:
    if px is None:
        return None

    fig = px.scatter(
        df,
        x="mean_time_per_game",
        y="win_rate",
        color="exploration_constant",
        symbol="rave_label",
        hover_data=[
            "exploration_constant",
            "fpu_reduction",
            "widening_constant",
            "rave_label",
            "wins",
            "losses",
            "ties",
        ],
        title="Win rate vs average game time",
    )
    fig.update_layout(
        legend_title_text="Exploration constant",
        xaxis_title="Mean time per game (s)",
        yaxis_title="Win rate",
    )

    output_path = output_dir / "pareto_scatter.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visual summaries of MCTS tuning runs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/tuning/tuning_results.json"),
        help="Path to tuning results JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_output/tuning_plots"),
        help="Directory for saved visualisations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input)

    plot_heatmap(df, output_dir)
    plot_rave_box(df, output_dir)
    plot_pareto(df, output_dir)
    plot_pareto_dark(df, output_dir)
    plot_dark_heatmaps(df, output_dir)

    interactive_path = plot_interactive(df, output_dir)

    print(f"Saved static figures to {output_dir}")
    if interactive_path is not None:
        print(f"Interactive HTML written to {interactive_path}")
    else:
        print("Plotly not available; skipped interactive output.")


if __name__ == "__main__":
    main()
