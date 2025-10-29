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
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional dependency
    px = None
    go = None


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


def _format_float(value: float) -> str:
    """Format a float concisely for axis display."""
    text = f"{value:.3f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _bucket_series(series: pd.Series, max_bins: int = 8) -> tuple[pd.Series, list[str]]:
    """Coerce a numeric series into coarse bins for readable heatmap axes."""
    numeric = pd.to_numeric(series, errors="coerce")
    labels = pd.Series(index=series.index, dtype=object)

    if numeric.notna().sum() and numeric.nunique() > max_bins:
        binned = pd.cut(numeric, bins=max_bins, duplicates="drop")
        mapping = {
            cat: f"{_format_float(cat.left)}–{_format_float(cat.right)}"
            for cat in binned.cat.categories
        }
        labels = binned.map(mapping)
    elif numeric.notna().sum():
        mapping = {
            val: _format_float(val) for val in sorted(numeric.dropna().unique().tolist())
        }
        labels = numeric.map(mapping)
    else:
        labels = series.astype(object)

    labels = pd.Series(labels, index=series.index, dtype="object")
    labels = labels.where(pd.notna(labels), "None")
    order: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label not in seen:
            order.append(label)
            seen.add(label)
    return labels.astype(str), order


def plot_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Baseline heatmap (light theme) for exploration vs widening."""
    subset = df[df["rave_label"] == "None"]
    if subset.empty:
        subset = df

    pivot = subset.pivot_table(
        index="exploration_constant",
        columns="widening_constant",
        values="win_rate",
        aggfunc="mean",
    ).sort_index()

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
        0.02, # df['win_rate'].max() - 0.01,
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
        {
            "title": "Win rate (RAVE disabled)",
            "subset": base_df[base_df["rave_label"] == "None"],
            "row": ("exploration_constant", "exploration_constant", "Exploration constant"),
            "col": ("widening_label", "widening_constant", "Progressive widening constant"),
        },
        {
            "title": "Win rate vs FPU (RAVE disabled)",
            "subset": base_df[base_df["rave_label"] == "None"],
            "row": ("exploration_constant", "exploration_constant", "Exploration constant"),
            "col": ("fpu_label", "fpu_reduction", "FPU reduction"),
        },
        {
            "title": "Win rate vs FPU/Widening (RAVE disabled)",
            "subset": base_df[base_df["rave_label"] == "None"],
            "row": ("fpu_label", "fpu_reduction", "FPU reduction"),
            "col": ("widening_label", "widening_constant", "Progressive widening constant"),
        },
        {
            "title": "Win rate vs RAVE constant",
            "subset": base_df,
            "row": ("exploration_constant", "exploration_constant", "Exploration constant"),
            "col": ("rave_label", "rave_constant", "RAVE constant"),
        },
    ]

    for spec in comparisons:
        subset = spec["subset"]
        if subset.empty:
            continue

        row_key, row_numeric, row_title = spec["row"]
        col_key, col_numeric, col_title = spec["col"]

        row_source = subset[row_numeric] if row_numeric else subset[row_key]
        col_source = subset[col_numeric] if col_numeric else subset[col_key]

        row_labels, row_order = _bucket_series(row_source)
        col_labels, col_order = _bucket_series(col_source)
        working = subset.assign(__row=row_labels, __col=col_labels)

        pivot = working.pivot_table(
            index="__row",
            columns="__col",
            values="win_rate",
            aggfunc="mean",
        )
        pivot = pivot.reindex(index=row_order, columns=col_order)
        pivot = pivot.dropna(how="all", axis=0).dropna(how="all", axis=1)
        if pivot.empty:
            continue
        row_order = [label for label in row_order if label in pivot.index]
        col_order = [label for label in col_order if label in pivot.columns]
        pivot = pivot.reindex(index=row_order, columns=col_order)
        pivot.index.name = row_title
        pivot.columns.name = col_title

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("black")
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
            ax=ax,
        )
        ax.set_title(spec["title"], color="white")
        ax.set_xlabel(col_title, color="white")
        ax.set_ylabel(row_title, color="white")
        ax.tick_params(colors="white")
        ax.tick_params(axis="x", labelrotation=45)
        for spine in ax.spines.values():
            spine.set_color("white")

        if hm and hm.collections:
            cbar = hm.collections[0].colorbar
            cbar.set_label("Mean win rate", color="white")
            cbar.ax.tick_params(colors="white")
            for label in cbar.ax.get_yticklabels():
                label.set_color("white")

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


def plot_hyperparam_scatter_3d(df: pd.DataFrame, output_dir: Path) -> None:
    """Render a dark-themed 3D scatter of the hyperparameter cube."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    rave_markers = {
        label: marker
        for label, marker in zip(
            df["rave_label"].unique(),
            ["o", "^", "s", "D", "P", "X", "v", "*", ">"],
        )
    }

    norm = plt.Normalize(df["win_rate"].min(), df["win_rate"].max())
    cmap = plt.cm.magma

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    for label, subset in df.groupby("rave_label"):
        ax.scatter(
            subset["fpu_reduction"].fillna(0.0),
            subset["widening_constant"].fillna(0.0),
            subset["exploration_constant"],
            c=cmap(norm(subset["win_rate"])),
            edgecolor="white",
            linewidths=0.3,
            marker=rave_markers.get(label, "o"),
            label=f"RAVE {label}",
            s=28,
        )

    ax.set_xlabel("FPU reduction", color="white")
    ax.set_ylabel("Widening constant", color="white")
    ax.set_zlabel("Exploration constant", color="white")
    ax.set_title("Win rate across hyperparameter cube", color="white")
    ax.tick_params(colors="white")

    panes = []
    for axis in (getattr(ax, "xaxis", None), getattr(ax, "yaxis", None), getattr(ax, "zaxis", None)):
        pane = getattr(axis, "pane", None)
        if pane is not None:
            panes.append(pane)
    for pane in panes:
        pane.set_facecolor((0.05, 0.05, 0.05, 0.7))
        pane.set_edgecolor("white")

    legend = ax.legend(loc="upper left")
    for text in legend.get_texts():
        text.set_color("white")

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("Win rate", color="white")
    cbar.ax.tick_params(colors="white")

    plt.tight_layout()
    (output_dir / "hyperparams_scatter_3d.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "hyperparams_scatter_3d.png", dpi=200, facecolor="black")
    plt.close(fig)


def plot_isosurface(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Optionally emit an isosurface plotly figure if plotly is available."""
    if go is None:
        return None

    required = {"fpu_reduction", "widening_constant", "exploration_constant"}
    if not required.issubset(df.columns):
        return None

    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=df["fpu_reduction"].fillna(0.0),
            y=df["widening_constant"].fillna(0.0),
            z=df["exploration_constant"],
            mode="markers",
            marker=dict(
                size=4,
                color=df["win_rate"],
                colorscale="Magma",
                opacity=0.7,
                colorbar=dict(title="Win rate"),
            ),
            text=[f"RAVE {label}" for label in df["rave_label"]],
        )
    )

    try:
        figure.add_trace(
            go.Isosurface(
                x=df["fpu_reduction"].fillna(0.0),
                y=df["widening_constant"].fillna(0.0),
                z=df["exploration_constant"],
                value=df["win_rate"],
                isomin=df["win_rate"].quantile(0.75),
                isomax=df["win_rate"].max(),
                opacity=0.2,
                surface_count=3,
                colorscale="Magma",
                showscale=False,
            )
        )
    except Exception:
        # Plotly may fail if the data are too sparse for an isosurface;
        # continue gracefully with just the scatter trace.
        pass

    figure.update_layout(
        title="Hyperparameter isosurface (interactive)",
        scene=dict(
            xaxis_title="FPU reduction",
            yaxis_title="Widening constant",
            zaxis_title="Exploration constant",
            bgcolor="black",
        ),
        paper_bgcolor="black",
        font=dict(color="white"),
    )

    output_path = output_dir / "hyperparams_isosurface.html"
    figure.write_html(output_path, include_plotlyjs="cdn")
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
    plot_hyperparam_scatter_3d(df, output_dir)

    interactive_path = plot_interactive(df, output_dir)
    isosurface_path = plot_isosurface(df, output_dir)

    print(f"Saved static figures to {output_dir}")
    if interactive_path is not None:
        print(f"Interactive HTML written to {interactive_path}")
    else:
        print("Plotly not available; skipped interactive output.")
    if isosurface_path is not None:
        print(f"Isosurface HTML written to {isosurface_path}")
    elif go is None:
        print("Plotly graph_objects not available; skipped isosurface plot.")


if __name__ == "__main__":
    main()
