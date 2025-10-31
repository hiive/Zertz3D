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

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats as scipy_stats
from collections import defaultdict

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
    if "hyperparameters.exploration_constant" not in df.columns:
        raise ValueError("Expected flattened hyperparameters in the JSON input.")

    # Promote hyperparameters to dedicated columns for convenience.
    df = df.rename(
        columns=lambda name: name.replace("hyperparameters.", "")
        if name.startswith("hyperparameters.")
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


def _prepare_axis_values(
    series: pd.Series,
) -> tuple[np.ndarray, list[float] | None, list[str] | None, bool]:
    """Return numeric axis positions, ticks, labels, and whether the data are numeric."""
    numeric_series = pd.to_numeric(series, errors="coerce")
    has_numeric = numeric_series.notna().any()

    if has_numeric:
        values = numeric_series.copy()
        placeholder = None

        if series.isna().any():
            finite = numeric_series.dropna()
            if finite.empty:
                placeholder = 0.0
            else:
                min_val = float(finite.min())
                max_val = float(finite.max())
                span = max(max_val - min_val, 1.0)
                placeholder = min_val - 0.1 * span
            values = values.fillna(placeholder)
        else:
            values = values.fillna(0.0)

        unique_vals = np.unique(values)
        tick_positions = unique_vals.tolist() if unique_vals.size <= 12 else None
        tick_labels = (
            [
                "None" if placeholder is not None and val == placeholder else _format_float(val)
                for val in unique_vals
            ]
            if tick_positions is not None
            else None
        )
        return values.to_numpy(dtype=float), tick_positions, tick_labels, True

    categories = series.fillna("None").astype(str)
    mapping = {cat: idx for idx, cat in enumerate(categories.unique())}
    values = categories.map(mapping).to_numpy(dtype=float)
    tick_positions = list(mapping.values())
    tick_labels = list(mapping.keys())
    return values, tick_positions, tick_labels, False


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
            linewidths=0.3,
            s=10,
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
            cmap="turbo",
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


def plot_dark_pareto(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot win rate vs each hyperparameter on a dark background."""
    if df.empty:
        return

    columns = [
        ("exploration_constant", "Exploration constant"),
        ("fpu_reduction", "FPU reduction"),
        ("widening_constant", "Progressive widening constant"),
        ("rave_constant", "RAVE constant"),
    ]

    cmap = plt.cm.turbo
    norm = plt.Normalize(df["mean_time_per_game"].min(), df["mean_time_per_game"].max())

    for column, label in columns:
        if column not in df.columns:
            continue

        x_values, tick_positions, tick_labels, is_numeric = _prepare_axis_values(df[column])
        y_values = df["win_rate"].to_numpy(dtype=float)
        colors = cmap(norm(df["mean_time_per_game"]))

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        ax.scatter(
            x_values,
            y_values,
            c=colors,
            edgecolors="white",
            linewidths=0.4,
            s=36,
        )

        ax.set_title(f"Win rate vs {label}", color="white")
        ax.set_xlabel(label, color="white")
        ax.set_ylabel("Win rate", color="white")
        ax.tick_params(colors="white")

        if tick_positions is not None and tick_labels is not None:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=30, ha="right", color="white")
        elif not is_numeric:
            ax.set_xticks([])

        ax.grid(alpha=0.2, color="white")
        for spine in ax.spines.values():
            spine.set_color("white")

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Mean time per game (s)", color="white")
        cbar.ax.tick_params(colors="white")

        filename = output_dir / f"pareto_{column}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=200, facecolor="black")
        plt.close(fig)


def plot_interactive(df: pd.DataFrame, output_dir: Path) -> tuple[Path | None, Path | None]:
    if px is None:
        return None, None

    numeric_cols = ["exploration_constant", "fpu_reduction", "widening_constant", "rave_constant"]
    color_column = "win_rate"

    df_numeric = df.copy()
    for col in numeric_cols:
        if col not in df_numeric:
            return None
        df_numeric[col] = df_numeric[col].fillna(-1.0)

    parallel_fig = px.parallel_coordinates(
        df_numeric,
        dimensions=numeric_cols + [color_column],
        color=color_column,
        color_continuous_scale=px.colors.sequential.Turbo,
        labels={
            "exploration_constant": "Exploration",
            "fpu_reduction": "FPU",
            "widening_constant": "Widening",
            "rave_constant": "RAVE",
            "win_rate": "Win rate",
        },
    )
    parallel_fig.update_layout(
        title="Hyperparameter parallel coordinates",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
    )
    parallel_path = output_dir / "hyperparams_parallel_coordinates.html"
    parallel_fig.write_html(parallel_path, include_plotlyjs="cdn")

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
    return output_path, parallel_path


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
    cmap = plt.cm.turbo

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
                colorscale="Turbo",
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
                colorscale="Turbo",
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


def detect_repetition_mode(df: pd.DataFrame) -> dict[tuple, pd.DataFrame]:
    """Detect if data is from repetition mode by finding duplicate hyperparameter configurations.

    Returns:
        Dictionary mapping hyperparameter tuples to groups of results.
        Empty dict if not in repetition mode (all configs unique).
    """
    config_groups = defaultdict(list)

    for idx, row in df.iterrows():
        # Create a hashable key from hyperparameters
        key = (
            row["exploration_constant"],
            row.get("fpu_reduction"),
            row.get("max_simulation_depth"),
            row.get("widening_constant"),
            row.get("rave_constant")
        )
        config_groups[key].append(idx)

    # Only return groups if we have repetitions
    repetition_groups = {}
    for key, indices in config_groups.items():
        if len(indices) > 1:
            repetition_groups[key] = df.loc[indices]

    return repetition_groups


def plot_repetition_violin(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot violin plots showing win rate distribution for each configuration with individual points."""
    repetition_groups = detect_repetition_mode(df)

    if not repetition_groups:
        return

    # Create summary data for plotting
    plot_data = []
    for config_idx, (key, group_df) in enumerate(repetition_groups.items(), 1):
        exploration, fpu, depth, widening, rave = key
        config_label = f"Config {config_idx}\nExp={exploration:.2f}"

        for _, row in group_df.iterrows():
            plot_data.append({
                "config": config_label,
                "config_idx": config_idx,
                "win_rate": row["win_rate"],
                "exploration": exploration,
            })

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Create violin plot
    parts = ax.violinplot(
        [plot_df[plot_df["config_idx"] == i]["win_rate"].values
         for i in sorted(plot_df["config_idx"].unique())],
        positions=sorted(plot_df["config_idx"].unique()),
        widths=0.7,
        showmeans=True,
        showmedians=True,
    )

    # Style violin plot for dark mode
    for pc in parts["bodies"]:
        pc.set_facecolor("#8B008B")  # Dark magenta
        pc.set_edgecolor("white")
        pc.set_alpha(0.6)
        pc.set_linewidth(0.8)

    for partname in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
        if partname in parts:
            parts[partname].set_edgecolor("white")
            parts[partname].set_linewidth(1.2)

    # Overlay individual points with jitter
    for config_idx in sorted(plot_df["config_idx"].unique()):
        config_data = plot_df[plot_df["config_idx"] == config_idx]
        y_values = config_data["win_rate"].values
        x_values = config_idx + np.random.normal(0, 0.08, size=len(y_values))

        ax.scatter(
            x_values,
            y_values,
            c="cyan",
            s=20,
            alpha=0.5,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    ax.set_title("Win Rate Distribution by Configuration (Repetition Mode)", color="white", fontsize=14)
    ax.set_xlabel("Configuration", color="white")
    ax.set_ylabel("Win Rate", color="white")
    ax.tick_params(colors="white")

    # Set x-axis labels
    ax.set_xticks(sorted(plot_df["config_idx"].unique()))
    ax.set_xticklabels(
        [plot_df[plot_df["config_idx"] == i]["config"].iloc[0]
         for i in sorted(plot_df["config_idx"].unique())],
        rotation=0,
        ha="center",
        color="white",
        fontsize=9
    )

    ax.grid(axis="y", alpha=0.2, color="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    plt.tight_layout()
    plt.savefig(output_dir / "repetition_violin.png", dpi=200, facecolor="black")
    plt.close(fig)


def plot_repetition_qq(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot Q-Q plots to check normality of win rate distributions."""
    repetition_groups = detect_repetition_mode(df)

    if not repetition_groups:
        return

    n_configs = len(repetition_groups)
    n_cols = min(3, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.patch.set_facecolor("black")

    if n_configs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for config_idx, ((key, group_df), ax) in enumerate(zip(repetition_groups.items(), axes), 1):
        exploration, fpu, depth, widening, rave = key
        win_rates = group_df["win_rate"].values

        # Create Q-Q plot
        scipy_stats.probplot(win_rates, dist="norm", plot=ax)

        # Style for dark mode
        ax.set_facecolor("black")
        ax.get_lines()[0].set_markerfacecolor("cyan")
        ax.get_lines()[0].set_markeredgecolor("white")
        ax.get_lines()[0].set_markersize(6)
        ax.get_lines()[0].set_markeredgewidth(0.5)
        ax.get_lines()[1].set_color("magenta")
        ax.get_lines()[1].set_linewidth(2)

        config_label = f"Config {config_idx}: Exp={exploration:.2f}"
        ax.set_title(config_label, color="white", fontsize=10)
        ax.set_xlabel("Theoretical Quantiles", color="white")
        ax.set_ylabel("Sample Quantiles (Win Rate)", color="white")
        ax.tick_params(colors="white")
        ax.grid(alpha=0.2, color="white")

        for spine in ax.spines.values():
            spine.set_color("white")

    # Hide unused subplots
    for idx in range(n_configs, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "repetition_qq_plots.png", dpi=200, facecolor="black")
    plt.close(fig)


def plot_repetition_histogram(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot histograms with KDE overlay for each configuration."""
    repetition_groups = detect_repetition_mode(df)

    if not repetition_groups:
        return

    n_configs = len(repetition_groups)
    n_cols = min(3, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.patch.set_facecolor("black")

    if n_configs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for config_idx, ((key, group_df), ax) in enumerate(zip(repetition_groups.items(), axes), 1):
        exploration, fpu, depth, widening, rave = key
        win_rates = group_df["win_rate"].values

        ax.set_facecolor("black")

        # Histogram
        n, bins, patches = ax.hist(
            win_rates,
            bins="auto",
            density=True,
            alpha=0.6,
            color="magenta",
            edgecolor="white",
            linewidth=0.8,
        )

        # KDE overlay
        if len(win_rates) > 1:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(win_rates)
            x_range = np.linspace(win_rates.min(), win_rates.max(), 200)
            ax.plot(x_range, kde(x_range), color="cyan", linewidth=2, label="KDE")

        # Add vertical lines for mean and median
        mean_val = np.mean(win_rates)
        median_val = np.median(win_rates)
        ax.axvline(mean_val, color="yellow", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1%}")
        ax.axvline(median_val, color="lime", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.1%}")

        config_label = f"Config {config_idx}: Exp={exploration:.2f}"
        ax.set_title(config_label, color="white", fontsize=10)
        ax.set_xlabel("Win Rate", color="white")
        ax.set_ylabel("Density", color="white")
        ax.tick_params(colors="white")
        ax.grid(alpha=0.2, color="white")

        legend = ax.legend(facecolor="black", edgecolor="white", fontsize=8)
        for text in legend.get_texts():
            text.set_color("white")

        for spine in ax.spines.values():
            spine.set_color("white")

    # Hide unused subplots
    for idx in range(n_configs, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "repetition_histograms.png", dpi=200, facecolor="black")
    plt.close(fig)


def plot_repetition_timeseries(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot win rate vs repetition number to check for temporal trends."""
    repetition_groups = detect_repetition_mode(df)

    if not repetition_groups:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    colors = plt.cm.turbo(np.linspace(0.2, 0.9, len(repetition_groups)))

    for config_idx, ((key, group_df), color) in enumerate(zip(repetition_groups.items(), colors), 1):
        exploration, fpu, depth, widening, rave = key

        # Sort by original index to preserve temporal order
        group_sorted = group_df.sort_index()
        win_rates = group_sorted["win_rate"].values
        repetitions = np.arange(1, len(win_rates) + 1)

        config_label = f"Config {config_idx}: Exp={exploration:.2f}"

        # Plot individual points
        ax.scatter(
            repetitions,
            win_rates,
            c=[color],
            s=30,
            alpha=0.6,
            edgecolors="white",
            linewidths=0.5,
            label=config_label,
            zorder=2,
        )

        # Plot running average if enough points
        if len(win_rates) >= 5:
            window = min(10, len(win_rates) // 3)
            running_avg = pd.Series(win_rates).rolling(window=window, center=True).mean()
            ax.plot(
                repetitions,
                running_avg,
                color=color,
                linewidth=2,
                alpha=0.8,
                zorder=3,
            )

    ax.set_title("Win Rate vs Repetition Number (Temporal Trends)", color="white", fontsize=14)
    ax.set_xlabel("Repetition Number", color="white")
    ax.set_ylabel("Win Rate", color="white")
    ax.tick_params(colors="white")
    ax.grid(alpha=0.2, color="white")

    legend = ax.legend(
        facecolor="black",
        edgecolor="white",
        framealpha=0.8,
        loc="best",
    )
    for text in legend.get_texts():
        text.set_color("white")

    for spine in ax.spines.values():
        spine.set_color("white")

    plt.tight_layout()
    plt.savefig(output_dir / "repetition_timeseries.png", dpi=200, facecolor="black")
    plt.close(fig)


def plot_repetition_errorbar(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot mean with 95% confidence intervals for each configuration."""
    repetition_groups = detect_repetition_mode(df)

    if not repetition_groups:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    config_labels = []
    means = []
    ci_lower = []
    ci_upper = []

    for config_idx, (key, group_df) in enumerate(repetition_groups.items(), 1):
        exploration, fpu, depth, widening, rave = key
        win_rates = group_df["win_rate"].values

        n = len(win_rates)
        mean_wr = np.mean(win_rates)
        std_wr = np.std(win_rates, ddof=1) if n > 1 else 0.0

        # Calculate 95% CI
        if n > 1:
            sem = std_wr / np.sqrt(n)
            ci_95 = scipy_stats.t.interval(0.95, n - 1, loc=mean_wr, scale=sem)
        else:
            ci_95 = (mean_wr, mean_wr)

        config_labels.append(f"Config {config_idx}\nExp={exploration:.2f}")
        means.append(mean_wr)
        ci_lower.append(mean_wr - ci_95[0])
        ci_upper.append(ci_95[1] - mean_wr)

    x_pos = np.arange(len(config_labels))

    ax.errorbar(
        x_pos,
        means,
        yerr=[ci_lower, ci_upper],
        fmt="o",
        markersize=10,
        color="cyan",
        ecolor="magenta",
        elinewidth=2,
        capsize=5,
        capthick=2,
        markeredgecolor="white",
        markeredgewidth=0.8,
    )

    ax.set_title("Mean Win Rate with 95% Confidence Intervals", color="white", fontsize=14)
    ax.set_xlabel("Configuration", color="white")
    ax.set_ylabel("Win Rate", color="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels, rotation=0, ha="center", color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.grid(axis="y", alpha=0.2, color="white")

    for spine in ax.spines.values():
        spine.set_color("white")

    plt.tight_layout()
    plt.savefig(output_dir / "repetition_ci_errorbar.png", dpi=200, facecolor="black")
    plt.close(fig)


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
        default=None,
        help="Directory for saved visualisations. Defaults to 'plots' subdirectory next to input file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Default output directory is 'plots' subdirectory next to input file
    if args.output_dir is None:
        output_dir = args.input.parent / "plots"
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input)

    # Detect if this is repetition mode
    repetition_groups = detect_repetition_mode(df)
    is_repetition = len(repetition_groups) > 0

    if is_repetition:
        print(f"Detected repetition mode: {len(repetition_groups)} configuration(s) with multiple trials")
        print("Generating repetition-specific visualizations...")

        # Repetition mode plots
        plot_repetition_violin(df, output_dir)
        plot_repetition_qq(df, output_dir)
        plot_repetition_histogram(df, output_dir)
        plot_repetition_timeseries(df, output_dir)
        plot_repetition_errorbar(df, output_dir)

        print("Saved repetition mode figures:")
        print(f"  - {output_dir / 'repetition_violin.png'}")
        print(f"  - {output_dir / 'repetition_qq_plots.png'}")
        print(f"  - {output_dir / 'repetition_histograms.png'}")
        print(f"  - {output_dir / 'repetition_timeseries.png'}")
        print(f"  - {output_dir / 'repetition_ci_errorbar.png'}")
    else:
        print("Detected search mode (unique configurations)")
        print("Generating search-specific visualizations...")

        # Search mode plots (dark theme only)
        plot_pareto_dark(df, output_dir)
        plot_dark_heatmaps(df, output_dir)
        plot_dark_pareto(df, output_dir)
        plot_hyperparam_scatter_3d(df, output_dir)

        # Interactive plots (if available)
        interactive_path, parallel_path = plot_interactive(df, output_dir)
        isosurface_path = plot_isosurface(df, output_dir)

        print(f"Saved static figures to {output_dir}")
        if interactive_path is not None:
            print(f"Interactive HTML written to {interactive_path}")
        else:
            print("Plotly not available; skipped interactive output.")
        if parallel_path is not None:
            print(f"Parallel coordinates HTML written to {parallel_path}")
        elif px is None:
            print("Plotly not available; skipped parallel coordinates.")
        if isosurface_path is not None:
            print(f"Isosurface HTML written to {isosurface_path}")
        elif go is None:
            print("Plotly graph_objects not available; skipped isosurface plot.")


if __name__ == "__main__":
    main()
