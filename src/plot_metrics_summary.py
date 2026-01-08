
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    
    df.columns = [c.strip().lower() for c in df.columns]

    expected_cols = {
        "task",
        "method",
        "algorithm",
        "silhouette_score",
        "calinski_harabasz_index",
        "davies_bouldin_index",
        "adjusted_rand_index",
        "nmi",
        "cluster_purity",
    }

    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    df["name"] = df["method"].fillna(df["algorithm"]).fillna("unknown")
    df["task"] = df["task"].fillna("unknown")

    return df


def make_metrics_table(df: pd.DataFrame, out_path: Path):
    display_cols = [
        "task",
        "name",
        "silhouette_score",
        "calinski_harabasz_index",
        "davies_bouldin_index",
        "adjusted_rand_index",
        "nmi",
        "cluster_purity",
    ]

    table_df = df[display_cols].copy()

    for col in display_cols[2:]:
        table_df[col] = table_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    fig, ax = plt.subplots(figsize=(16, max(4, 0.4 * len(table_df) + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=[c.replace("_", " ").title() for c in display_cols],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax.set_title("Clustering Metrics Summary", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_metric_bars(df: pd.DataFrame, out_path: Path):
    metrics = [
        ("silhouette_score", "Silhouette (higher better)"),
        ("calinski_harabasz_index", "Calinski-Harabasz (higher better)"),
        ("davies_bouldin_index", "Davies-Bouldin (lower better)"),
        ("adjusted_rand_index", "ARI (higher better)"),
        ("nmi", "NMI (higher better)"),
        ("cluster_purity", "Purity (higher better)"),
    ]

    df_sorted = df.sort_values(["task", "name"])
    methods = df_sorted["name"].unique()
    tasks = df_sorted["task"].unique()

    n_cols = 2
    n_rows = int(np.ceil(len(metrics) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]
        sub = df_sorted[["task", "name", metric]].dropna()
        if sub.empty:
            ax.text(0.5, 0.5, f"No data for {label}", ha="center", va="center")
            ax.axis("off")
            continue

        x = np.arange(len(methods))
        width = 0.8 / max(1, len(tasks)) 

        for t_i, task in enumerate(tasks):
            vals = []
            for m in methods:
                row = sub[(sub["task"] == task) & (sub["name"] == m)]
                vals.append(row[metric].values[0] if not row.empty else np.nan)
            offset = (t_i - (len(tasks) - 1) / 2) * width
            ax.bar(x + offset, vals, width=width, label=task)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(title="Task", fontsize=8)

    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "results" / "clustering_metrics.csv"
    out_dir = project_root / "results" / "latent_visualization"
    table_path = out_dir / "metrics_table.png"
    bars_path = out_dir / "metrics_bars.png"

    df = load_metrics(csv_path)
    make_metrics_table(df, table_path)
    make_metric_bars(df, bars_path)
    print(f"Saved metrics table to: {table_path}")
    print(f"Saved metrics bar charts to: {bars_path}")


if __name__ == "__main__":
    main()

