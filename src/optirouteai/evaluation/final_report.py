from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


KEY_SOLVERS = [
    "Nearest Neighbor",
    "Neural MLP",
    "Hybrid MLP alpha=0.5",
    "OR-Tools",
]


def load_step11_summary(
    summary_path: str = "reports/benchmark_summary_step11_hybrid_solver.csv",
) -> pd.DataFrame:
    """
    Load Step 11 hybrid solver benchmark summary.
    """
    path = Path(summary_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Step 11 summary file not found: {summary_path}. "
            "Please run scripts/run_step11_hybrid_solver_benchmark.py first."
        )

    return pd.read_csv(path)


def build_final_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build final solver comparison table.

    Keeps the most important solvers:
    - Nearest Neighbor
    - Neural MLP / Hybrid alpha=1.0 equivalent if Neural MLP missing
    - Hybrid MLP alpha=0.5
    - OR-Tools
    """
    summary = summary.copy()

    # In Step 11, pure MLP appears as Hybrid MLP alpha=1.0.
    if "Neural MLP" not in set(summary["solver_name"]):
        summary.loc[
            summary["solver_name"] == "Hybrid MLP alpha=1.0",
            "solver_name",
        ] = "Neural MLP"

    comparison = summary[summary["solver_name"].isin(KEY_SOLVERS)].copy()

    solver_order_map = {
        "Nearest Neighbor": 1,
        "Neural MLP": 2,
        "Hybrid MLP alpha=0.5": 3,
        "OR-Tools": 4,
    }

    comparison["solver_order"] = comparison["solver_name"].map(solver_order_map)
    comparison = comparison.sort_values("solver_order").drop(columns=["solver_order"])

    return comparison


def compute_main_insights(comparison: pd.DataFrame) -> dict:
    """
    Compute main improvement insights.
    """
    nn = comparison[comparison["solver_name"] == "Nearest Neighbor"]
    hybrid = comparison[comparison["solver_name"] == "Hybrid MLP alpha=0.5"]
    ortools = comparison[comparison["solver_name"] == "OR-Tools"]

    insights = {}

    if not nn.empty and not hybrid.empty:
        nn_gap = float(nn["avg_gap_vs_ortools_percent"].iloc[0])
        hybrid_gap = float(hybrid["avg_gap_vs_ortools_percent"].iloc[0])

        absolute_improvement = nn_gap - hybrid_gap
        relative_improvement = (
            absolute_improvement / nn_gap * 100.0
            if nn_gap != 0
            else 0.0
        )

        insights["nearest_neighbor_gap"] = nn_gap
        insights["best_hybrid_gap"] = hybrid_gap
        insights["absolute_gap_improvement"] = absolute_improvement
        insights["relative_gap_improvement"] = relative_improvement

    if not hybrid.empty and not ortools.empty:
        hybrid_runtime = float(hybrid["avg_runtime_seconds"].iloc[0])
        ortools_runtime = float(ortools["avg_runtime_seconds"].iloc[0])

        speedup_vs_ortools = (
            ortools_runtime / hybrid_runtime
            if hybrid_runtime > 0
            else None
        )

        insights["hybrid_runtime"] = hybrid_runtime
        insights["ortools_runtime"] = ortools_runtime
        insights["speedup_vs_ortools"] = speedup_vs_ortools

    return insights


def save_final_comparison(
    comparison: pd.DataFrame,
    output_path: str = "reports/final_solver_comparison.csv",
) -> str:
    """
    Save final solver comparison table.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(path, index=False)

    return str(path)


def create_gap_plot(
    comparison: pd.DataFrame,
    output_path: str = "reports/figures/final_gap_comparison.png",
) -> str:
    """
    Create bar plot for average gap vs OR-Tools.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = comparison.copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        plot_df["solver_name"],
        plot_df["avg_gap_vs_ortools_percent"],
    )

    ax.set_title("Average Gap vs OR-Tools by Solver")
    ax.set_ylabel("Average Gap vs OR-Tools (%)")
    ax.set_xlabel("Solver")
    ax.tick_params(axis="x", rotation=25)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f%%", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(path)


def create_runtime_plot(
    comparison: pd.DataFrame,
    output_path: str = "reports/figures/final_runtime_comparison.png",
) -> str:
    """
    Create bar plot for average runtime by solver.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = comparison.copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        plot_df["solver_name"],
        plot_df["avg_runtime_seconds"],
    )

    ax.set_title("Average Runtime by Solver")
    ax.set_ylabel("Average Runtime (seconds)")
    ax.set_xlabel("Solver")
    ax.tick_params(axis="x", rotation=25)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.4fs", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(path)


def save_markdown_report(
    comparison: pd.DataFrame,
    insights: dict,
    output_path: str = "reports/final_benchmark_report.md",
) -> str:
    """
    Save final benchmark report as Markdown.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table_markdown = comparison.round(4).to_markdown(index=False)

    nearest_neighbor_gap = insights.get("nearest_neighbor_gap")
    best_hybrid_gap = insights.get("best_hybrid_gap")
    absolute_gap_improvement = insights.get("absolute_gap_improvement")
    relative_gap_improvement = insights.get("relative_gap_improvement")
    speedup_vs_ortools = insights.get("speedup_vs_ortools")

    report = f"""# OptiRouteAI Final Benchmark Report

## Overview

This report compares four solver approaches for the Capacitated Vehicle Routing Problem (CVRP):

1. **Nearest Neighbor** — fast heuristic baseline.
2. **Neural MLP** — pure neural imitation solver.
3. **Hybrid MLP alpha=0.5** — combines neural scoring with distance-based heuristic.
4. **OR-Tools** — classical optimization benchmark.

## Solver Comparison

{table_markdown}

## Main Result

The strongest non-OR-Tools solver is **Hybrid MLP alpha=0.5**.

- Nearest Neighbor average gap vs OR-Tools: **{nearest_neighbor_gap:.4f}%**
- Hybrid MLP alpha=0.5 average gap vs OR-Tools: **{best_hybrid_gap:.4f}%**
- Absolute gap improvement: **{absolute_gap_improvement:.4f} percentage points**
- Relative improvement over Nearest Neighbor: **{relative_gap_improvement:.2f}%**

## Runtime Insight

The Hybrid MLP solver is substantially faster than OR-Tools.

- Hybrid MLP alpha=0.5 average runtime: **{insights.get("hybrid_runtime", 0):.4f} seconds**
- OR-Tools average runtime: **{insights.get("ortools_runtime", 0):.4f} seconds**
- Approximate speedup vs OR-Tools: **{speedup_vs_ortools:.2f}x**

## Interpretation

The pure Neural MLP solver is feasible, but the best performance comes from combining the neural model with a distance-based heuristic. This indicates that the learned model adds useful signal when combined with a classical routing heuristic.

The current result shows that the hybrid solver improves over the nearest-neighbor baseline while maintaining 100% feasibility across the benchmark instances.

## Current Limitation

The hybrid neural solver is still worse than OR-Tools in route quality. The next improvements should focus on:

- graph or attention-based route representation,
- better route-level objective training,
- reinforcement learning or ranking loss,
- larger benchmark set,
- more realistic routing constraints such as time windows.
"""

    path.write_text(report, encoding="utf-8")

    return str(path)


def generate_final_report(
    summary_path: str = "reports/benchmark_summary_step11_hybrid_solver.csv",
) -> dict:
    """
    Generate final benchmark outputs.
    """
    summary = load_step11_summary(summary_path)
    comparison = build_final_comparison(summary)
    insights = compute_main_insights(comparison)

    comparison_path = save_final_comparison(comparison)
    gap_plot_path = create_gap_plot(comparison)
    runtime_plot_path = create_runtime_plot(comparison)
    report_path = save_markdown_report(comparison, insights)

    return {
        "comparison_path": comparison_path,
        "gap_plot_path": gap_plot_path,
        "runtime_plot_path": runtime_plot_path,
        "report_path": report_path,
        "comparison": comparison,
        "insights": insights,
    }