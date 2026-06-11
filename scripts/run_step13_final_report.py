from optirouteai.evaluation.final_report import generate_final_report


def main():
    print("=== OptiRouteAI Step 13: Final Benchmark Report ===")
    print()

    outputs = generate_final_report(
        summary_path="reports/benchmark_summary_step11_hybrid_solver.csv",
    )

    print("Final solver comparison:")
    print(outputs["comparison"].round(4).to_string(index=False))
    print()

    insights = outputs["insights"]

    print("Main insights:")
    print(
        "Nearest Neighbor avg gap: "
        f"{insights['nearest_neighbor_gap']:.4f}%"
    )
    print(
        "Best Hybrid avg gap: "
        f"{insights['best_hybrid_gap']:.4f}%"
    )
    print(
        "Absolute gap improvement: "
        f"{insights['absolute_gap_improvement']:.4f} percentage points"
    )
    print(
        "Relative improvement over Nearest Neighbor: "
        f"{insights['relative_gap_improvement']:.2f}%"
    )

    if insights.get("speedup_vs_ortools") is not None:
        print(
            "Approximate speedup vs OR-Tools: "
            f"{insights['speedup_vs_ortools']:.2f}x"
        )

    print()
    print("Saved outputs:")
    print(f"- {outputs['comparison_path']}")
    print(f"- {outputs['report_path']}")
    print(f"- {outputs['gap_plot_path']}")
    print(f"- {outputs['runtime_plot_path']}")


if __name__ == "__main__":
    main()