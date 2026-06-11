import math

import pandas as pd

from optirouteai.data.generator import generate_cvrp_instance
from optirouteai.evaluation.benchmark import (
    compute_gap_percent,
    solve_with_nearest_neighbor,
)
from optirouteai.optimization.hybrid_solver import HybridNeuralSolver
from optirouteai.optimization.ortools_solver import solve_cvrp_with_ortools


def main():
    print("=== OptiRouteAI Step 11: Hybrid Neural Solver Benchmark ===")
    print()

    alphas = [0.0, 0.25, 0.50, 0.75, 1.0]
    num_instances = 20

    hybrid_solvers = {
        alpha: HybridNeuralSolver(
            checkpoint_path="models/mlp_policy_step9.pkl",
            alpha=alpha,
        )
        for alpha in alphas
    }

    rows = []

    for instance_id in range(num_instances):
        instance = generate_cvrp_instance(
            num_customers=25,
            num_vehicles=5,
            vehicle_capacity=45,
            demand_min=1,
            demand_max=8,
            seed=42 + instance_id,
        )

        nn_result = solve_with_nearest_neighbor(instance)

        ortools_result = solve_cvrp_with_ortools(
            instance=instance,
            time_limit_seconds=1,
        )

        solver_results = [
            nn_result,
            ortools_result,
        ]

        for alpha, solver in hybrid_solvers.items():
            hybrid_result = solver.solve(instance)
            solver_results.append(hybrid_result)

        reference_distance = None
        if (
            ortools_result.is_feasible
            and math.isfinite(ortools_result.total_distance)
            and ortools_result.total_distance > 0
        ):
            reference_distance = ortools_result.total_distance

        for result in solver_results:
            if reference_distance is None:
                gap = None
            else:
                gap = compute_gap_percent(
                    solver_distance=result.total_distance,
                    reference_distance=reference_distance,
                )

            rows.append(
                {
                    "instance_id": instance_id,
                    "solver_name": result.solver_name,
                    "total_distance": result.total_distance,
                    "runtime_seconds": result.runtime_seconds,
                    "is_feasible": result.is_feasible,
                    "gap_vs_ortools_percent": gap,
                }
            )

    results = pd.DataFrame(rows)

    summary = (
        results.groupby("solver_name")
        .agg(
            num_runs=("instance_id", "count"),
            avg_distance=("total_distance", "mean"),
            median_distance=("total_distance", "median"),
            avg_runtime_seconds=("runtime_seconds", "mean"),
            median_runtime_seconds=("runtime_seconds", "median"),
            feasibility_rate=("is_feasible", "mean"),
            avg_gap_vs_ortools_percent=("gap_vs_ortools_percent", "mean"),
            median_gap_vs_ortools_percent=("gap_vs_ortools_percent", "median"),
        )
        .reset_index()
    )

    summary["feasibility_rate"] = summary["feasibility_rate"] * 100.0

    results_path = "reports/benchmark_results_step11_hybrid_solver.csv"
    summary_path = "reports/benchmark_summary_step11_hybrid_solver.csv"

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Benchmark summary:")
    print(summary.round(4).to_string(index=False))
    print()

    print(f"Saved detailed results to: {results_path}")
    print(f"Saved summary results to: {summary_path}")
    print()

    non_ortools = summary[summary["solver_name"] != "OR-Tools"].copy()
    best_row = non_ortools.sort_values("avg_gap_vs_ortools_percent").iloc[0]

    print("Best non-OR-Tools solver:")
    print(f"Solver: {best_row['solver_name']}")
    print(f"Average gap vs OR-Tools: {best_row['avg_gap_vs_ortools_percent']:.4f}%")
    print(f"Median gap vs OR-Tools: {best_row['median_gap_vs_ortools_percent']:.4f}%")


if __name__ == "__main__":
    main()