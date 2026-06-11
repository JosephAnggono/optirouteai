import math
from pathlib import Path

import pandas as pd

from optirouteai.data.generator import generate_cvrp_instance
from optirouteai.evaluation.benchmark import compute_gap_percent, solve_with_nearest_neighbor
from optirouteai.optimization.ortools_solver import solve_cvrp_with_ortools


def run_batch_benchmark(
    num_instances: int = 20,
    num_customers: int = 25,
    num_vehicles: int = 5,
    vehicle_capacity: int = 45,
    demand_min: int = 1,
    demand_max: int = 8,
    base_seed: int = 42,
    ortools_time_limit_seconds: int = 1,
) -> pd.DataFrame:
    """
    Run benchmark over multiple generated CVRP instances.

    Each instance is solved using:
    1. Capacity-aware nearest neighbor
    2. OR-Tools

    Args:
        num_instances: Number of CVRP instances to generate.
        num_customers: Number of customers per instance.
        num_vehicles: Number of vehicles.
        vehicle_capacity: Capacity of each vehicle.
        demand_min: Minimum customer demand.
        demand_max: Maximum customer demand.
        base_seed: Base seed for reproducible generation.
        ortools_time_limit_seconds: OR-Tools time limit per instance.

    Returns:
        DataFrame containing solver results for all instances.
    """
    rows = []

    for instance_id in range(num_instances):
        seed = base_seed + instance_id

        instance = generate_cvrp_instance(
            num_customers=num_customers,
            num_vehicles=num_vehicles,
            vehicle_capacity=vehicle_capacity,
            demand_min=demand_min,
            demand_max=demand_max,
            seed=seed,
        )

        total_demand = int(instance.demands.sum())
        total_capacity = int(instance.num_vehicles * instance.vehicle_capacity)

        nearest_neighbor_result = solve_with_nearest_neighbor(instance)

        ortools_result = solve_cvrp_with_ortools(
            instance=instance,
            time_limit_seconds=ortools_time_limit_seconds,
        )

        reference_distance = None
        if (
            ortools_result.is_feasible
            and math.isfinite(ortools_result.total_distance)
            and ortools_result.total_distance > 0
        ):
            reference_distance = ortools_result.total_distance

        solver_results = [
            nearest_neighbor_result,
            ortools_result,
        ]

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
                    "seed": seed,
                    "num_customers": num_customers,
                    "num_vehicles": num_vehicles,
                    "vehicle_capacity": vehicle_capacity,
                    "total_demand": total_demand,
                    "total_capacity": total_capacity,
                    "solver_name": result.solver_name,
                    "total_distance": result.total_distance,
                    "runtime_seconds": result.runtime_seconds,
                    "is_feasible": result.is_feasible,
                    "gap_vs_ortools_percent": gap,
                }
            )

    return pd.DataFrame(rows)


def summarize_batch_results(results: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize benchmark results by solver.

    Args:
        results: DataFrame returned by run_batch_benchmark.

    Returns:
        Summary DataFrame.
    """
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

    return summary


def save_batch_results(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str = "reports",
) -> tuple[str, str]:
    """
    Save detailed benchmark results and summary.

    Args:
        results: Detailed benchmark result DataFrame.
        summary: Summary DataFrame.
        output_dir: Output folder.

    Returns:
        Tuple of result CSV path and summary CSV path.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results_path = str(Path(output_dir) / "benchmark_results_step6.csv")
    summary_path = str(Path(output_dir) / "benchmark_summary_step6.csv")

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    return results_path, summary_path