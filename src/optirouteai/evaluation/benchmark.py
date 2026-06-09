import time
from dataclasses import dataclass

from optirouteai.data.generator import CVRPInstance
from optirouteai.optimization.constraints import check_solution_feasibility
from optirouteai.optimization.distance import solution_distance
from optirouteai.optimization.heuristics import capacity_aware_nearest_neighbor
from optirouteai.optimization.ortools_solver import SolverResult


@dataclass
class BenchmarkResult:
    """
    Stores benchmark result for one solver.
    """

    solver_name: str
    total_distance: float
    runtime_seconds: float
    is_feasible: bool
    gap_vs_reference_percent: float | None


def solve_with_nearest_neighbor(instance: CVRPInstance) -> SolverResult:
    """
    Solve a CVRP instance using capacity-aware nearest neighbor.

    Args:
        instance: CVRP instance.

    Returns:
        SolverResult.
    """
    start_time = time.perf_counter()

    routes = capacity_aware_nearest_neighbor(instance)

    runtime_seconds = time.perf_counter() - start_time
    total_distance = solution_distance(routes, instance)
    feasibility_report = check_solution_feasibility(routes, instance)

    return SolverResult(
        solver_name="Nearest Neighbor",
        routes=routes,
        total_distance=total_distance,
        runtime_seconds=runtime_seconds,
        is_feasible=feasibility_report.is_feasible,
    )


def compute_gap_percent(
    solver_distance: float,
    reference_distance: float,
) -> float:
    """
    Compute percentage gap against reference distance.

    Args:
        solver_distance: Distance achieved by solver.
        reference_distance: Reference/best distance.

    Returns:
        Gap percentage.
    """
    if reference_distance <= 0:
        return 0.0

    return ((solver_distance - reference_distance) / reference_distance) * 100.0


def benchmark_solver_results(
    solver_results: list[SolverResult],
    reference_solver_name: str = "OR-Tools",
) -> list[BenchmarkResult]:
    """
    Convert solver results into benchmark results.

    Args:
        solver_results: List of SolverResult.
        reference_solver_name: Solver used as reference.

    Returns:
        List of BenchmarkResult.
    """
    reference_result = None

    for result in solver_results:
        if result.solver_name == reference_solver_name:
            reference_result = result
            break

    reference_distance = (
        reference_result.total_distance
        if reference_result is not None
        else None
    )

    benchmark_results = []

    for result in solver_results:
        if reference_distance is None:
            gap = None
        else:
            gap = compute_gap_percent(
                solver_distance=result.total_distance,
                reference_distance=reference_distance,
            )

        benchmark_results.append(
            BenchmarkResult(
                solver_name=result.solver_name,
                total_distance=result.total_distance,
                runtime_seconds=result.runtime_seconds,
                is_feasible=result.is_feasible,
                gap_vs_reference_percent=gap,
            )
        )

    return benchmark_results