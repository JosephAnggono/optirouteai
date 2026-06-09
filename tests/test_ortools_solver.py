import math

from optirouteai.data.generator import generate_cvrp_instance
from optirouteai.optimization.constraints import check_solution_feasibility
from optirouteai.optimization.ortools_solver import (
    build_distance_matrix,
    solve_cvrp_with_ortools,
)


def test_build_distance_matrix_shape():
    instance = generate_cvrp_instance(
        num_customers=5,
        num_vehicles=2,
        vehicle_capacity=20,
        seed=42,
    )

    distance_matrix = build_distance_matrix(instance)

    assert len(distance_matrix) == 6
    assert len(distance_matrix[0]) == 6


def test_build_distance_matrix_diagonal_zero():
    instance = generate_cvrp_instance(
        num_customers=5,
        num_vehicles=2,
        vehicle_capacity=20,
        seed=42,
    )

    distance_matrix = build_distance_matrix(instance)

    for i in range(len(distance_matrix)):
        assert distance_matrix[i][i] == 0


def test_solve_cvrp_with_ortools_feasible_instance():
    instance = generate_cvrp_instance(
        num_customers=8,
        num_vehicles=3,
        vehicle_capacity=30,
        seed=42,
    )

    result = solve_cvrp_with_ortools(
        instance=instance,
        time_limit_seconds=2,
    )

    assert result.solver_name == "OR-Tools"
    assert result.is_feasible is True
    assert math.isfinite(result.total_distance)

    report = check_solution_feasibility(result.routes, instance)

    assert report.is_feasible is True