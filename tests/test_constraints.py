from optirouteai.data.generator import generate_cvrp_instance
from optirouteai.optimization.constraints import (
    check_solution_feasibility,
    route_load,
    solution_loads,
)


def test_route_load():
    instance = generate_cvrp_instance(num_customers=5, seed=42)

    route = [0, 1, 2]
    expected_load = int(
        instance.demands[0]
        + instance.demands[1]
        + instance.demands[2]
    )

    assert route_load(route, instance) == expected_load


def test_solution_loads():
    instance = generate_cvrp_instance(num_customers=5, seed=42)

    routes = [
        [0, 1],
        [2, 3, 4],
    ]

    loads = solution_loads(routes, instance)

    assert len(loads) == 2
    assert loads[0] == int(instance.demands[0] + instance.demands[1])
    assert loads[1] == int(
        instance.demands[2]
        + instance.demands[3]
        + instance.demands[4]
    )


def test_feasible_solution():
    instance = generate_cvrp_instance(
        num_customers=5,
        num_vehicles=2,
        vehicle_capacity=100,
        seed=42,
    )

    routes = [
        [0, 1],
        [2, 3, 4],
    ]

    report = check_solution_feasibility(routes, instance)

    assert report.is_feasible is True
    assert report.num_missing_customers == 0
    assert report.num_duplicate_visits == 0
    assert report.num_capacity_violations == 0


def test_infeasible_solution_missing_customer():
    instance = generate_cvrp_instance(
        num_customers=5,
        num_vehicles=2,
        vehicle_capacity=100,
        seed=42,
    )

    routes = [
        [0, 1],
        [2, 3],
    ]

    report = check_solution_feasibility(routes, instance)

    assert report.is_feasible is False
    assert report.num_missing_customers == 1


def test_infeasible_solution_duplicate_customer():
    instance = generate_cvrp_instance(
        num_customers=5,
        num_vehicles=2,
        vehicle_capacity=100,
        seed=42,
    )

    routes = [
        [0, 1, 1],
        [2, 3, 4],
    ]

    report = check_solution_feasibility(routes, instance)

    assert report.is_feasible is False
    assert report.num_duplicate_visits == 1


def test_infeasible_solution_capacity_violation():
    instance = generate_cvrp_instance(
        num_customers=5,
        num_vehicles=2,
        vehicle_capacity=1,
        seed=42,
    )

    routes = [
        [0, 1],
        [2, 3, 4],
    ]

    report = check_solution_feasibility(routes, instance)

    assert report.is_feasible is False
    assert report.num_capacity_violations > 0