from optirouteai.data.generator import generate_cvrp_instance
from optirouteai.optimization.constraints import check_solution_feasibility
from optirouteai.optimization.heuristics import capacity_aware_nearest_neighbor


def test_capacity_aware_nearest_neighbor_returns_routes():
    instance = generate_cvrp_instance(
        num_customers=10,
        num_vehicles=3,
        vehicle_capacity=30,
        seed=42,
    )

    routes = capacity_aware_nearest_neighbor(instance)

    assert isinstance(routes, list)
    assert len(routes) <= instance.num_vehicles


def test_capacity_aware_nearest_neighbor_feasible_small_instance():
    instance = generate_cvrp_instance(
        num_customers=10,
        num_vehicles=3,
        vehicle_capacity=30,
        seed=42,
    )

    routes = capacity_aware_nearest_neighbor(instance)
    report = check_solution_feasibility(routes, instance)

    assert report.is_feasible is True
    assert report.num_missing_customers == 0
    assert report.num_duplicate_visits == 0
    assert report.num_capacity_violations == 0


def test_capacity_aware_nearest_neighbor_capacity_limit():
    instance = generate_cvrp_instance(
        num_customers=6,
        num_vehicles=3,
        vehicle_capacity=20,
        seed=42,
    )

    routes = capacity_aware_nearest_neighbor(instance)
    report = check_solution_feasibility(routes, instance)

    assert report.num_capacity_violations == 0