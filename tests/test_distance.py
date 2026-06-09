import numpy as np

from optirouteai.data.generator import generate_cvrp_instance
from optirouteai.optimization.distance import (
    euclidean_distance,
    route_distance,
    solution_distance,
)


def test_euclidean_distance():
    point_a = np.array([0.0, 0.0])
    point_b = np.array([3.0, 4.0])

    assert euclidean_distance(point_a, point_b) == 5.0


def test_route_distance_empty_route():
    instance = generate_cvrp_instance(num_customers=5, seed=42)

    assert route_distance([], instance) == 0.0


def test_solution_distance_is_sum_of_route_distances():
    instance = generate_cvrp_instance(num_customers=5, seed=42)

    routes = [
        [0, 1],
        [2, 3, 4],
    ]

    expected = (
        route_distance(routes[0], instance)
        + route_distance(routes[1], instance)
    )

    assert solution_distance(routes, instance) == expected