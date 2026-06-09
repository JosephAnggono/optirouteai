import numpy as np

from optirouteai.data.generator import CVRPInstance


def euclidean_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """
    Compute Euclidean distance between two 2D points.

    Args:
        point_a: First point with shape (2,).
        point_b: Second point with shape (2,).

    Returns:
        Euclidean distance as a float.
    """
    return float(np.linalg.norm(point_a - point_b))


def route_distance(route: list[int], instance: CVRPInstance) -> float:
    """
    Compute total distance of one vehicle route.

    Route format:
        [customer_index_1, customer_index_2, ...]

    The route automatically starts and ends at the depot.

    Args:
        route: List of customer indices.
        instance: CVRP instance.

    Returns:
        Total route distance.
    """
    if len(route) == 0:
        return 0.0

    total_distance = 0.0
    current_point = instance.depot

    for customer_idx in route:
        next_point = instance.customers[customer_idx]
        total_distance += euclidean_distance(current_point, next_point)
        current_point = next_point

    total_distance += euclidean_distance(current_point, instance.depot)

    return total_distance


def solution_distance(routes: list[list[int]], instance: CVRPInstance) -> float:
    """
    Compute total distance of all vehicle routes.

    Args:
        routes: List of vehicle routes.
        instance: CVRP instance.

    Returns:
        Total distance across all vehicles.
    """
    return sum(route_distance(route, instance) for route in routes)