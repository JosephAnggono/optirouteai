from optirouteai.data.generator import CVRPInstance
from optirouteai.optimization.distance import euclidean_distance


def capacity_aware_nearest_neighbor(instance: CVRPInstance) -> list[list[int]]:
    """
    Build CVRP routes using a capacity-aware nearest neighbor heuristic.

    Algorithm:
    1. Start each vehicle at the depot.
    2. Repeatedly visit the nearest unvisited customer that fits remaining capacity.
    3. If no feasible customer fits, move to the next vehicle.
    4. Stop when all customers are visited or all vehicles are used.

    Args:
        instance: CVRP instance.

    Returns:
        List of routes, where each route is a list of customer indices.
    """
    num_customers = len(instance.customers)
    unvisited = set(range(num_customers))

    routes: list[list[int]] = []

    for _ in range(instance.num_vehicles):
        route: list[int] = []
        current_point = instance.depot
        remaining_capacity = instance.vehicle_capacity

        while unvisited:
            feasible_customers = [
                customer_idx
                for customer_idx in unvisited
                if instance.demands[customer_idx] <= remaining_capacity
            ]

            if not feasible_customers:
                break

            nearest_customer = min(
                feasible_customers,
                key=lambda customer_idx: euclidean_distance(
                    current_point,
                    instance.customers[customer_idx],
                ),
            )

            route.append(nearest_customer)
            unvisited.remove(nearest_customer)

            remaining_capacity -= int(instance.demands[nearest_customer])
            current_point = instance.customers[nearest_customer]

        routes.append(route)

        if not unvisited:
            break

    return routes