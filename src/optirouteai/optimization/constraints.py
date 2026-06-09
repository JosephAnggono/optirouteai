from dataclasses import dataclass

from optirouteai.data.generator import CVRPInstance


@dataclass
class FeasibilityReport:
    """
    Stores feasibility-checking results for a CVRP solution.
    """

    is_feasible: bool
    all_customers_visited_once: bool
    capacity_constraints_satisfied: bool
    num_missing_customers: int
    num_duplicate_visits: int
    num_capacity_violations: int
    route_loads: list[int]


def route_load(route: list[int], instance: CVRPInstance) -> int:
    """
    Compute total demand served by one vehicle route.

    Args:
        route: List of customer indices.
        instance: CVRP instance.

    Returns:
        Total demand/load of the route.
    """
    return int(sum(instance.demands[customer_idx] for customer_idx in route))


def solution_loads(routes: list[list[int]], instance: CVRPInstance) -> list[int]:
    """
    Compute load for each vehicle route.

    Args:
        routes: List of routes.
        instance: CVRP instance.

    Returns:
        List of route loads.
    """
    return [route_load(route, instance) for route in routes]


def check_customer_coverage(
    routes: list[list[int]],
    instance: CVRPInstance,
) -> tuple[bool, int, int]:
    """
    Check whether every customer is visited exactly once.

    Args:
        routes: List of routes.
        instance: CVRP instance.

    Returns:
        A tuple:
        - all_customers_visited_once
        - num_missing_customers
        - num_duplicate_visits
    """
    num_customers = len(instance.customers)

    visited_customers = []
    for route in routes:
        visited_customers.extend(route)

    visited_set = set(visited_customers)
    expected_set = set(range(num_customers))

    missing_customers = expected_set - visited_set

    num_duplicate_visits = len(visited_customers) - len(visited_set)

    all_customers_visited_once = (
        len(missing_customers) == 0
        and num_duplicate_visits == 0
        and visited_set == expected_set
    )

    return (
        all_customers_visited_once,
        len(missing_customers),
        num_duplicate_visits,
    )


def check_capacity_constraints(
    routes: list[list[int]],
    instance: CVRPInstance,
) -> tuple[bool, int, list[int]]:
    """
    Check whether all vehicle routes satisfy vehicle capacity.

    Args:
        routes: List of routes.
        instance: CVRP instance.

    Returns:
        A tuple:
        - capacity_constraints_satisfied
        - num_capacity_violations
        - route_loads
    """
    loads = solution_loads(routes, instance)

    num_capacity_violations = sum(
        load > instance.vehicle_capacity for load in loads
    )

    capacity_constraints_satisfied = num_capacity_violations == 0

    return (
        capacity_constraints_satisfied,
        int(num_capacity_violations),
        loads,
    )


def check_solution_feasibility(
    routes: list[list[int]],
    instance: CVRPInstance,
) -> FeasibilityReport:
    """
    Check full feasibility of a CVRP solution.

    A solution is feasible if:
    1. Every customer is visited exactly once.
    2. No vehicle route exceeds vehicle capacity.

    Args:
        routes: List of vehicle routes.
        instance: CVRP instance.

    Returns:
        FeasibilityReport.
    """
    (
        all_customers_visited_once,
        num_missing_customers,
        num_duplicate_visits,
    ) = check_customer_coverage(routes, instance)

    (
        capacity_constraints_satisfied,
        num_capacity_violations,
        route_loads,
    ) = check_capacity_constraints(routes, instance)

    is_feasible = (
        all_customers_visited_once
        and capacity_constraints_satisfied
    )

    return FeasibilityReport(
        is_feasible=is_feasible,
        all_customers_visited_once=all_customers_visited_once,
        capacity_constraints_satisfied=capacity_constraints_satisfied,
        num_missing_customers=num_missing_customers,
        num_duplicate_visits=num_duplicate_visits,
        num_capacity_violations=num_capacity_violations,
        route_loads=route_loads,
    )