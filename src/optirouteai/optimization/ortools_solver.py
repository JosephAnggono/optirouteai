import math
import time
from dataclasses import dataclass

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from optirouteai.data.generator import CVRPInstance
from optirouteai.optimization.constraints import check_solution_feasibility
from optirouteai.optimization.distance import euclidean_distance, solution_distance


@dataclass
class SolverResult:
    """
    Stores the result of a CVRP solver.
    """

    solver_name: str
    routes: list[list[int]]
    total_distance: float
    runtime_seconds: float
    is_feasible: bool


def build_distance_matrix(instance: CVRPInstance, scale: int = 10000) -> list[list[int]]:
    """
    Build an integer distance matrix for OR-Tools.

    OR-Tools routing solvers typically work with integer costs.
    We scale Euclidean distances by a constant and round to integers.

    Node indexing:
        0 = depot
        1..num_customers = customers

    Args:
        instance: CVRP instance.
        scale: Scaling factor for converting float distances to integers.

    Returns:
        Integer distance matrix.
    """
    points = np.vstack([instance.depot, instance.customers])
    num_nodes = len(points)

    distance_matrix = []

    for i in range(num_nodes):
        row = []
        for j in range(num_nodes):
            distance = euclidean_distance(points[i], points[j])
            row.append(int(round(distance * scale)))
        distance_matrix.append(row)

    return distance_matrix


def solve_cvrp_with_ortools(
    instance: CVRPInstance,
    time_limit_seconds: int = 5,
) -> SolverResult:
    """
    Solve a CVRP instance using Google OR-Tools.

    Args:
        instance: CVRP instance.
        time_limit_seconds: Maximum solver time.

    Returns:
        SolverResult containing routes and metrics.
    """
    start_time = time.perf_counter()

    distance_matrix = build_distance_matrix(instance)

    # OR-Tools node indexing:
    # 0 = depot
    # 1..num_customers = customers
    num_locations = len(distance_matrix)
    depot_index = 0

    manager = pywrapcp.RoutingIndexManager(
        num_locations,
        instance.num_vehicles,
        depot_index,
    )

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Demands:
    # node 0 is depot, so demand is 0.
    # customer node i corresponds to instance.demands[i - 1].
    demands = [0] + [int(demand) for demand in instance.demands]

    def demand_callback(from_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [int(instance.vehicle_capacity)] * instance.num_vehicles,
        True,  # start cumul to zero
        "Capacity",
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    search_parameters.time_limit.seconds = time_limit_seconds

    solution = routing.SolveWithParameters(search_parameters)

    runtime_seconds = time.perf_counter() - start_time

    if solution is None:
        return SolverResult(
            solver_name="OR-Tools",
            routes=[],
            total_distance=math.inf,
            runtime_seconds=runtime_seconds,
            is_feasible=False,
        )

    routes = extract_routes_from_solution(
        manager=manager,
        routing=routing,
        solution=solution,
        num_vehicles=instance.num_vehicles,
    )

    total_distance = solution_distance(routes, instance)
    feasibility_report = check_solution_feasibility(routes, instance)

    return SolverResult(
        solver_name="OR-Tools",
        routes=routes,
        total_distance=total_distance,
        runtime_seconds=runtime_seconds,
        is_feasible=feasibility_report.is_feasible,
    )


def extract_routes_from_solution(
    manager: pywrapcp.RoutingIndexManager,
    routing: pywrapcp.RoutingModel,
    solution,
    num_vehicles: int,
) -> list[list[int]]:
    """
    Extract customer routes from an OR-Tools solution.

    OR-Tools node indexing:
        0 = depot
        1..num_customers = customers

    Project route indexing:
        0..num_customers-1 = customers

    Args:
        manager: OR-Tools routing index manager.
        routing: OR-Tools routing model.
        solution: OR-Tools solution.
        num_vehicles: Number of vehicles.

    Returns:
        Routes using project customer indices.
    """
    routes = []

    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)

            # skip depot node 0
            if node != 0:
                customer_idx = node - 1
                route.append(customer_idx)

            index = solution.Value(routing.NextVar(index))

        routes.append(route)

    return routes