from pathlib import Path

import pandas as pd
from tqdm import tqdm

from optirouteai.data.generator import CVRPInstance, generate_cvrp_instance
from optirouteai.optimization.distance import euclidean_distance
from optirouteai.optimization.ortools_solver import solve_cvrp_with_ortools


def build_imitation_examples_from_solution(
    instance: CVRPInstance,
    routes: list[list[int]],
    instance_id: int,
    solver_name: str = "OR-Tools",
) -> pd.DataFrame:
    """
    Convert a solved CVRP route into supervised imitation learning examples.

    Each row represents one candidate customer at one decision step.

    Label:
        1 = candidate customer was selected by OR-Tools
        0 = candidate customer was not selected

    Args:
        instance: CVRP instance.
        routes: OR-Tools routes.
        instance_id: ID of the generated instance.
        solver_name: Name of the solver used as teacher.

    Returns:
        DataFrame of imitation learning examples.
    """
    rows = []
    unvisited = set(range(len(instance.customers)))

    for vehicle_id, route in enumerate(routes):
        current_point = instance.depot
        remaining_capacity = instance.vehicle_capacity

        for route_position, selected_customer in enumerate(route):
            feasible_candidates = sorted(
                [
                    customer_idx
                    for customer_idx in unvisited
                    if int(instance.demands[customer_idx]) <= remaining_capacity
                ]
            )

            # If the selected customer is not feasible, skip this decision.
            # This should not happen for a valid OR-Tools solution.
            if selected_customer not in feasible_candidates:
                continue

            decision_id = f"{instance_id}_{vehicle_id}_{route_position}"

            for candidate_customer in feasible_candidates:
                customer_point = instance.customers[candidate_customer]
                customer_demand = int(instance.demands[candidate_customer])
                distance_to_current = euclidean_distance(
                    current_point,
                    customer_point,
                )

                rows.append(
                    {
                        "instance_id": instance_id,
                        "decision_id": decision_id,
                        "teacher_solver": solver_name,
                        "vehicle_id": vehicle_id,
                        "route_position": route_position,
                        "current_x": float(current_point[0]),
                        "current_y": float(current_point[1]),
                        "customer_idx": candidate_customer,
                        "customer_x": float(customer_point[0]),
                        "customer_y": float(customer_point[1]),
                        "dx": float(customer_point[0] - current_point[0]),
                        "dy": float(customer_point[1] - current_point[1]),
                        "distance_to_current": float(distance_to_current),
                        "customer_demand": customer_demand,
                        "remaining_capacity": int(remaining_capacity),
                        "demand_capacity_ratio": float(
                            customer_demand / remaining_capacity
                        ),
                        "remaining_capacity_after": int(
                            remaining_capacity - customer_demand
                        ),
                        "num_unvisited": len(unvisited),
                        "selected_customer_idx": selected_customer,
                        "label": int(candidate_customer == selected_customer),
                    }
                )

            unvisited.remove(selected_customer)
            remaining_capacity -= int(instance.demands[selected_customer])
            current_point = instance.customers[selected_customer]

    return pd.DataFrame(rows)


def generate_imitation_dataset(
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
    Generate multiple CVRP instances, solve them using OR-Tools,
    and convert the OR-Tools routes into imitation learning data.

    Args:
        num_instances: Number of CVRP instances.
        num_customers: Number of customers per instance.
        num_vehicles: Number of vehicles.
        vehicle_capacity: Capacity of each vehicle.
        demand_min: Minimum customer demand.
        demand_max: Maximum customer demand.
        base_seed: Base random seed.
        ortools_time_limit_seconds: OR-Tools time limit per instance.

    Returns:
        DataFrame containing imitation learning examples.
    """
    all_examples = []

    for instance_id in tqdm(range(num_instances), desc="Generating imitation dataset"):
        seed = base_seed + instance_id

        instance = generate_cvrp_instance(
            num_customers=num_customers,
            num_vehicles=num_vehicles,
            vehicle_capacity=vehicle_capacity,
            demand_min=demand_min,
            demand_max=demand_max,
            seed=seed,
        )

        ortools_result = solve_cvrp_with_ortools(
            instance=instance,
            time_limit_seconds=ortools_time_limit_seconds,
        )

        if not ortools_result.is_feasible:
            continue

        examples = build_imitation_examples_from_solution(
            instance=instance,
            routes=ortools_result.routes,
            instance_id=instance_id,
            solver_name=ortools_result.solver_name,
        )

        if not examples.empty:
            examples["seed"] = seed
            examples["num_customers"] = num_customers
            examples["num_vehicles"] = num_vehicles
            examples["vehicle_capacity"] = vehicle_capacity
            examples["total_demand"] = int(instance.demands.sum())
            examples["teacher_total_distance"] = ortools_result.total_distance
            examples["teacher_runtime_seconds"] = ortools_result.runtime_seconds

            all_examples.append(examples)

    if not all_examples:
        return pd.DataFrame()

    return pd.concat(all_examples, ignore_index=True)


def save_imitation_dataset(
    dataset: pd.DataFrame,
    output_path: str = "data/processed/imitation_dataset_step7.csv",
) -> str:
    """
    Save imitation dataset to CSV.

    Args:
        dataset: Imitation dataset.
        output_path: Output CSV path.

    Returns:
        Output path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)

    return str(path)