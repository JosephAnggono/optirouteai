from pathlib import Path

import pandas as pd
from tqdm import tqdm

from optirouteai.data.generator import CVRPInstance, generate_cvrp_instance
from optirouteai.optimization.distance import euclidean_distance
from optirouteai.optimization.ortools_solver import solve_cvrp_with_ortools
from optirouteai.utils.routing_features import build_candidate_features


def build_imitation_examples_from_solution(
    instance: CVRPInstance,
    routes: list[list[int]],
    instance_id: int,
    solver_name: str = "OR-Tools",
) -> pd.DataFrame:
    """
    Convert solved CVRP routes into supervised imitation learning examples.

    Each row represents one feasible candidate customer at one decision step.
    """
    all_rows = []
    unvisited = set(range(len(instance.customers)))

    for vehicle_id, route in enumerate(routes):
        current_point = instance.depot
        remaining_capacity = instance.vehicle_capacity
        route_load_used = 0

        for route_position, selected_customer in enumerate(route):
            feasible_candidates = sorted(
                [
                    customer_idx
                    for customer_idx in unvisited
                    if int(instance.demands[customer_idx]) <= remaining_capacity
                ]
            )

            if selected_customer not in feasible_candidates:
                continue

            decision_id = f"{instance_id}_{vehicle_id}_{route_position}"

            features_df = build_candidate_features(
                instance=instance,
                current_point=current_point,
                feasible_candidates=feasible_candidates,
                unvisited=unvisited,
                remaining_capacity=remaining_capacity,
                vehicle_id=vehicle_id,
                route_position=route_position,
                route_load_used=route_load_used,
            )

            features_df["instance_id"] = instance_id
            features_df["decision_id"] = decision_id
            features_df["teacher_solver"] = solver_name
            features_df["selected_customer_idx"] = selected_customer
            features_df["label"] = (
                features_df["customer_idx"] == selected_customer
            ).astype(int)

            all_rows.append(features_df)

            unvisited.remove(selected_customer)
            selected_demand = int(instance.demands[selected_customer])
            remaining_capacity -= selected_demand
            route_load_used += selected_demand
            current_point = instance.customers[selected_customer]

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)

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