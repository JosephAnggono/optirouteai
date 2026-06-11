import math

import pandas as pd

from optirouteai.data.generator import CVRPInstance
from optirouteai.optimization.distance import euclidean_distance


ENHANCED_FEATURE_COLUMNS = [
    "dx",
    "dy",
    "distance_to_current",
    "distance_customer_to_depot",
    "distance_current_to_depot",
    "angle_to_customer",
    "customer_demand",
    "customer_demand_share_total",
    "remaining_capacity",
    "remaining_capacity_ratio",
    "demand_capacity_ratio",
    "remaining_capacity_after",
    "route_load_used",
    "route_load_ratio",
    "num_unvisited",
    "route_position",
    "vehicle_id",
    "nearest_rank",
    "nearest_rank_pct",
    "avg_distance_to_unvisited",
]


def build_candidate_features(
    instance: CVRPInstance,
    current_point,
    feasible_candidates: list[int],
    unvisited: set[int],
    remaining_capacity: int,
    vehicle_id: int,
    route_position: int,
    route_load_used: int,
) -> pd.DataFrame:
    """
    Build enhanced candidate-level features for routing decisions.

    Each row represents one feasible candidate customer at the current decision step.
    """
    if not feasible_candidates:
        return pd.DataFrame()

    total_demand = max(int(instance.demands.sum()), 1)
    vehicle_capacity = max(int(instance.vehicle_capacity), 1)

    distances_to_current = {
        customer_idx: euclidean_distance(
            current_point,
            instance.customers[customer_idx],
        )
        for customer_idx in feasible_candidates
    }

    sorted_by_distance = sorted(
        feasible_candidates,
        key=lambda idx: distances_to_current[idx],
    )

    nearest_rank_map = {
        customer_idx: rank + 1
        for rank, customer_idx in enumerate(sorted_by_distance)
    }

    rows = []

    distance_current_to_depot = euclidean_distance(
        current_point,
        instance.depot,
    )

    for customer_idx in feasible_candidates:
        customer_point = instance.customers[customer_idx]
        customer_demand = int(instance.demands[customer_idx])

        dx = float(customer_point[0] - current_point[0])
        dy = float(customer_point[1] - current_point[1])

        distance_to_current = distances_to_current[customer_idx]
        distance_customer_to_depot = euclidean_distance(
            customer_point,
            instance.depot,
        )

        angle_to_customer = math.atan2(dy, dx)

        remaining_capacity_after = int(remaining_capacity - customer_demand)

        other_unvisited = [
            idx for idx in unvisited
            if idx != customer_idx
        ]

        if other_unvisited:
            avg_distance_to_unvisited = sum(
                euclidean_distance(
                    customer_point,
                    instance.customers[other_idx],
                )
                for other_idx in other_unvisited
            ) / len(other_unvisited)
        else:
            avg_distance_to_unvisited = 0.0

        nearest_rank = nearest_rank_map[customer_idx]
        nearest_rank_pct = nearest_rank / max(len(feasible_candidates), 1)

        rows.append(
            {
                "customer_idx": customer_idx,
                "dx": dx,
                "dy": dy,
                "distance_to_current": float(distance_to_current),
                "distance_customer_to_depot": float(distance_customer_to_depot),
                "distance_current_to_depot": float(distance_current_to_depot),
                "angle_to_customer": float(angle_to_customer),
                "customer_demand": customer_demand,
                "customer_demand_share_total": float(customer_demand / total_demand),
                "remaining_capacity": int(remaining_capacity),
                "remaining_capacity_ratio": float(remaining_capacity / vehicle_capacity),
                "demand_capacity_ratio": float(customer_demand / max(remaining_capacity, 1)),
                "remaining_capacity_after": remaining_capacity_after,
                "route_load_used": int(route_load_used),
                "route_load_ratio": float(route_load_used / vehicle_capacity),
                "num_unvisited": int(len(unvisited)),
                "route_position": int(route_position),
                "vehicle_id": int(vehicle_id),
                "nearest_rank": int(nearest_rank),
                "nearest_rank_pct": float(nearest_rank_pct),
                "avg_distance_to_unvisited": float(avg_distance_to_unvisited),
            }
        )

    return pd.DataFrame(rows)