from dataclasses import dataclass

import numpy as np


@dataclass
class CVRPInstance:
    """
    Represents a Capacitated Vehicle Routing Problem instance.

    Attributes:
        depot: Coordinate of the depot with shape (2,).
        customers: Coordinates of customers with shape (num_customers, 2).
        demands: Demand of each customer with shape (num_customers,).
        num_vehicles: Number of available vehicles.
        vehicle_capacity: Maximum capacity of each vehicle.
    """

    depot: np.ndarray
    customers: np.ndarray
    demands: np.ndarray
    num_vehicles: int
    vehicle_capacity: int


def generate_cvrp_instance(
    num_customers: int = 30,
    num_vehicles: int = 3,
    vehicle_capacity: int = 30,
    coordinate_min: float = 0.0,
    coordinate_max: float = 1.0,
    demand_min: int = 1,
    demand_max: int = 10,
    seed: int = 42,
) -> CVRPInstance:
    """
    Generate a synthetic CVRP instance.

    Args:
        num_customers: Number of customer locations.
        num_vehicles: Number of vehicles.
        vehicle_capacity: Capacity of each vehicle.
        coordinate_min: Minimum coordinate value.
        coordinate_max: Maximum coordinate value.
        demand_min: Minimum customer demand.
        demand_max: Maximum customer demand.
        seed: Random seed for reproducibility.

    Returns:
        A CVRPInstance object.
    """
    rng = np.random.default_rng(seed)

    depot = rng.uniform(
        low=coordinate_min,
        high=coordinate_max,
        size=2,
    )

    customers = rng.uniform(
        low=coordinate_min,
        high=coordinate_max,
        size=(num_customers, 2),
    )

    demands = rng.integers(
        low=demand_min,
        high=demand_max + 1,
        size=num_customers,
    )

    return CVRPInstance(
        depot=depot,
        customers=customers,
        demands=demands,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
    )