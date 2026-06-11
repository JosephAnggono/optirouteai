import math
import pickle
import time
from pathlib import Path

import pandas as pd
import torch

from optirouteai.data.generator import CVRPInstance
from optirouteai.models.mlp_policy import MLPPolicy
from optirouteai.optimization.constraints import check_solution_feasibility
from optirouteai.optimization.distance import euclidean_distance, solution_distance
from optirouteai.optimization.ortools_solver import SolverResult


FEATURE_COLUMNS = [
    "dx",
    "dy",
    "distance_to_current",
    "customer_demand",
    "remaining_capacity",
    "demand_capacity_ratio",
    "remaining_capacity_after",
    "num_unvisited",
]


class MLPNeuralSolver:
    """
    Neural CVRP solver using trained MLP imitation model.
    """

    def __init__(
        self,
        checkpoint_path: str = "models/mlp_policy_step9.pkl",
        hidden_dim: int = 64,
        dropout: float = 0.10,
    ):
        path = Path(checkpoint_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Please run Step 9 first."
            )

        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        self.feature_columns = checkpoint.get("feature_columns", FEATURE_COLUMNS)
        self.scaler = checkpoint["scaler"]

        input_dim = checkpoint.get("input_dim", len(self.feature_columns))

        self.model = MLPPolicy(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def score_candidates(
        self,
        features_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Score candidate customers using the trained MLP.
        """
        X = self.scaler.transform(features_df[self.feature_columns])
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(X_tensor)
            probabilities = torch.sigmoid(logits).numpy()

        return pd.Series(probabilities, index=features_df.index)

    def solve(self, instance: CVRPInstance) -> SolverResult:
        """
        Solve a CVRP instance using greedy neural candidate selection.
        """
        start_time = time.perf_counter()

        unvisited = set(range(len(instance.customers)))
        routes: list[list[int]] = []

        for _ in range(instance.num_vehicles):
            route: list[int] = []
            current_point = instance.depot
            remaining_capacity = instance.vehicle_capacity

            while unvisited:
                feasible_candidates = sorted(
                    [
                        customer_idx
                        for customer_idx in unvisited
                        if int(instance.demands[customer_idx]) <= remaining_capacity
                    ]
                )

                if not feasible_candidates:
                    break

                feature_rows = []

                for customer_idx in feasible_candidates:
                    customer_point = instance.customers[customer_idx]
                    customer_demand = int(instance.demands[customer_idx])
                    distance_to_current = euclidean_distance(
                        current_point,
                        customer_point,
                    )

                    feature_rows.append(
                        {
                            "customer_idx": customer_idx,
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
                        }
                    )

                features_df = pd.DataFrame(feature_rows)

                scores = self.score_candidates(features_df)

                best_row_idx = scores.idxmax()
                selected_customer = int(features_df.loc[best_row_idx, "customer_idx"])

                route.append(selected_customer)
                unvisited.remove(selected_customer)

                remaining_capacity -= int(instance.demands[selected_customer])
                current_point = instance.customers[selected_customer]

            routes.append(route)

            if not unvisited:
                break

        runtime_seconds = time.perf_counter() - start_time

        total_distance = solution_distance(routes, instance)
        feasibility_report = check_solution_feasibility(routes, instance)

        if not feasibility_report.is_feasible:
            total_distance = math.inf

        return SolverResult(
            solver_name="Neural MLP",
            routes=routes,
            total_distance=total_distance,
            runtime_seconds=runtime_seconds,
            is_feasible=feasibility_report.is_feasible,
        )