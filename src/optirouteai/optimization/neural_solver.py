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
from optirouteai.utils.routing_features import (
    ENHANCED_FEATURE_COLUMNS,
    build_candidate_features,
)


FEATURE_COLUMNS = ENHANCED_FEATURE_COLUMNS


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

                route_load_used = instance.vehicle_capacity - remaining_capacity

                features_df = build_candidate_features(
                    instance=instance,
                    current_point=current_point,
                    feasible_candidates=feasible_candidates,
                    unvisited=unvisited,
                    remaining_capacity=remaining_capacity,
                    vehicle_id=len(routes),
                    route_position=len(route),
                    route_load_used=route_load_used,
                )

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