import math
import time

import pandas as pd

from optirouteai.data.generator import CVRPInstance
from optirouteai.optimization.constraints import check_solution_feasibility
from optirouteai.optimization.distance import euclidean_distance, solution_distance
from optirouteai.optimization.neural_solver import MLPNeuralSolver
from optirouteai.optimization.ortools_solver import SolverResult
from optirouteai.utils.routing_features import build_candidate_features

class HybridNeuralSolver:
    """
    Hybrid CVRP solver.

    Combines:
    1. MLP learned probability
    2. Distance-based nearest-neighbor score

    final_score = alpha * mlp_score + (1 - alpha) * distance_score
    """

    def __init__(
        self,
        checkpoint_path: str = "models/mlp_policy_step9.pkl",
        alpha: float = 0.5,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")

        self.alpha = alpha
        self.neural_solver = MLPNeuralSolver(checkpoint_path=checkpoint_path)

    @staticmethod
    def compute_distance_scores(features_df: pd.DataFrame) -> pd.Series:
        """
        Convert distance into a score where closer customers receive higher scores.
        """
        distances = features_df["distance_to_current"]

        min_dist = distances.min()
        max_dist = distances.max()

        if max_dist == min_dist:
            return pd.Series(1.0, index=features_df.index)

        distance_scores = 1.0 - ((distances - min_dist) / (max_dist - min_dist))

        return distance_scores

    def solve(self, instance: CVRPInstance) -> SolverResult:
        """
        Solve CVRP using hybrid neural + distance scoring.
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

                mlp_scores = self.neural_solver.score_candidates(features_df)
                distance_scores = self.compute_distance_scores(features_df)

                final_scores = (
                    self.alpha * mlp_scores
                    + (1.0 - self.alpha) * distance_scores
                )

                best_row_idx = final_scores.idxmax()
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
            solver_name=f"Hybrid MLP alpha={self.alpha}",
            routes=routes,
            total_distance=total_distance,
            runtime_seconds=runtime_seconds,
            is_feasible=feasibility_report.is_feasible,
        )