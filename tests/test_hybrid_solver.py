from pathlib import Path

import pytest

from optirouteai.data.generator import generate_cvrp_instance
from optirouteai.optimization.hybrid_solver import HybridNeuralSolver


def test_hybrid_solver_runs_if_checkpoint_exists():
    checkpoint_path = Path("models/mlp_policy_step9.pkl")

    if not checkpoint_path.exists():
        pytest.skip("Step 9 checkpoint not found. Run Step 9 first.")

    instance = generate_cvrp_instance(
        num_customers=8,
        num_vehicles=3,
        vehicle_capacity=30,
        demand_min=1,
        demand_max=6,
        seed=42,
    )

    solver = HybridNeuralSolver(
        checkpoint_path=str(checkpoint_path),
        alpha=0.5,
    )

    result = solver.solve(instance)

    assert result.solver_name == "Hybrid MLP alpha=0.5"
    assert isinstance(result.routes, list)
    assert result.runtime_seconds >= 0