import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.streamlit_dashboard import build_results_table
from optirouteai.data.generator import generate_cvrp_instance
from optirouteai.evaluation.benchmark import solve_with_nearest_neighbor


def test_build_results_table():
    instance = generate_cvrp_instance(
        num_customers=8,
        num_vehicles=3,
        vehicle_capacity=30,
        demand_min=1,
        demand_max=6,
        seed=42,
    )

    result = solve_with_nearest_neighbor(instance)

    table = build_results_table([result])

    assert not table.empty
    assert "Solver" in table.columns
    assert "Total Distance" in table.columns
    assert "Runtime (s)" in table.columns
    assert "Feasible" in table.columns