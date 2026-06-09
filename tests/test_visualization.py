from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from optirouteai.data.generator import generate_cvrp_instance
from optirouteai.optimization.heuristics import capacity_aware_nearest_neighbor
from optirouteai.visualization.plot_routes import plot_routes


def test_plot_routes_saves_file(tmp_path):
    instance = generate_cvrp_instance(
        num_customers=10,
        num_vehicles=3,
        vehicle_capacity=30,
        seed=42,
    )

    routes = capacity_aware_nearest_neighbor(instance)

    save_path = tmp_path / "routes.png"

    plot_routes(
        instance=instance,
        routes=routes,
        title="Test Route Plot",
        save_path=str(save_path),
        show=False,
    )

    assert Path(save_path).exists()