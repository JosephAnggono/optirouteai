from fastapi.testclient import TestClient

from optirouteai.api.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "OptiRouteAI API"


def test_solve_nearest_neighbor():
    payload = {
        "num_customers": 8,
        "num_vehicles": 3,
        "vehicle_capacity": 30,
        "demand_min": 1,
        "demand_max": 6,
        "seed": 42,
        "solver": "nearest_neighbor",
        "alpha": 0.5,
        "ortools_time_limit_seconds": 1,
    }

    response = client.post("/solve", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["solution"]["solver_name"] == "Nearest Neighbor"
    assert data["solution"]["is_feasible"] is True
    assert isinstance(data["solution"]["routes"], list)


def test_compare_solvers_basic():
    payload = {
        "num_customers": 8,
        "num_vehicles": 3,
        "vehicle_capacity": 30,
        "demand_min": 1,
        "demand_max": 6,
        "seed": 42,
    }

    response = client.post("/compare", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "results" in data
    assert len(data["results"]) >= 2

    solver_names = {item["solver_name"] for item in data["results"]}

    assert "Nearest Neighbor" in solver_names
    assert "OR-Tools" in solver_names