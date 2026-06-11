from pathlib import Path

from optirouteai.data.generator import generate_cvrp_instance
from optirouteai.data.imitation_dataset import (
    build_imitation_examples_from_solution,
    generate_imitation_dataset,
    save_imitation_dataset,
)
from optirouteai.optimization.ortools_solver import solve_cvrp_with_ortools


def test_build_imitation_examples_from_solution():
    instance = generate_cvrp_instance(
        num_customers=8,
        num_vehicles=3,
        vehicle_capacity=30,
        demand_min=1,
        demand_max=6,
        seed=42,
    )

    result = solve_cvrp_with_ortools(
        instance=instance,
        time_limit_seconds=1,
    )

    examples = build_imitation_examples_from_solution(
        instance=instance,
        routes=result.routes,
        instance_id=0,
    )

    assert not examples.empty
    assert "decision_id" in examples.columns
    assert "label" in examples.columns
    assert set(examples["label"].unique()).issubset({0, 1})


def test_each_decision_has_one_positive_label():
    dataset = generate_imitation_dataset(
        num_instances=2,
        num_customers=8,
        num_vehicles=3,
        vehicle_capacity=30,
        demand_min=1,
        demand_max=6,
        base_seed=42,
        ortools_time_limit_seconds=1,
    )

    assert not dataset.empty

    positive_count_per_decision = dataset.groupby("decision_id")["label"].sum()

    assert (positive_count_per_decision == 1).all()


def test_save_imitation_dataset_creates_file(tmp_path):
    dataset = generate_imitation_dataset(
        num_instances=2,
        num_customers=8,
        num_vehicles=3,
        vehicle_capacity=30,
        demand_min=1,
        demand_max=6,
        base_seed=42,
        ortools_time_limit_seconds=1,
    )

    output_path = tmp_path / "imitation_dataset.csv"

    saved_path = save_imitation_dataset(
        dataset=dataset,
        output_path=str(output_path),
    )

    assert Path(saved_path).exists()