from optirouteai.data.generator import generate_cvrp_instance
from optirouteai.utils.routing_features import (
    ENHANCED_FEATURE_COLUMNS,
    build_candidate_features,
)


def test_build_candidate_features_contains_enhanced_columns():
    instance = generate_cvrp_instance(
        num_customers=8,
        num_vehicles=3,
        vehicle_capacity=30,
        demand_min=1,
        demand_max=6,
        seed=42,
    )

    unvisited = set(range(len(instance.customers)))
    feasible_candidates = sorted(list(unvisited))

    features = build_candidate_features(
        instance=instance,
        current_point=instance.depot,
        feasible_candidates=feasible_candidates,
        unvisited=unvisited,
        remaining_capacity=instance.vehicle_capacity,
        vehicle_id=0,
        route_position=0,
        route_load_used=0,
    )

    assert not features.empty

    for col in ENHANCED_FEATURE_COLUMNS:
        assert col in features.columns

    assert "customer_idx" in features.columns