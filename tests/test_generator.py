from optirouteai.data.generator import generate_cvrp_instance


def test_generate_cvrp_instance_shapes():
    instance = generate_cvrp_instance(
        num_customers=10,
        num_vehicles=3,
        vehicle_capacity=30,
        seed=42,
    )

    assert instance.depot.shape == (2,)
    assert instance.customers.shape == (10, 2)
    assert instance.demands.shape == (10,)
    assert instance.num_vehicles == 3
    assert instance.vehicle_capacity == 30


def test_generate_cvrp_instance_demands_are_valid():
    instance = generate_cvrp_instance(
        num_customers=10,
        demand_min=1,
        demand_max=10,
        seed=42,
    )

    assert instance.demands.min() >= 1
    assert instance.demands.max() <= 10