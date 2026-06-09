from optirouteai.data.generator import generate_cvrp_instance


def main():
    instance = generate_cvrp_instance(
        num_customers=10,
        num_vehicles=3,
        vehicle_capacity=30,
        seed=42,
    )

    print("=== OptiRouteAI Step 1 Demo ===")
    print()
    print("Depot:")
    print(instance.depot)
    print()
    print("Customers:")
    print(instance.customers)
    print()
    print("Demands:")
    print(instance.demands)
    print()
    print("Number of vehicles:")
    print(instance.num_vehicles)
    print()
    print("Vehicle capacity:")
    print(instance.vehicle_capacity)
    print()
    print("Total demand:")
    print(instance.demands.sum())


if __name__ == "__main__":
    main()