from optirouteai.data.generator import generate_cvrp_instance
from optirouteai.optimization.constraints import check_solution_feasibility
from optirouteai.optimization.distance import route_distance, solution_distance
from optirouteai.optimization.heuristics import capacity_aware_nearest_neighbor


def main():
    instance = generate_cvrp_instance(
        num_customers=10,
        num_vehicles=3,
        vehicle_capacity=30,
        seed=42,
    )

    print("=== OptiRouteAI Step 3 Demo ===")
    print()

    print("Depot:")
    print(instance.depot)
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

    routes = capacity_aware_nearest_neighbor(instance)

    print("Generated routes:")
    print(routes)
    print()

    for vehicle_id, route in enumerate(routes, start=1):
        distance = route_distance(route, instance)
        load = sum(instance.demands[idx] for idx in route)

        print(f"Vehicle {vehicle_id}")
        print(f"  Route: {route}")
        print(f"  Load: {load}/{instance.vehicle_capacity}")
        print(f"  Distance: {distance:.4f}")
        print()

    total_distance = solution_distance(routes, instance)
    print(f"Total solution distance: {total_distance:.4f}")
    print()

    report = check_solution_feasibility(routes, instance)

    print("Feasibility report:")
    print(f"Is feasible: {report.is_feasible}")
    print(f"All customers visited once: {report.all_customers_visited_once}")
    print(f"Capacity constraints satisfied: {report.capacity_constraints_satisfied}")
    print(f"Missing customers: {report.num_missing_customers}")
    print(f"Duplicate visits: {report.num_duplicate_visits}")
    print(f"Capacity violations: {report.num_capacity_violations}")
    print(f"Route loads: {report.route_loads}")


if __name__ == "__main__":
    main()