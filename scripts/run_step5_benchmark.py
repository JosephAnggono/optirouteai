from optirouteai.data.generator import generate_cvrp_instance
from optirouteai.evaluation.benchmark import (
    benchmark_solver_results,
    solve_with_nearest_neighbor,
)
from optirouteai.optimization.constraints import check_solution_feasibility
from optirouteai.optimization.ortools_solver import solve_cvrp_with_ortools
from optirouteai.visualization.plot_routes import plot_routes


def print_routes(name, routes, instance):
    print(f"{name} routes:")
    for vehicle_id, route in enumerate(routes, start=1):
        load = sum(instance.demands[idx] for idx in route)
        print(f"  Vehicle {vehicle_id}: route={route}, load={load}/{instance.vehicle_capacity}")
    print()


def main():
    instance = generate_cvrp_instance(
        num_customers=25,
        num_vehicles=4,
        vehicle_capacity=40,
        seed=42,
    )

    print("=== OptiRouteAI Step 5 Benchmark ===")
    print()
    print(f"Customers: {len(instance.customers)}")
    print(f"Vehicles: {instance.num_vehicles}")
    print(f"Vehicle capacity: {instance.vehicle_capacity}")
    print(f"Total demand: {instance.demands.sum()}")
    print(f"Total vehicle capacity: {instance.num_vehicles * instance.vehicle_capacity}")
    print()

    nearest_neighbor_result = solve_with_nearest_neighbor(instance)

    ortools_result = solve_cvrp_with_ortools(
        instance=instance,
        time_limit_seconds=5,
    )

    solver_results = [
        nearest_neighbor_result,
        ortools_result,
    ]

    benchmark_results = benchmark_solver_results(
        solver_results=solver_results,
        reference_solver_name="OR-Tools",
    )

    print("Benchmark results:")
    print("-" * 90)
    print(
        f"{'Solver':<20} "
        f"{'Distance':>12} "
        f"{'Runtime(s)':>12} "
        f"{'Feasible':>12} "
        f"{'Gap vs OR-Tools':>18}"
    )
    print("-" * 90)

    for result in benchmark_results:
        gap_str = (
            "N/A"
            if result.gap_vs_reference_percent is None
            else f"{result.gap_vs_reference_percent:.2f}%"
        )

        print(
            f"{result.solver_name:<20} "
            f"{result.total_distance:>12.4f} "
            f"{result.runtime_seconds:>12.4f} "
            f"{str(result.is_feasible):>12} "
            f"{gap_str:>18}"
        )

    print("-" * 90)
    print()

    print_routes("Nearest Neighbor", nearest_neighbor_result.routes, instance)
    print_routes("OR-Tools", ortools_result.routes, instance)

    nn_report = check_solution_feasibility(nearest_neighbor_result.routes, instance)
    ortools_report = check_solution_feasibility(ortools_result.routes, instance)

    print("Nearest Neighbor feasibility:", nn_report)
    print()
    print("OR-Tools feasibility:", ortools_report)
    print()

    plot_routes(
        instance=instance,
        routes=nearest_neighbor_result.routes,
        title="Nearest Neighbor CVRP Solution",
        save_path="reports/figures/step5_nearest_neighbor_solution.png",
    )

    plot_routes(
        instance=instance,
        routes=ortools_result.routes,
        title="OR-Tools CVRP Solution",
        save_path="reports/figures/step5_ortools_solution.png",
    )


if __name__ == "__main__":
    main()