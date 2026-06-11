from optirouteai.evaluation.batch_benchmark import (
    run_batch_benchmark,
    save_batch_results,
    summarize_batch_results,
)


def main():
    print("=== OptiRouteAI Step 6: Batch Benchmark ===")
    print()

    results = run_batch_benchmark(
        num_instances=20,
        num_customers=25,
        num_vehicles=5,
        vehicle_capacity=45,
        demand_min=1,
        demand_max=8,
        base_seed=42,
        ortools_time_limit_seconds=1,
    )

    summary = summarize_batch_results(results)

    results_path, summary_path = save_batch_results(
        results=results,
        summary=summary,
        output_dir="reports",
    )

    print("Detailed benchmark results:")
    print(results.head())
    print()

    print("Benchmark summary:")
    print(summary.round(4))
    print()

    print(f"Saved detailed results to: {results_path}")
    print(f"Saved summary results to: {summary_path}")


if __name__ == "__main__":
    main()