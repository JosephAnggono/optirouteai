from optirouteai.data.imitation_dataset import (
    generate_imitation_dataset,
    save_imitation_dataset,
)


def main():
    print("=== OptiRouteAI Step 7: Generate Imitation Learning Dataset ===")
    print()

    dataset = generate_imitation_dataset(
        num_instances=20,
        num_customers=25,
        num_vehicles=5,
        vehicle_capacity=45,
        demand_min=1,
        demand_max=8,
        base_seed=42,
        ortools_time_limit_seconds=1,
    )

    if dataset.empty:
        print("No imitation dataset generated.")
        return

    output_path = save_imitation_dataset(
        dataset=dataset,
        output_path="data/processed/imitation_dataset_step7.csv",
    )

    num_rows = len(dataset)
    num_decisions = dataset["decision_id"].nunique()
    num_instances = dataset["instance_id"].nunique()
    positive_labels = int(dataset["label"].sum())
    negative_labels = int((dataset["label"] == 0).sum())

    print("Dataset summary:")
    print(f"Instances used:          {num_instances}")
    print(f"Decision steps:          {num_decisions}")
    print(f"Total candidate rows:    {num_rows}")
    print(f"Positive labels:         {positive_labels}")
    print(f"Negative labels:         {negative_labels}")
    print()

    print("Sample rows:")
    print(dataset.head())
    print()

    print(f"Saved imitation dataset to: {output_path}")


if __name__ == "__main__":
    main()