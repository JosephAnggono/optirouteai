from optirouteai.training.train_model import (
    load_dataset,
    save_model,
    train_model,
)


def main():
    print("=== OptiRouteAI Step 8: Train Imitation Learning Model ===")
    print()

    dataset_path = "data/processed/imitation_dataset_step7.csv"

    dataset = load_dataset(dataset_path)

    print("Dataset loaded successfully.")
    print(f"Total rows: {len(dataset):,}")
    print(f"Positive labels: {int(dataset['label'].sum()):,}")
    print(f"Negative labels: {int((dataset['label'] == 0).sum()):,}")
    print(f"Positive rate: {dataset['label'].mean():.4f}")
    print()

    model, metrics = train_model(dataset)

    print("Model performance:")
    print(f"Train rows: {metrics['num_train_rows']:,}")
    print(f"Test rows: {metrics['num_test_rows']:,}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC:  {metrics['roc_auc']:.4f}")
    print()

    model_path = save_model(model)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()