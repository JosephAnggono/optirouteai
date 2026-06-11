from optirouteai.training.train_mlp import (
    load_imitation_dataset,
    save_mlp_checkpoint,
    train_mlp_model,
)


def main():
    print("=== OptiRouteAI Step 9: Train PyTorch MLP Imitation Model ===")
    print()

    dataset_path = "data/processed/imitation_dataset_step7.csv"

    dataset = load_imitation_dataset(dataset_path)

    print("Dataset loaded successfully.")
    print(f"Total rows: {len(dataset):,}")
    print(f"Positive labels: {int(dataset['label'].sum()):,}")
    print(f"Negative labels: {int((dataset['label'] == 0).sum()):,}")
    print(f"Positive rate: {dataset['label'].mean():.4f}")
    print()

    model, scaler, metrics, history = train_mlp_model(
        dataset=dataset,
        hidden_dim=64,
        dropout=0.10,
        batch_size=256,
        learning_rate=1e-3,
        num_epochs=30,
        random_state=42,
    )

    print()
    print("Final model performance:")
    print(f"Train rows: {metrics['num_train_rows']:,}")
    print(f"Test rows: {metrics['num_test_rows']:,}")
    print(f"Train instances: {metrics['num_train_instances']}")
    print(f"Test instances: {metrics['num_test_instances']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"Decision Top-1 Accuracy: {metrics['decision_top1_accuracy']:.4f}")
    print()

    model_path = save_mlp_checkpoint(
        model=model,
        scaler=scaler,
        metrics=metrics,
        history=history,
        output_path="models/mlp_policy_step9.pkl",
    )

    print(f"Model checkpoint saved to: {model_path}")


if __name__ == "__main__":
    main()