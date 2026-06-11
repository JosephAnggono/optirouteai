from optirouteai.training.train_mlp import (
    load_imitation_dataset,
    train_mlp_model,
)


def test_train_mlp_model_runs():
    dataset = load_imitation_dataset(
        "data/processed/imitation_dataset_step7.csv"
    )

    # Use a small sample for fast testing.
    sample_size = min(1200, len(dataset))
    dataset_sample = dataset.sample(
        n=sample_size,
        random_state=42,
    )

    model, scaler, metrics, history = train_mlp_model(
        dataset=dataset_sample,
        hidden_dim=32,
        dropout=0.10,
        batch_size=128,
        learning_rate=1e-3,
        num_epochs=2,
        random_state=42,
    )

    assert model is not None
    assert scaler is not None
    assert not history.empty

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["average_precision"] <= 1.0
    assert 0.0 <= metrics["decision_top1_accuracy"] <= 1.0