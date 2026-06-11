from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from optirouteai.models.mlp_policy import MLPPolicy
from optirouteai.utils.routing_features import ENHANCED_FEATURE_COLUMNS


FEATURE_COLUMNS = ENHANCED_FEATURE_COLUMNS


def load_imitation_dataset(path: str) -> pd.DataFrame:
    """
    Load imitation dataset for neural model training.
    """
    df = pd.read_csv(path)

    required_columns = FEATURE_COLUMNS + ["label", "decision_id", "instance_id"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.dropna(subset=required_columns).copy()
    df["label"] = df["label"].astype(int)

    return df


def split_by_instance(
    dataset: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by instance_id to avoid leakage.

    This is better than random row splitting because rows from the same routing
    problem should not appear in both train and test sets.
    """
    rng = np.random.default_rng(random_state)

    instance_ids = np.array(sorted(dataset["instance_id"].unique()))
    rng.shuffle(instance_ids)

    num_test = max(1, int(len(instance_ids) * test_size))

    test_ids = set(instance_ids[:num_test])
    train_ids = set(instance_ids[num_test:])

    train_df = dataset[dataset["instance_id"].isin(train_ids)].copy()
    test_df = dataset[dataset["instance_id"].isin(test_ids)].copy()

    return train_df, test_df


def prepare_tensors(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
    """
    Scale features and convert train/test data into PyTorch tensors.
    """
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    X_test = scaler.transform(test_df[FEATURE_COLUMNS])

    y_train = train_df["label"].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.float32)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler


def compute_decision_top1_accuracy(
    test_df: pd.DataFrame,
    probabilities: np.ndarray,
) -> float:
    """
    Compute decision-level top-1 accuracy.

    For each decision_id, choose the candidate with the highest predicted
    probability. The prediction is correct if that candidate has label = 1.

    This metric is more meaningful for routing than row-level accuracy.
    """
    eval_df = test_df.copy()
    eval_df["pred_prob"] = probabilities

    correct = 0
    total = 0

    for _, group in eval_df.groupby("decision_id"):
        best_idx = group["pred_prob"].idxmax()
        if int(eval_df.loc[best_idx, "label"]) == 1:
            correct += 1
        total += 1

    if total == 0:
        return 0.0

    return correct / total


def train_mlp_model(
    dataset: pd.DataFrame,
    hidden_dim: int = 64,
    dropout: float = 0.10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    num_epochs: int = 30,
    random_state: int = 42,
):
    """
    Train an MLP imitation model.
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    train_df, test_df = split_by_instance(
        dataset=dataset,
        test_size=0.2,
        random_state=random_state,
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
    ) = prepare_tensors(train_df, test_df)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    input_dim = X_train.shape[1]

    model = MLPPolicy(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    positive_count = float(y_train.sum().item())
    negative_count = float(len(y_train) - positive_count)

    if positive_count == 0:
        pos_weight = torch.tensor(1.0)
    else:
        pos_weight = torch.tensor(negative_count / positive_count)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            test_probabilities = torch.sigmoid(test_logits).numpy()
            test_predictions = (test_probabilities >= 0.5).astype(int)

        y_test_np = y_test.numpy().astype(int)

        accuracy = accuracy_score(y_test_np, test_predictions)
        roc_auc = roc_auc_score(y_test_np, test_probabilities)
        average_precision = average_precision_score(y_test_np, test_probabilities)
        top1_accuracy = compute_decision_top1_accuracy(test_df, test_probabilities)

        metrics = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc),
            "average_precision": float(average_precision),
            "decision_top1_accuracy": float(top1_accuracy),
        }

        history.append(metrics)

        if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={avg_loss:.4f} | "
                f"acc={accuracy:.4f} | "
                f"auc={roc_auc:.4f} | "
                f"ap={average_precision:.4f} | "
                f"top1={top1_accuracy:.4f}"
            )

    final_metrics = history[-1]
    final_metrics.update(
        {
            "num_rows": len(dataset),
            "num_train_rows": len(train_df),
            "num_test_rows": len(test_df),
            "positive_labels": int(dataset["label"].sum()),
            "negative_labels": int((dataset["label"] == 0).sum()),
            "positive_rate": float(dataset["label"].mean()),
            "num_train_instances": int(train_df["instance_id"].nunique()),
            "num_test_instances": int(test_df["instance_id"].nunique()),
        }
    )

    return model, scaler, final_metrics, pd.DataFrame(history)


def save_mlp_checkpoint(
    model,
    scaler,
    metrics: dict,
    history: pd.DataFrame,
    output_path: str = "models/mlp_policy_step9.pkl",
) -> str:
    """
    Save model checkpoint, scaler, metrics, and feature columns.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "scaler": scaler,
        "metrics": metrics,
        "history": history,
    }

    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)

    return str(path)