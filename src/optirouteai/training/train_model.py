from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS = [
    "dx",
    "dy",
    "distance_to_current",
    "customer_demand",
    "remaining_capacity",
    "demand_capacity_ratio",
    "remaining_capacity_after",
    "num_unvisited",
]


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load imitation learning dataset.

    Args:
        path: Path to imitation dataset CSV.

    Returns:
        Cleaned DataFrame.
    """
    df = pd.read_csv(path)

    required_columns = FEATURE_COLUMNS + ["label"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.dropna(subset=required_columns).copy()
    df["label"] = df["label"].astype(int)

    return df


def train_model(dataset: pd.DataFrame):
    """
    Train a logistic regression imitation model.

    Args:
        dataset: Imitation learning dataset.

    Returns:
        model: Trained sklearn pipeline.
        metrics: Dictionary of evaluation metrics.
    """
    X = dataset[FEATURE_COLUMNS]
    y = dataset["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "num_rows": len(dataset),
        "num_train_rows": len(X_train),
        "num_test_rows": len(X_test),
        "positive_labels": int(y.sum()),
        "negative_labels": int((y == 0).sum()),
        "positive_rate": float(y.mean()),
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
    }

    return model, metrics


def save_model(model, output_path: str = "models/model_step8.pkl") -> str:
    """
    Save trained model to disk.

    Args:
        model: Trained sklearn model.
        output_path: Output model path.

    Returns:
        Output path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)

    return str(path)