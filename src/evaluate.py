import joblib
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def evaluate():
    params = load_params()

    # Load the trained model
    model = joblib.load("models/model.pkl")

    # Recreate the exact same test split as train.py
    iris = load_iris()
    X, y = iris.data, iris.target
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=params["model"]["test_size"],
        random_state=params["model"]["random_state"]
    )

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=iris.target_names))

    # Save metrics to JSON
    Path("results").mkdir(exist_ok=True)
    metrics = {
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4)
    }
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Metrics saved to results/metrics.json")

    # Generate and save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=100)
    plt.close()
    print("Confusion matrix saved to results/confusion_matrix.png")


if __name__ == "__main__":
    evaluate()