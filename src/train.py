import mlflow
import mlflow.sklearn
import joblib
import yaml
import json
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def train():
    params = load_params()

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["model"]["test_size"],
        random_state=params["model"]["random_state"]
    )

    # Train
    model = RandomForestClassifier(
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depth"],
        random_state=params["model"]["random_state"]
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Save model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    # Save metrics
    Path("results").mkdir(exist_ok=True)
    metrics = {"accuracy": round(accuracy, 4), "f1": round(f1, 4)}
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Model saved to models/model.pkl")
    print("Metrics saved to results/metrics.json")


if __name__ == "__main__":
    train()