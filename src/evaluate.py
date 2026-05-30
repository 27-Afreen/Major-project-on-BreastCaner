from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from data_utils import TARGET_NAMES, load_dataset


MODEL_PATH = Path("models/breast_cancer_model.joblib")


def main() -> None:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    x, y = load_dataset()
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    predictions = model.predict(x_test)
    print(f"Model: {artifact['model_name']}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=[TARGET_NAMES[0], TARGET_NAMES[1]]))


if __name__ == "__main__":
    main()
