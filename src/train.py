from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_utils import TARGET_NAMES, load_dataset


MODEL_PATH = Path("models/breast_cancer_model.joblib")
REPORT_PATH = Path("reports/training_report.txt")
IMPORTANCE_PATH = Path("reports/feature_importance.csv")


def main() -> None:
    x, y = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, random_state=42)),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
            ]
        ),
    }

    best_name = ""
    best_model = None
    best_accuracy = -1.0
    report_lines: list[str] = []

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        report_lines.append(f"{name} accuracy: {accuracy:.4f}")
        if accuracy > best_accuracy:
            best_name = name
            best_model = model
            best_accuracy = accuracy

    if best_model is None:
        raise RuntimeError("No model was trained.")

    predictions = best_model.predict(x_test)
    report_lines.append("")
    report_lines.append(f"Best model: {best_name}")
    report_lines.append(f"Best accuracy: {best_accuracy:.4f}")
    report_lines.append("")
    report_lines.append("Confusion Matrix:")
    report_lines.append(str(confusion_matrix(y_test, predictions)))
    report_lines.append("")
    report_lines.append("Classification Report:")
    report_lines.append(
        classification_report(
            y_test,
            predictions,
            target_names=[TARGET_NAMES[0], TARGET_NAMES[1]],
        )
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "accuracy": best_accuracy,
            "feature_names": list(x.columns),
            "target_names": TARGET_NAMES,
        },
        MODEL_PATH,
    )
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    save_feature_importance(best_model, list(x.columns))

    print("\n".join(report_lines))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")


def save_feature_importance(model: Pipeline, columns: list[str]) -> None:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        scores = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        scores = abs(estimator.coef_[0])
    else:
        return

    frame = pd.DataFrame({"feature": columns, "importance": scores})
    frame = frame.sort_values("importance", ascending=False)
    IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(IMPORTANCE_PATH, index=False)


if __name__ == "__main__":
    main()
