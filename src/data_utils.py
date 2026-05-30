from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


TARGET_NAMES = {0: "Malignant", 1: "Benign"}


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    dataset = load_breast_cancer()
    features = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    target = pd.Series(dataset.target, name="target")
    return features, target


def export_dataset(output_path: str | Path) -> pd.DataFrame:
    features, target = load_dataset()
    data = features.copy()
    data["target"] = target
    data["diagnosis"] = target.map(TARGET_NAMES)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return data


def feature_names() -> list[str]:
    features, _ = load_dataset()
    return list(features.columns)
