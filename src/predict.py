from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from data_utils import load_dataset


MODEL_PATH = Path("models/breast_cancer_model.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict breast cancer diagnosis.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", type=int, help="Predict a dataset sample by row index.")
    group.add_argument("--input", help="CSV file containing one or more patient rows.")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    target_names = artifact["target_names"]

    if args.sample is not None:
        features, actual = load_dataset()
        if args.sample < 0 or args.sample >= len(features):
            raise SystemExit(f"Sample index must be between 0 and {len(features) - 1}")
        input_data = features.iloc[[args.sample]]
        actual_label = target_names[int(actual.iloc[args.sample])]
    else:
        input_data = pd.read_csv(args.input)
        missing = [name for name in feature_names if name not in input_data.columns]
        if missing:
            raise SystemExit(f"Input CSV is missing columns: {missing}")
        input_data = input_data[feature_names]
        actual_label = None

    probabilities = model.predict_proba(input_data)
    predictions = model.predict(input_data)

    for row_index, prediction in enumerate(predictions):
        label = target_names[int(prediction)]
        confidence = probabilities[row_index, int(prediction)] * 100
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2f}%")
        if actual_label is not None:
            print(f"Actual Label: {actual_label}")
        if args.verbose:
            print("Probabilities:")
            for class_index, class_name in target_names.items():
                print(f"  {class_name}: {probabilities[row_index, int(class_index)] * 100:.2f}%")


if __name__ == "__main__":
    main()
