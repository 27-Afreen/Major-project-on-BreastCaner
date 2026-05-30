from __future__ import annotations

from pathlib import Path

from data_utils import export_dataset, feature_names, load_dataset


def main() -> None:
    data_path = Path("data/breast_cancer.csv")
    sample_path = Path("data/sample_patient.csv")

    data = export_dataset(data_path)
    features, _ = load_dataset()
    sample = features.iloc[[0]].copy()
    sample.to_csv(sample_path, index=False)

    print(f"Exported dataset: {data_path}")
    print(f"Rows: {len(data)}")
    print(f"Features: {len(feature_names())}")
    print(f"Sample input CSV: {sample_path}")


if __name__ == "__main__":
    main()
