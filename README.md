# Breast Cancer Prediction System

This project builds a machine learning-based healthcare prediction system for breast cancer diagnosis using Scikit-Learn.

It uses the Wisconsin Breast Cancer Diagnostic dataset included with Scikit-Learn. The model predicts whether a tumor is:

- `Benign`
- `Malignant`

## Project Structure

```text
Breast-Cancer-Prediction-System/
  .vscode/              VS Code run/debug setup
  data/                 Exported dataset and sample input CSV
  models/               Saved trained model
  reports/              Metrics and feature importance outputs
  src/                  Python source code
```

## Quick Start

Run these commands from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\prepare_data.py
python src\train.py
python src\evaluate.py
python src\predict.py --sample 0
python src\predict.py --input data\sample_patient.csv
```

If `python` is not recognized, use:

```powershell
.\.venv\Scripts\python.exe src\predict.py --sample 0
```

## Input

The input is clinical numeric tumor measurements such as:

- mean radius
- mean texture
- mean perimeter
- mean area
- mean smoothness
- mean compactness
- mean concavity
- mean concave points
- mean symmetry
- mean fractal dimension

The full dataset has 30 features.

## Output

The prediction output looks like:

```text
Prediction: Malignant
Confidence: 99.12%
```

This is an educational project and should not be used for real medical diagnosis.
