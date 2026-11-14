# MGUS to Myeloma Transformation Predictor

A standalone Streamlit application for exploring longitudinal blood-test records and estimating the time to transformation from MGUS to multiple myeloma. It supports a clinical-threshold heuristic and pre-trained regression models, interactive biomarker charts, and data-entry tools.

> **Clinical limitation:** This is a research/demo application, not a medical device or clinical decision-support system. Predictions are retrospective model estimates and must not be used to diagnose, treat, or make care decisions. Consult qualified clinicians for medical decisions.

## What is included

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI, patient-data handling, biomarker analysis, and prediction logic |
| `data/` | Local cohort CSVs required by the app; not included in this repository |
| `models/` | Local trained regression models; not included in this repository |
| `requirements.txt` | Python package requirements; scikit-learn is pinned to the model serialization version |

Patient data and trained model artifacts are excluded from this repository. Before running the app, provide the required cohort CSVs and any model files locally in `data/` and `models/`.

## Features

- Browse patients from either cohort or the combined cohort.
- Select a prediction cutoff date so only tests up to that date are analyzed.
- Choose from Lasso Regression (selected by default), the heuristic method, or any available local trained model:
  - Elastic Net
  - Gradient Boosting
  - Lasso Regression
  - Naive Mean
  - Random Forest
  - Ridge Regression
  - Seasonal-Naive Median
  - Support Vector Regression
- Review the estimated transformation date, risk assessment, contributing biomarkers, and progression charts.
- Explore historical biomarker results and add results from a CSV or manual entry.

The app initially selects **Lasso Regression**. To switch methods, open **Select Prediction Model** on the Prediction tab and choose another model or the heuristic method.

## Requirements

- Python 3.9 or newer
- `pip`
- Internet access for the initial dependency installation

The expected model files were serialized with scikit-learn 1.6.1, so `requirements.txt` pins that version. Pickle/joblib files can execute code when loaded; only use model files from a trusted source.

## Run locally

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally <http://localhost:8501>.

The app expects the cohort files `data/cancer_patients_89.csv` and `data/mgus_patients_104.csv`. Model selection is based on any `models/model_advanced_*.pkl` files present.

The app has a demonstration login:

- **Username:** `admin`
- **Password:** `admin`

These credentials are hard-coded for demonstration only. Do not expose this app to the public internet or use it for production without implementing appropriate authentication, authorization, privacy, and security controls.

## Data format and updates

The local cohort CSVs need the following columns:

`PatientID`, `UniqueID`, `DateTestTaken`, `TestCode`, `TestName`, `Result`, `Unit`, `Gender`, `Age`, `TransformationDate`

Dates are parsed as day-first. The app expects the eight key biomarkers `BAND`, `SKLRAT`, `CREA`, `ADCAL`, `HB`, `IGA`, `IGM`, and `B2MG` for its heuristic and progression summaries. Missing model features are filled with zero by the prediction code.

The Upload Results tab accepts CSV files containing at least `DateTestTaken`, `TestCode`, and `Result`. Confirming an upload appends the provided rows to the selected cohort CSV. Manual entries are appended there as well. These changes are local to the folder and modify the local cohort CSV; back up the original data before entering or importing records.

## How predictions work

The application calculates each key biomarker's latest value, observation count, variability, and trend over time. The heuristic combines threshold abnormalities and trends into a risk score and estimates time to transformation using its clinical-threshold logic.

The regression models instead receive 52 progression and metadata features derived from the biomarker histories. The selected model estimates days to transformation; the UI presents the prediction alongside the biomarker-based risk analysis. A model's estimate is not a calibrated probability of transformation.

## Layout

```text
myeloma_predictor_standalone/
├── README.md
├── app.py
├── requirements.txt
├── data/                       # local cohort CSVs (not included)
└── models/                     # local trained models (not included)
```
