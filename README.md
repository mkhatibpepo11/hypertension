# Hypertension Classification with Machine Learning

An **educational** binary classification project using a **synthetic** hypertension dataset. This repository rebuilds an earlier exploratory notebook into a more reproducible training workflow and a small Streamlit demonstration.

> **Safety:** Not a medical device, diagnostic tool, or clinically validated risk estimator. Do not use for patient care.

## What it does

- Loads the original Hugging Face dataset and uses its existing train, validation, and test splits.
- Excludes columns that may reveal the outcome or are unsuitable as predictive inputs.
- Fits missing-value handling and one-hot encoding **on training data only**.
- Trains a Random Forest classifier and reports precision, recall, F1, ROC-AUC, and a confusion matrix.
- Saves the entire preprocessing + model pipeline for the demo app.

## Quick start

Requires Python 3.10+ and internet access to download the dataset on first run.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
streamlit run app/streamlit_app.py
```

Training creates `models/model.joblib` and `models/metrics.json` locally. These files are excluded from Git by default. The Streamlit demo needs the saved model file.

## Project structure

```text
app/streamlit_app.py        Demo interface
src/train.py               Training and evaluation pipeline
notebooks/original_project.ipynb  Original exploratory work (archive)
docs/PROJECT_NOTES.md      Methodology, limitations, and improvement plan
models/                    Generated model and metrics (not committed)
```

## Results

The original notebook contains saved metrics, but **the rebuilt pipeline has not yet been executed and independently verified**. Do not publish the old metrics as results of this rebuilt version. Run `python src/train.py`, then copy verified metrics from `models/metrics.json` into this section.

## Key limitations

- Synthetic data may not reflect real patients or clinical practice.
- Excluding blood-pressure and diagnosis-related columns reduces obvious target leakage, but does not prove that the remaining variables are free of leakage.
- The original 0.30 classification threshold is used for comparison only; it is not clinically validated or re-optimized.
- No SMOTE is used in the rebuilt baseline. This avoids applying regular SMOTE to encoded categorical features without validating the resulting synthetic values.
- The demo accepts raw values and is meant to illustrate the pipeline, not provide health guidance.

## Data source

[Hugging Face synthetic hypertension dataset](https://huggingface.co/datasets/electricsheepafrica/africa-synth-hypertension-hypertension-cvd-dataset-all). Check the dataset card and license before redistributing data.

## Original work

The archived notebook is included for traceability. Its saved outputs are historical and may depend on earlier notebook session state. The rebuilt baseline is a separate implementation, not an exact reproduction.
