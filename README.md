# FairAI Studio

FairAI Studio is a Streamlit application and CLI for auditing binary decision models for bias, applying mitigation, tuning decision thresholds, and exporting reusable fairness reports.

## Features

- Upload a CSV or use the included sample dataset
- Select the decision target, positive outcome, and protected attribute
- Run single-attribute or intersectional fairness audits
- Train a logistic regression model with protected attributes excluded from model features
- Compare baseline and mitigated models with:
  - accuracy
  - predicted positive rate gaps
  - true positive rate gaps
  - false positive rate gaps
  - per-group confusion matrices
- Apply mitigation with reweighting or resampling
- Review feature importance, risk scoring, warnings, insights, and recommendations
- Tune decision thresholds to compare fairness and accuracy tradeoffs
- Export JSON reports, executive HTML reports, report bundles, and model bundles
- Reload saved model bundles for single-row or batch prediction

## Install

```bash
pip install -r requirements.txt
```

## Launch The App

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit.

## Run A CLI Audit

```bash
python bias_analysis.py --csv sample_data.csv --target hired --sensitive gender --mitigation reweighting --output-dir artifacts --save-model-dir saved_model
```

For multiclass targets, choose the positive outcome and group the rest:

```bash
python bias_analysis.py --csv applications.csv --target application_status --sensitive gender --positive-label approved --negative-label other_outcomes --mitigation reweighting
```

For an intersectional audit, pass comma-separated protected columns:

```bash
python bias_analysis.py --csv sample_data.csv --target hired --sensitive gender --intersectional gender,age_group --mitigation reweighting
```

## Predict With A Saved Model

Single input:

```bash
python bias_analysis.py --load-model-dir saved_model --predict-json "{\"age_group\":\"30-50\",\"income\":72000,\"education_level\":\"Bachelor\"}"
```

Batch input:

```bash
python bias_analysis.py --load-model-dir saved_model --predict-csv new_applicants.csv
```

## Deploy On Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Make sure these files are committed:
   - `app.py`
   - `bias_analysis.py`
   - `requirements.txt`
   - `sample_data.csv`
   - `.streamlit/config.toml`
3. Create a new app on Streamlit Community Cloud.
4. Select the repo, branch, and main file path: `app.py`.
5. Deploy.

No secrets are required for the current app.

## Responsible Use

FairAI supports model auditing, monitoring, and documentation. It should not be used as the sole decision-maker for hiring, lending, healthcare, education, or other high-stakes outcomes.

## Notes

- The selected protected attribute, and any additional intersectional protected attributes, are excluded from training features.
- Fairness metrics are evaluated on the held-out test split.
- Threshold tuning changes the operating point of the mitigated model; it does not retrain the model.
