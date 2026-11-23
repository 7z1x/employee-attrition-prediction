<h1 align="center">Human-Resources-Problem</h1>

<p align="center">
  Predictive analytics solution for employee attrition â€” XGBoost model + Metabase dashboard to help HR reduce turnover.<br>
  <strong><a href="#usage">Explore usage Â»</a></strong>
  <br><br>
  <a href="https://github.com/7z1x/Human-Resources-Problem/issues/new?labels=bug">Report bug</a>
  Â·
  <a href="https://github.com/7z1x/Human-Resources-Problem/issues/new?labels=enhancement">Request feature</a>
  Â·
  <a href="https://github.com/7z1x/Human-Resources-Problem">Repository</a>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://xgboost.ai/"><img src="https://img.shields.io/badge/Model-XGBoost-brightgreen?logo=xgboost&logoColor=white" alt="XGBoost"></a>
  <a href="https://www.metabase.com/"><img src="https://img.shields.io/badge/Dashboard-Metabase-509EE3?logo=metabase&logoColor=white" alt="Metabase"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn"></a>
</p>

---

## Overview

**Human-Resources-Problem** is an end-to-end data science project that tackles high
employee attrition at **Jaya Jaya Maju**, a multinational company with over 1,000
employees and an attrition rate above 10%. The project combines exploratory data
analysis, a predictive **XGBoost** classification model, and a **Metabase** business
dashboard so the HR team can monitor turnover drivers and act proactively.

### How it works

1. **Data ingestion** â€” employee records (`employee_data.csv`) are loaded and cleaned;
   missing `Attrition` labels are imputed with the mode.
2. **Feature engineering** â€” categorical features are one-hot encoded; feature names are
   normalized to a readable title-cased format.
3. **Scaling & balancing** â€” `StandardScaler` normalizes features, then **SMOTE**
   oversamples the minority class to address attrition imbalance.
4. **Prediction** â€” a trained `XGBoost` model (persisted as `model_attrition_xgboost.pkl`)
   scores each employee and appends an `Attrition Prediction` column.
5. **Output** â€” predictions are exported to `predicted_attrition.csv` for downstream use.
6. **Dashboarding** â€” a Metabase instance visualizes summary metrics, organizational
   structure, satisfaction, demographics, and workload factors driving attrition.

---

## Files

- **[README.md](README.md)** â€” the file you are currently reading.
- **[requirements.txt](requirements.txt)** â€” Python dependencies.
- **[predict.py](predict.py)** â€” end-to-end prediction script (load â†’ encode â†’ scale â†’ SMOTE â†’ predict â†’ export).
- **[notebook.ipynb](notebook.ipynb)** â€” EDA, model training, and evaluation notebook.
- **[dataset/employee_data.csv](dataset/employee_data.csv)** â€” source employee dataset.
- **[model/model_attrition_xgboost.pkl](model/model_attrition_xgboost.pkl)** â€” trained XGBoost model artifact.
- **[image/](image/)** â€” dashboard screenshots:
  - `zulfahmi17z_dashboard.jpg` â€” summary (total employees, total resign, resign %).
  - `zulfahmi17z_dashboard(1).jpg` â€” organizational structure (dept, job level, job role).
  - `zulfahmi17z_dashboard(2).jpg` â€” satisfaction & involvement comparison.
  - `zulfahmi17z_dashboard(3).jpg` â€” summary metrics panel.
  - `zulfahmi17z_dashboard(4).jpg` â€” workload impact (OverTime, BusinessTravel).
- **[metabase.db.mv.db](metabase.db.mv.db)** â€” local Metabase H2 database (dashboard state).
- **[zulfahmi17z_video.mov](zulfahmi17z_video.mov)** â€” walkthrough video.

---

## Requirements

- **Python 3.10+**

| Package | Purpose |
|---|---|
| [xgboost](https://xgboost.ai/) | Gradient-boosted tree classifier |
| [scikit-learn](https://scikit-learn.org/) | `StandardScaler`, metrics, preprocessing |
| [imbalanced-learn](https://imbalanced-learn.org/) | `SMOTE` for class balancing |
| [pandas](https://pandas.pydata.org/) | Data manipulation |
| [numpy](https://numpy.org/) | Numerical operations |
| [joblib](https://joblib.readthedocs.io/) | Model serialization |
| [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) | Visualization |

---

## Installing

```bash
git clone https://github.com/7z1x/Human-Resources-Problem.git
cd Human-Resources-Problem
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### Run the prediction script

```bash
python predict.py
```

The script will:

1. Load `dataset/employee_data.csv`.
2. Impute missing `Attrition` values with the mode.
3. One-hot encode categorical features and normalize names.
4. Scale features with `StandardScaler` and balance classes with `SMOTE`.
5. Load `model/model_attrition_xgboost.pkl` and generate predictions.
6. Save results to `predicted_attrition.csv`.

### Explore the Metabase dashboard

A deployed dashboard is available at:

- **[Dashboard Metabase](https://metabase.turu.eu.org/dashboard/2-jaya-jaya-maju-zulfahmi)**

Login credentials:

- **Email:** `root@mail.com`
- **Password:** `root123`

The dashboard covers:

- **Summary** â€” total employees, total resign, resign percentage.
- **Organizational structure** â€” distribution by department, job level, and job role.
- **Satisfaction & involvement** â€” `EnvironmentSatisfaction`, `JobInvolvement`,
  `JobSatisfaction` for resigned vs. retained employees.
- **Demographics** â€” attrition by gender and marital status.
- **Workload** â€” impact of `OverTime` and `BusinessTravel` frequency on attrition.

---

## Key Insights

- **Overall attrition** â€” 179 of 1,470 employees resigned (12.18%).
- **OverTime** â€” employees working overtime have a **23.56%** resign rate vs. **7.69%**
  for those who do not.
- **Marital status** â€” single employees show the highest resign rate (~30%).
- **Department** â€” Sales has the highest attrition (14.8%), followed by HR (12%).
- **Travel** â€” frequent business travel raises resign rate to ~20%.
- **Satisfaction** â€” retained employees consistently report higher environment, job, and
  involvement satisfaction.

---

## Conclusion

Key drivers of attrition identified by the model and dashboard include **overtime status**,
**stock option level**, **job role/level**, **environment satisfaction**, and **job
involvement**. Employees without overtime, higher stock options, and higher job levels
tend to stay. The company should focus on overtime policy, benefits, satisfaction programs,
and travel flexibility to reduce turnover.

---

## Acknowledgements

Built as a final project for the Dicoding *Belajar Penerapan Data Science* course using
[XGBoost](https://xgboost.ai/), [scikit-learn](https://scikit-learn.org/),
[Metabase](https://www.metabase.com/), and the
[Dicoding employee dataset](https://github.com/dicodingacademy/dicoding_dataset).
