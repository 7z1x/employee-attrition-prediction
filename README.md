# 🧑‍💼 Employee Attrition Prediction — Jaya Jaya Maju

Predicting which employees are likely to resign, and turning those predictions into retention actions HR can actually execute.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-brightgreen?logo=xgboost)](https://xgboost.ai/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Metabase](https://img.shields.io/badge/Dashboard-Metabase-509EE3?logo=metabase&logoColor=white)](https://www.metabase.com/)

**📊 Live Dashboard:** [metabase.turu.eu.org](https://metabase.turu.eu.org/dashboard/2-jaya-jaya-maju-zulfahmi)

---

## 📌 Executive Summary

| Question | Answer |
|---|---|
| How big is the problem? | **12.18%** of employees resigned (179 of 1,470) |
| Can it be predicted? | Yes — **94.58%** accuracy and **0.91 recall** on the attrition class |
| What drives it most? | Overtime, stock option level, job level, and environment satisfaction |
| What should HR do? | Overtime policy review, equity-based retention, targeted check-ins for high-risk segments |

> **Key insight:** attrition is not random. Employees who work overtime resign at **3×** the rate of those who don't (31.9% vs 10.8%), and single employees resign at roughly **2×** the rate of married employees. These are addressable policy levers, not personality traits.

---

## 🎯 Business Understanding

### Background

Jaya Jaya Maju is a multinational company with over 1,000 employees and an attrition rate above 10% — high enough to be a recurring cost problem. Each resignation carries replacement cost, lost productivity during the vacancy, and knowledge drain from the team.

The company currently learns about attrition **after** the resignation letter arrives. HR has no systematic way to see risk building up months earlier.

### Business Problems

1. **No early warning** — there is no mechanism to flag employees whose profile matches past leavers.
2. **Untargeted retention spend** — retention budget is spread evenly, while risk is concentrated in specific segments (overtime workers, junior levels, frequent travelers).
3. **No shared visibility** — leadership has no dashboard tracking turnover drivers over time.

### Objectives

| # | Objective | Output |
|---|---|---|
| 1 | Identify the factors most associated with attrition | EDA + feature importance |
| 2 | Build a predictive model for attrition risk | XGBoost model (`.pkl`) |
| 3 | Produce a scored employee list for HR follow-up | `predicted_attrition.csv` |
| 4 | Give HR a monitoring view | Metabase dashboard |

### Success Metrics

- **Business metric:** reduction in annual attrition rate; share of high-risk employees who receive a retention conversation.
- **Model metric:** **recall on the attrition class** is prioritized over raw accuracy. A missed at-risk employee (false negative) costs a full replacement hire; a false alarm costs only one extra check-in.

### Scope

**Included:** historical analysis, predictive modeling, scoring script, business dashboard, retention recommendations.
**Excluded:** A/B testing on live employees, integration with the HRIS, compensation benchmarking, exit-interview analysis.

---

## 🗂️ Dataset

| Attribute | Detail |
|---|---|
| **Source** | [Dicoding employee dataset](https://github.com/dicodingacademy/dicoding_dataset) |
| **Rows** | 1,470 employees |
| **Columns** | 35 (34 features + target) |
| **Duplicate rows** | 0 |
| **Missing labels** | **412 rows** with no `Attrition` value |
| **Label encoding** | `1.0` = resigned, `0.0` = stayed |

**Target distribution (among the 1,058 labelled rows):**

| Class | Count | Share |
|---|---:|---:|
| Stayed | 879 | 83.08% |
| **Resigned** | **179** | **16.92%** |

> **Two different attrition rates appear in this project — both are correct:**
> - **16.92%** — among employees whose status is actually known.
> - **12.18%** — after imputing the 412 missing labels with the mode (`0.0`), which is the figure used in the dashboard.
>
> The gap matters: if those 412 unlabelled employees are actually people who already left (the most plausible reading — a label is usually blank *because* they exited), the true attrition rate is higher than either number and the model is trained on noisy labels. See [Limitations](#-limitations--assumptions).

**Feature groups:**

- **Demographics:** age, gender, marital status, distance from home
- **Job & role:** department, job role, job level, business travel, overtime
- **Compensation:** monthly income, daily/hourly/monthly rate, salary hike %, stock option level
- **Satisfaction & engagement:** environment satisfaction, job satisfaction, job involvement, relationship satisfaction, work-life balance
- **Tenure:** years at company, in current role, since last promotion, with current manager, total working years
- **Performance:** performance rating, training times last year, number of companies worked

---

## 🔬 Methodology

```
Data Loading → EDA → Cleaning → Encoding → Scaling + SMOTE → XGBoost → Evaluation → Scoring + Dashboard
```

1. **Data Loading** — `dataset/employee_data.csv`: 1,470 rows, 34 features.
2. **EDA** — target distribution, correlation heatmap, categorical and numerical breakdowns against attrition.
3. **Cleaning** — 412 missing `Attrition` labels imputed with the mode; zero duplicates found.
4. **Encoding** — categorical columns one-hot encoded (55 features after encoding).
5. **Scaling + SMOTE** — `StandardScaler`, then SMOTE to balance the minority class.
6. **Modeling** — `XGBClassifier` (100 trees, `max_depth=6`, `learning_rate=0.1`, `binary:logistic`).
7. **Evaluation** — confusion matrix, classification report, accuracy.
8. **Scoring + Dashboard** — batch scoring to `predicted_attrition.csv`, insights surfaced in Metabase.

> **Note on preprocessing order:** the notebook applies scaling and SMOTE *before* the train/test split. This is a known leakage pattern that inflates the test score — it is documented openly rather than hidden.

---

## 📊 Key Findings (EDA)

### 1. Overtime is the strongest actionable signal

| Overtime | Attrition Rate |
|---|---:|
| Yes | **31.92%** |
| No | 10.79% |

Working overtime nearly **triples** resignation risk. Unlike demographics, this is a lever HR can actually pull.

### 2. Career level and equity predict retention

| Feature | Stayed (mean) | Resigned (mean) |
|---|---:|---:|
| Job level | 2.12 | 1.66 |
| Stock option level | 0.84 | 0.46 |
| Environment satisfaction (1–4) | 2.77 | 2.39 |
| Job involvement (1–4) | 2.76 | 2.50 |
| Job satisfaction (1–4) | 2.76 | 2.53 |

Employees who leave sit lower on every one of these dimensions. **Stock option level** shows the widest gap relative to its scale — leavers hold roughly half the equity stake of stayers.

### 3. Single employees leave far more often

| Marital Status | Attrition Rate |
|---|---:|
| Single | **26.70%** |
| Married | 13.36% |
| Divorced | 9.50% |

### 4. Frequent travel carries clear risk

| Business Travel | Attrition Rate |
|---|---:|
| Travel_Frequently | **24.88%** |
| Travel_Rarely | 15.68% |
| Non-Travel | 10.28% |

### 5. Sales is the most exposed department

| Department | Attrition Rate |
|---|---:|
| Sales | **20.69%** |
| Human Resources | 15.79% |
| Research & Development | 15.26% |

### 6. Correlation structure

Attrition correlates negatively with `Age` (−0.17), `MonthlyIncome` (−0.16), and `YearsAtCompany` (−0.14) — younger, lower-paid, newer employees are more likely to leave. Note that `MonthlyIncome` and `JobLevel` correlate at ~0.95, so they carry largely the same signal.

---

## 🤖 Model Results

### Performance on the Test Set (n = 517)

| Metric | Value |
|---|---:|
| **Accuracy** | **94.58%** |
| Macro avg Precision | 0.95 |
| Macro avg Recall | 0.95 |
| Macro avg F1 | 0.95 |

**Per class:**

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Stayed (0) | 0.92 | 0.98 | 0.95 | 259 |
| **Resigned (1)** | **0.98** | **0.91** | **0.94** | 258 |

**How to read this for HR:** on this test set, the model correctly identified 91% of employees who resigned. But read the caveat below before quoting these numbers to leadership — the test set was balanced with synthetic data, so these scores will not hold on a real workforce where only ~12% of employees leave.

### Top 10 Features by Importance

| # | Feature | Importance |
|---:|---|---:|
| 1 | `OverTime_No` | 0.1407 |
| 2 | `StockOptionLevel` | 0.1073 |
| 3 | `JobRole_Manufacturing Director` | 0.0588 |
| 4 | `JobLevel` | 0.0586 |
| 5 | `EnvironmentSatisfaction` | 0.0455 |
| 6 | `JobInvolvement` | 0.0431 |
| 7 | `MaritalStatus_Single` | 0.0414 |
| 8 | `JobSatisfaction` | 0.0300 |
| 9 | `JobRole_Sales Executive` | 0.0282 |
| 10 | `BusinessTravel_Travel_Frequently` | 0.0238 |

The top two features — overtime and stock options — account for **~25%** of total importance, and both are policy-controlled. That is good news: the strongest drivers are the ones management can change.

---

## ⚠️ Limitations & Assumptions

Stated openly, because they determine how far these results can be trusted:

1. **The test set is not realistic.** SMOTE was applied before the split, so the test set is ~50/50 synthetic-balanced. Accuracy of 94.58% will **not** transfer to a workforce with a ~12% attrition rate; precision in production will be materially lower.
2. **Label noise is significant.** 412 of 1,470 rows (28%) had no `Attrition` value and were filled with the mode (`0.0`). If those employees actually left, the model is being trained to call leavers "stayed".
3. **Data leakage in preprocessing.** `StandardScaler` was fit on the full dataset and SMOTE was applied before `train_test_split`. Both let test-set information reach training.
4. **Degenerate features included.** `EmployeeCount`, `StandardHours`, and `Over18` each hold a single constant value; `EmployeeId` is a unique identifier. None carry signal — they add noise and inflate dimensionality.
5. **No hyperparameter tuning.** Parameters are fixed defaults, so this is a solid baseline rather than a tuned model.
6. **Default 0.5 threshold.** For retention triage, a lower threshold (e.g. 0.3–0.35) would raise recall and surface more at-risk employees, at the cost of more false alarms.
7. **No temporal validation.** The model has not been trained on older cohorts and tested on newer ones, so stability over time is unverified.
8. **Correlation is not causation.** Feature importance shows association, not cause. The recommendations below are hypotheses to pilot, not proven interventions.

---

## 💡 Recommended Action Items

Ordered by **impact × ease of execution**.

### Priority 1 — Overtime Policy Review (highest impact, directly controllable)

- Audit overtime distribution by department and role to find where it concentrates.
- Cap sustained overtime and require manager justification beyond a threshold.
- Pilot a **"no-overtime month"** in the highest-risk team and compare attrition intent before/after.
- Add overtime hours to the monthly HR risk review — it is the single strongest predictor.

### Priority 2 — Equity & Career Progression for Junior Levels

- Expand **stock option eligibility** downward. Leavers hold roughly half the equity of stayers (0.46 vs 0.84).
- Build a **visible promotion path for job levels 1–2**, where leavers cluster (mean level 1.66 vs 2.12).
- Run stay-interviews for employees in their first 2 years, when attrition risk peaks.

### Priority 3 — Targeted Engagement for High-Risk Segments

- **Single employees** (26.7% attrition): structured mentorship and peer-community programs.
- **Frequent travelers** (24.9%): review travel frequency, offer recovery time after trips, and rotate assignments.
- **Sales department** (20.7%): review quota pressure and incentive structure.
- **Environment satisfaction** scores below 2.5 should trigger a manager check-in automatically.

### Priority 4 — Make the Model Operational

- Score the workforce **quarterly** and deliver risk lists to HR business partners, not as a CSV nobody opens.
- Calibrate the decision threshold together with HR, based on how many retention conversations the team can realistically handle.
- Pair every score with **the top contributing factors** so managers know *what to discuss*, not just *who to watch*.

### Priority 5 — Governance & Measurement

- Form a cross-functional retention task force (HR, department heads, finance).
- Track intervention outcomes per cohort to measure real lift, not just model accuracy.
- Collect qualitative exit-interview data to validate — or challenge — the model's drivers.

### Suggested KPIs

| KPI | Baseline | Target (12 months) |
|---|---|---|
| Overall attrition rate | 12.18% | < 10% |
| Attrition among overtime workers | 23.56% | < 18% |
| Attrition among single employees | 20.00% | < 15% |
| High-risk employees receiving a retention conversation | — | ≥ 80% |
| Time from risk flag → first manager conversation | — | ≤ 14 days |

---

## 📈 Business Dashboard (Metabase)

**🔗 [Open the live dashboard](https://metabase.turu.eu.org/dashboard/2-jaya-jaya-maju-zulfahmi)**

The dashboard covers:

- **Summary** — total employees, total resignations, resignation percentage
- **Organizational structure** — distribution by department, job level, job role
- **Satisfaction & involvement** — environment, job, and involvement scores for resigned vs retained
- **Demographics** — attrition by gender and marital status
- **Workload** — impact of overtime and business travel frequency

**Screenshots:**

| Summary | Organizational Structure |
|---|---|
| ![Dashboard 1](image/zulfahmi17z_dashboard.jpg) | ![Dashboard 2](image/zulfahmi17z_dashboard%281%29.jpg) |

| Satisfaction & Demographics | Workload Factors |
|---|---|
| ![Dashboard 3](image/zulfahmi17z_dashboard%282%29.jpg) | ![Dashboard 4](image/zulfahmi17z_dashboard%284%29.jpg) |

**Demo credentials** (public demo instance, for evaluation only):

- Email: `root@mail.com`
- Password: `root123`

> ⚠️ These are demo credentials on a public instance. Do not reuse them for anything containing real employee data.

---

## 🚀 Getting Started

### 1. Prerequisites

- Python **3.10+** — check with `python --version`

### 2. Clone & install

```bash
git clone https://github.com/7z1x/employee-attrition-prediction.git
cd employee-attrition-prediction

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the scoring script

```bash
python predict.py
```

The script will:

1. Load `dataset/employee_data.csv`.
2. Impute missing `Attrition` labels with the mode.
3. One-hot encode categorical features and normalize column names.
4. Scale features and balance classes with SMOTE.
5. Load `model/model_attrition_xgboost.pkl` and generate predictions.
6. Write results to `predicted_attrition.csv`.

### 4. Reproduce the analysis

Open `notebook.ipynb` (Jupyter or Google Colab) to re-run the full pipeline: EDA → cleaning → encoding → modeling → evaluation → model export.

---

## 🗂️ Project Structure

```
employee-attrition-prediction/
│
├── dataset/
│   └── employee_data.csv              # 1,470 rows, 35 columns
│
├── model/
│   └── model_attrition_xgboost.pkl    # Trained XGBoost model
│
├── image/                             # Dashboard screenshots
│   ├── zulfahmi17z_dashboard.jpg
│   └── zulfahmi17z_dashboard(1..4).jpg
│
├── notebook.ipynb                     # EDA, modeling, evaluation
├── predict.py                         # Batch scoring script
├── metabase.db.mv.db                  # Metabase database (H2)
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data analysis | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Visualization | Matplotlib, Seaborn |
| Model persistence | joblib |
| Business intelligence | Metabase |

---

## 👤 Author

**Zulfahmi M. Ardianto**
Laskar ID: A548YBM523

---

## 📄 Acknowledgements

Built as a final project for the Dicoding *Belajar Penerapan Data Science* course, using [XGBoost](https://xgboost.ai/), [scikit-learn](https://scikit-learn.org/), and [Metabase](https://www.metabase.com/), with the [Dicoding employee dataset](https://github.com/dicodingacademy/dicoding_dataset).
