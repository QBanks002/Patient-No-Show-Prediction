# Patient No-Show Prediction

Predicting appointment no-shows from a 110,000-row medical dataset using logistic regression and random forest classifiers. Built to help scheduling teams identify high-risk patients and reduce wasted clinic time.

---

## Business Question

*Which patient segments — by age, location, or health status — are at highest risk for no-shows, and how can we adjust scheduling accordingly?*

---

## Dataset

[Medical Appointment No-Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments) — Kaggle (public domain)  
110,527 appointments from Brazilian clinics, May 2016.  
Overall no-show rate: **20.2%**

---

## Key Findings

| Factor | No-show rate | Insight |
|---|---|---|
| Same-day booking | 4.6% | Lowest risk by far |
| Booked 31–60 days out | 34.1% | Highest risk group |
| Age 18–34 | 24.0% | Young adults miss most |
| Age 55–74 | 15.7% | Older patients most reliable |
| Saturday appointments | 23.1% | Higher risk than weekdays |

---

## Models

| Model | AUC | Notes |
|---|---|---|
| Logistic Regression | baseline | Interpretable coefficients |
| Random Forest | best | Feature importance |

Both models trained with `class_weight='balanced'` to handle the 80/20 class imbalance.

---

## Repo Structure

```
noshow-prediction/
├── data/
│   └── KaggleV2-May-2016.csv        # Raw dataset from Kaggle
├── notebooks/
│   └── noshow_analysis.ipynb        # Full walkthrough: cleaning → modeling → viz
├── src/
│   ├── clean.py                     # Data cleaning pipeline
│   └── model.py                     # Model training, evaluation, risk scoring
├── outputs/
│   ├── roc_pr_curves.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   └── segment_visualizations.png
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/your-username/noshow-prediction.git
cd noshow-prediction
pip install -r requirements.txt
```

Then open `notebooks/noshow_analysis.ipynb` and run all cells.

**Requirements:**
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## Pipeline Overview

1. **Clean** — fix column names, parse dates, drop invalid ages, remove scheduling errors
2. **Feature engineering** — lead time, age groups, health condition count, day of week
3. **Model** — logistic regression (baseline) + random forest (final)
4. **Evaluate** — ROC-AUC, precision-recall, confusion matrix
5. **Risk score** — assign Low / Medium / High tier to each appointment
6. **Recommend** — scheduling rules per risk tier

---

## Scheduling Recommendations

| Risk tier | Predicted probability | Action |
|---|---|---|
| Low | < 15% | Book normally |
| Medium | 15–35% | Send SMS reminder 48 hrs before |
| High | > 35% | Double-book slot or call to confirm |

---

## Tech Stack

Python · pandas · scikit-learn · matplotlib · seaborn 
