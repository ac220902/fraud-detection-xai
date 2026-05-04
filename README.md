# fraud-detection-xai

**Credit Card Fraud Detection with Explainable AI (SHAP)**

A production-grade machine learning pipeline for detecting fraudulent credit card transactions, with full model explainability using SHAP — directly aligned with UK/EU financial AI regulation requirements.

---

## Overview

This project trains an **XGBoost classifier** on 284,807 real credit card transactions (ULB dataset) and uses **SHAP (SHapley Additive exPlanations)** to explain every prediction at both the global and individual transaction level.

The combination of high-performance fraud detection and regulatory-grade explainability makes this approach directly applicable to production systems at banks and fintechs operating under FCA Consumer Duty and the EU AI Act.

---

## Key Results

| Metric | Score |
|--------|-------|
| ROC-AUC | **≥ 0.98** |
| AUPRC (Area Under Precision-Recall Curve) | **≥ 0.85** |
| Recall (Fraud class) | **≥ 0.85** |
| Precision (Fraud class) | **≥ 0.85** |

*Exact scores vary slightly by run; reproducibility seed is set to 42.*

---

## Visualisations

### SHAP Summary Plot (Beeswarm)
Shows which features most strongly drive fraud predictions across the entire test set. Each dot is one transaction, coloured by feature value (red = high, blue = low).

### SHAP Waterfall Plot
Explains a single high-confidence fraud prediction step-by-step — showing exactly how each feature pushed the score above or below the baseline. This is the kind of output an analyst would see when reviewing a flagged transaction.

### SHAP Dependence Plot
Reveals how the most important feature's influence changes across its value range, and which secondary feature it interacts with most.

### Precision-Recall Curve
The correct evaluation lens for imbalanced fraud data — shows the tradeoff between catching more fraud (recall) and maintaining investigator trust (precision).

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.11 |
| ML Framework | XGBoost 2.x |
| Explainability | SHAP 0.44+ |
| Imbalance handling | SMOTE (imbalanced-learn) |
| Data / EDA | pandas, NumPy, seaborn |
| Evaluation | scikit-learn |
| Environment | Jupyter Notebook |

---

## Dataset

**ULB Credit Card Fraud Detection** — Kaggle  
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions over 2 days (September 2013, European cardholders)
- 492 fraud cases (0.172% of all transactions)
- Features V1–V28: PCA-transformed (anonymised for privacy)
- Features Time, Amount: original values

> The dataset is **not included** in this repository (Kaggle terms of service). Download it separately and place `creditcard.csv` in the `data/` folder before running the notebook.

---

## Project Structure

```
fraud-detection-xai/
├── notebooks/
│   └── fraud_detection_xai.ipynb   # Main analysis notebook
├── data/
│   └── .gitkeep                    # Place creditcard.csv here
├── outputs/
│   ├── plots/                      # All generated visualisations
│   └── models/                     # Saved model artefacts
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/fraud-detection-xai.git
cd fraud-detection-xai

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset from Kaggle and place it in data/
#    → https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#    Place creditcard.csv in the data/ folder

# 5. Launch Jupyter and open the notebook
jupyter notebook notebooks/fraud_detection_xai.ipynb
```

Run all cells top to bottom. Outputs are saved to `outputs/plots/` and `outputs/models/`.

---

## Why SHAP for Fraud Detection?

Financial institutions face a dual challenge: **detect fraud accurately** and **explain decisions to regulators**.

SHAP solves the second problem by providing:

- **Per-transaction audit trails** — every flag can be explained with feature-level contributions
- **Global model transparency** — summary plots show which signals drive fraud detection across the whole portfolio
- **Data drift detection** — shifts in SHAP value distributions indicate the model is seeing new patterns and may need retraining
- **Regulatory compliance** — directly satisfies the explainability requirements in FCA Consumer Duty (2023) and EU AI Act (2024)

---

## Connection to Dissertation Work

This project extends my MSc dissertation on SHAP-based XAI by applying the same explainability framework to a real-world, highly imbalanced binary classification problem — the exact setting where model transparency is most critical and least often implemented.

The dissertation explored SHAP's theoretical properties (local accuracy, consistency, missingness). This project demonstrates those properties in a concrete financial use case with a dataset scale representative of production systems.

---

## Author

**Aditi Chauhan**  
MSc Data Science | XAI Researcher | Former Oracle & Infosys  
[LinkedIn](https://linkedin.com) · [GitHub](https://github.com)

---

*Dataset credit: Machine Learning Group, Université Libre de Bruxelles (ULB)*
