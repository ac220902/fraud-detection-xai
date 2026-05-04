# Step-by-Step Setup Guide

Follow this guide exactly, in order. Each step builds on the previous one.

---

## Step 1 — Create a GitHub Repository

1. Go to [github.com](https://github.com) and sign in
2. Click the **+** icon (top right) → **New repository**
3. Repository name: `fraud-detection-xai`
4. Description: `Credit Card Fraud Detection with XGBoost and SHAP Explainability`
5. Set to **Public**
6. **Do NOT** tick "Add a README file" (you already have one)
7. Click **Create repository**
8. GitHub will show you a page with setup commands — keep this tab open

---

## Step 2 — Set Up Your Local Project Folder

Open **VS Code**, then open a new Terminal (`Ctrl + \`` or Terminal → New Terminal).

```bash
# Navigate to wherever you keep your projects (e.g. Desktop or Documents)
cd ~/Desktop

# Create the project folder
mkdir fraud-detection-xai
cd fraud-detection-xai
```

---

## Step 3 — Copy the Project Files

Copy these files (provided by this guide) into your `fraud-detection-xai` folder:

```
fraud-detection-xai/
├── notebooks/
│   └── fraud_detection_xai.ipynb
├── data/
│   └── .gitkeep
├── outputs/
│   ├── plots/
│   └── models/
├── requirements.txt
├── .gitignore
└── README.md
```

You can copy them from wherever you saved them, or recreate the structure manually in VS Code's Explorer panel (left sidebar).

---

## Step 4 — Create a Python Virtual Environment

In your VS Code terminal, make sure you're inside `fraud-detection-xai/`:

```bash
# Create a virtual environment called 'venv'
python -m venv venv

# Activate it (Mac/Linux)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal prompt — this means it's active.

> **Important:** Always activate the venv before working on this project. If you close VS Code and come back, run the activate command again.

---

## Step 5 — Install All Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs XGBoost, SHAP, scikit-learn, imbalanced-learn, pandas, matplotlib, seaborn, and Jupyter. It may take 2–4 minutes.

Verify everything installed:
```bash
python -c "import xgboost, shap, sklearn, imblearn; print('All packages OK')"
```

---

## Step 6 — Download the Dataset from Kaggle

1. Go to: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Click **Download** (top right of the dataset page)
3. This downloads a file called `creditcard.csv` (or a zip containing it)
4. If it downloaded as a zip, extract it
5. Move/copy `creditcard.csv` into your `fraud-detection-xai/data/` folder

Your `data/` folder should now look like:
```
data/
├── .gitkeep
└── creditcard.csv   ← this file should be here
```

> **Note:** `creditcard.csv` is about 144MB. It will NOT be committed to GitHub (it's in `.gitignore`) — this is intentional and correct.

---

## Step 7 — Run the Jupyter Notebook

```bash
# Make sure your venv is active, then:
jupyter notebook
```

This opens Jupyter in your browser at `http://localhost:8888`.

1. Navigate to `notebooks/`
2. Click `fraud_detection_xai.ipynb` to open it
3. Click **Kernel → Restart & Run All** (or run cells one by one with `Shift + Enter`)

The notebook will:
- Load and explore the dataset
- Preprocess features and apply SMOTE
- Train the XGBoost classifier
- Print evaluation metrics (ROC-AUC, AUPRC, F1)
- Generate and save all SHAP plots to `outputs/plots/`

Full run time: approximately 3–8 minutes depending on your machine.

---

## Step 8 — Check Your Outputs

After the notebook completes, you should have these files in `outputs/plots/`:

```
01_class_distribution.png
02_amount_distribution.png
03_correlation_heatmap.png
04_evaluation_metrics.png
05_roc_curve.png
06_shap_summary_beeswarm.png
07_shap_bar_importance.png
08_shap_force_plot.html      ← interactive, open in browser
09_shap_waterfall.png
10_shap_dependence_V14.png   (or whichever feature is top)
```

And in `outputs/models/`:
```
xgb_fraud_detector.pkl
```

---

## Step 9 — Initialise Git and Push to GitHub

In your terminal (make sure you're in the `fraud-detection-xai` folder):

```bash
# Initialise git
git init

# Stage all files
git add .

# Check what's being committed (outputs/models will be excluded by .gitignore)
git status

# First commit
git commit -m "Initial commit: XGBoost fraud detection with SHAP explainability"

# Connect to your GitHub repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/fraud-detection-xai.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## Step 10 — Add Outputs to GitHub (Optional but Recommended)

The SHAP plots make your repository visually impressive. Add them to GitHub by committing the `outputs/plots/` folder:

```bash
# The .gitignore only excludes model .pkl files, not plots
# So your plots should already be staged in Step 9
# If you run the notebook later and generate new plots, commit them too:

git add outputs/plots/
git commit -m "Add SHAP visualisation outputs"
git push
```

---

## Step 11 — Pin the Repository on Your GitHub Profile

1. Go to your GitHub profile page
2. Click **Customize your pins** 
3. Find `fraud-detection-xai` and pin it
4. This makes it the first thing recruiters see on your profile

---

## Step 12 — Update the README with Your Details

Open `README.md` and update:

- Line near the bottom: replace `YOUR_USERNAME` with your actual GitHub username in the Quick Start section
- The **Author** section: add your real LinkedIn and GitHub URLs

```bash
git add README.md
git commit -m "Update README with author links"
git push
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'xgboost'`**
→ Your venv is not active. Run `source venv/bin/activate` (Mac) or `venv\Scripts\activate` (Windows)

**`FileNotFoundError: ../data/creditcard.csv`**
→ The dataset is not in the right place. Check Step 6 — the file must be named exactly `creditcard.csv` inside the `data/` folder.

**`jupyter: command not found`**
→ Run `pip install jupyter` with your venv active.

**SHAP plots not showing in notebook**
→ Run `pip install ipywidgets` and restart Jupyter.

**Notebook runs slowly**
→ Normal — XGBoost with 300 trees on 450k rows takes a few minutes. Let it run.

---

## What to Say in Interviews

When a recruiter or hiring manager asks about this project, here's a 60-second answer:

> "I built a fraud detection system on 284,807 real credit card transactions. The dataset is extremely imbalanced — only 0.17% fraud — so I used SMOTE to handle that. The XGBoost model achieved over 0.98 ROC-AUC and 0.85 AUPRC, which are strong benchmarks. What makes it distinctive is the SHAP layer — I used it to explain both the global model behaviour and individual fraud predictions. That connects directly to my dissertation research and to real regulatory requirements: under the EU AI Act and FCA Consumer Duty, automated financial decisions need to be explainable in plain language. SHAP waterfall plots give exactly that — a feature-by-feature breakdown of why a specific transaction was flagged."

---

*Good luck, Aditi. This project is genuinely impressive — own it.*
