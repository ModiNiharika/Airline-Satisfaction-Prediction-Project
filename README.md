# ✈️ Airline Passenger Satisfaction Prediction

Predicting whether an airline passenger is **satisfied** or **neutral/dissatisfied** from flight experience, in-flight service ratings, and travel conditions, using classical machine learning classifiers.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## At a Glance

| | |
|---|---|
| **Task** | Binary classification (satisfied vs. neutral/dissatisfied) |
| **Dataset** | 25,976 airline passenger survey records, 25 raw features |
| **Features used in model** | 23 features (identifiers excluded), 27 after one-hot encoding |
| **Models compared** | Logistic Regression, SVM, KNN, Naive Bayes, Decision Tree, Random Forest |
| **Best model** | Random Forest — **~95% accuracy**, 0.95 weighted F1-score |
| **Top driver of satisfaction** | Online boarding experience (highest correlation with satisfaction, r = 0.49) |

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Skills Demonstrated](#skills-demonstrated)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [Limitations & Future Work](#limitations--future-work)
- [Contributors](#contributors)

---

## Overview

Customer satisfaction is a key competitive factor in the airline industry. This project analyzes a passenger survey dataset to identify what drives satisfaction and builds classification models to predict it from passenger and service-quality data.

**Objectives:**
- Clean and prepare raw airline passenger survey data
- Explore relationships between service factors and satisfaction through statistical testing and visualization
- Train and compare multiple classification algorithms
- Identify the best-performing model and the key drivers of satisfaction

---

## Dataset

- **Records:** 25,976 passenger entries
- **Original features:** 25, including:
  - **Demographics:** `Gender`, `Age`, `Customer Type`
  - **Travel details:** `Type of Travel`, `Class`, `Flight Distance`
  - **Service ratings (0–5 scale):** `Inflight wifi service`, `Online boarding`, `Seat comfort`, `Inflight entertainment`, `On-board service`, `Leg room service`, `Baggage handling`, `Checkin service`, `Inflight service`, `Cleanliness`, `Food and drink`, `Ease of Online booking`, `Gate location`, `Departure/Arrival time convenient`
  - **Delay data:** `Departure Delay in Minutes`, `Arrival Delay in Minutes`
  - **Target:** `Satisfaction` (satisfied / neutral or dissatisfied)

- **Note:** `Record_id` and `id` are identifier columns present in the raw data. They are retained in the preprocessing CSV for traceability but are explicitly excluded from the model's feature set.

> The dataset CSV files are not included in this repository. See [Getting Started](#getting-started) for setup instructions.

---

## Project Workflow

<details>
<summary><strong>1. Data Preprocessing</strong> — <a href="Preprocessing.ipynb">Preprocessing.ipynb</a></summary>
<br>

- Loaded and inspected the dataset; renamed unlabelled columns and corrected row indices
- Called `df.info()` to confirm data types across all 25 columns
- Identified and imputed 83 missing values in `Arrival Delay in Minutes` using **median imputation**
- Converted target column (`Satisfaction`) to binary (1 = satisfied, 0 = neutral/dissatisfied)
- Printed duplicate-row count before and after `drop_duplicates()` for full transparency
- Binned `Age` into five labelled groups: Teen (0–18), Young (18–30), Mid (30–45), Senior (45–60), Old (60+)
- Computed descriptive statistics for all numeric features
- Ran a one-way **ANOVA test** on `Class` vs. `Satisfaction` (F = 4276.58, p < 0.001) — confirming travel class has a statistically significant effect
- Computed feature correlations against the target variable
- Exported cleaned data to `data/data_preprocessing.csv`

</details>

<details>
<summary><strong>2. Exploratory Data Analysis</strong> — <a href="Data_Visualization.ipynb">Data_Visualization.ipynb</a></summary>
<br>

- **Univariate analysis:** satisfaction distribution, age distribution, flight distance distribution, class distribution
- **Bivariate analysis:** class vs. satisfaction, inflight wifi vs. satisfaction, seat comfort & inflight entertainment vs. satisfaction, online boarding/food & drink/cleanliness/inflight services vs. satisfaction
- **Multivariate analysis:** correlation heatmap across all numeric features

</details>

<details>
<summary><strong>3. Modeling</strong> — <a href="ML_Model.ipynb">ML_Model.ipynb</a></summary>
<br>

- Dropped non-predictive identifier columns (`Record_id`, `id`) before the feature/target split
- Applied one-hot encoding (`pd.get_dummies`, `drop_first=True`) to categorical variables
- 80/20 train-test split (`random_state=42`)
- Applied `StandardScaler` fitted on training data only (no leakage); same scaler applied consistently to test data and new-customer inference
- Trained and evaluated six classification algorithms using accuracy, confusion matrix, and classification report
- Tested the best model on a sample new-customer input, passing it through the same scaler transform used during training

</details>

---

## Key Findings

- Business class passengers show notably higher satisfaction than Economy and Economy Plus passengers
- Most passengers fall in the 30–45 age range and take short-distance flights, where service quality and delays have an outsized impact on satisfaction
- Top correlates of satisfaction: **Online boarding** (0.49), **Inflight entertainment** (0.40), **Seat comfort** (0.35)
- No single feature dominates (no correlation ≥ 0.7) — satisfaction is a combination of factors, with **online boarding** standing out as the most consistent differentiator between satisfied and dissatisfied passengers
- ANOVA confirms travel class has a statistically significant effect on satisfaction (F = 4276.58, p < 0.001)

---

## Model Performance

> These results reflect the notebook's last full run. Re-run all three notebooks locally after setting up your `data/` folder to generate fresh outputs.

| Model | Accuracy |
|---|---|
| Naive Bayes | 83.14% |
| Logistic Regression | 86.78% |
| K-Nearest Neighbors (k=5) | 90.55% |
| Decision Tree | 92.76% |
| Support Vector Machine (SVM) | 93.88% |
| **Random Forest** | **~95%** |

**Best model — Random Forest Classifier**, with precision, recall, and F1-score of 0.95 (weighted average) and balanced performance across both satisfied and dissatisfied classes.

The final model was also tested on a sample new-customer input, passed through the same `StandardScaler` used during training to ensure a consistent feature space.

---

## Skills Demonstrated

- Data cleaning: missing-value imputation, deduplication with verification, type correction
- Feature engineering: age binning, one-hot encoding, exclusion of non-predictive identifiers
- Statistical analysis: ANOVA hypothesis testing, correlation analysis
- Exploratory data analysis: univariate, bivariate, and multivariate visualisation with seaborn/matplotlib
- Supervised machine learning: training and evaluating six classification algorithms
- Model evaluation: accuracy, confusion matrix, precision/recall/F1-score, classification reports
- Preprocessing hygiene: train-only scaler fitting, consistent transform applied at inference time
- Technical communication: structured notebook documentation and a formal project report

---

## Tech Stack

| Library | Purpose |
|---|---|
| pandas, numpy | Data loading, cleaning, and manipulation |
| matplotlib, seaborn | Visualisation |
| scipy | ANOVA hypothesis testing |
| scikit-learn | Encoding, scaling, model training, and evaluation |
| Jupyter Notebook | Interactive development environment |

---

## Repository Structure

```
AirlineProject/
├── Preprocessing.ipynb        # Data cleaning, formatting, ANOVA, correlation
├── Data_Visualization.ipynb   # Univariate, bivariate, multivariate EDA
├── ML_Model.ipynb             # Encoding, scaling, model training & evaluation
├── documentation.pdf          # Full project report
├── requirements.txt           # Python dependencies
└── .gitignore
```

> **Data folder (not committed):** Create a `data/` folder at the project root and place `test.csv` inside it before running the notebooks. The two intermediate CSVs (`data_preprocessing.csv`, `data_modeling.csv`) are written there automatically.

---

## Getting Started

1. Clone the repository
   ```bash
   git clone <repo-url>
   cd AirlineProject
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `data/` folder at the project root and place your copy of `test.csv` inside it
4. Run the notebooks in order:
   ```
   Preprocessing.ipynb → Data_Visualization.ipynb → ML_Model.ipynb
   ```

---

## Documentation

A full project report — including background, objectives, methodology flowcharts, and detailed model outputs — is available in [`documentation.pdf`](documentation.pdf).

---

## Limitations & Future Work

- Stored notebook outputs reflect the last full run; re-run all notebooks after setting up the `data/` folder to regenerate verified results
- No hyperparameter tuning or cross-validation was performed — all models use default scikit-learn parameters
- No saved model artifact or standalone inference script is currently included
- Potential next steps: `sklearn.Pipeline` for end-to-end preprocessing, `GridSearchCV` for hyperparameter tuning, SHAP values for feature-importance explainability, and a lightweight deployment demo (e.g. Streamlit)

---

## Contributors

- **Niharika Modi** (S221068)
- **Moulya Lenka** (S221065)

*Guide: Siva Rama Sastry G*
Rajiv Gandhi University of Knowledge Technologies, Srikakulam — B.Tech CSE, 2025–2026
