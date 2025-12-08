
# 🛡️ Insurance Claims Cost Prediction

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)

## 📌 Project Overview
This project targets the actuarial challenge of accurately predicting the `UltimateIncurredClaimCost` for workers' compensation claims. By leveraging an **Advanced Stacking Ensemble Model**, the solution integrates structured demographic/policy data with unstructured text analytics (NLP) to identify cost drivers that traditional linear models miss.

The final deliverable is an interactive Streamlit application that empowers claims adjusters with real-time scoring and explainable AI insights.

## 🚀 Key Features
*   **Hybrid Feature Engineering**: Combines financial metrics (`InitialIncurredClaimsCost`) with Latent Semantic Analysis (SVD) of accident descriptions.
*   **Stacking Architecture**: Ensembles **XGBoost** and **LightGBM** (Base Learners) with **Ridge Regression** (Meta-Learner) for superior generalization.
*   **Explainable AI**: Integrated **SHAP (SHapley Additive exPlanations)** values to provide "localized" reasons for every prediction (e.g., "High Cost due to 'Strain' injury + High Wage").
*   **Enterprise Dashboard**: A polished UI for data upload, batch scoring, and single-claim inference.

## 📊 Performance Metrics
The model was rigorously evaluated using 5-Fold Cross-Validation:

| Metric | Performance | Interpretation |
| :--- | :--- | :--- |
| **Validation RMSE** | **$25,578** | Avg. deviation from actual cost (Raw Scale). |
| **Raw R-Squared** | **0.47** | Explains ~47% of variance in dollar costs. |
| **Log R-Squared** | **0.92** | Excellent fit for the underlying log-normal distribution. |

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.8+
*   pip

### Steps
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/vikram-g-kanumula/claims-cost-prediction-app.git
    cd claims-cost-prediction-app
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Training Pipeline (Optional)**
    *   Regenerates models & artifacts in `models/` and `plots/`.
    ```bash
    python train_stacking.py
    ```

4.  **Launch the Application**
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure
```
├── app.py                      # Main Streamlit Application
├── train_stacking.py           # ML Training Pipeline (Data Prep -> Modeling -> Artifacts)
├── update_eda_final.py         # Utility to refresh EDA Notebooks
├── EDA_Analysis.ipynb          # Exploratory Data Analysis
├── requirements.txt            # Python Dependencies
├── data/                       # Input Datasets (train.csv, test.csv)
├── models/                     # Serialized Model Artifacts (pkl)
├── plots/                      # Generated Static Plots (SHAP)
├── StoryBoard_.md              # Strategic Project Narrative
└── Model_Evaluation_Report_.md # Detailed Actuarial Performance Report
```

## 📚 Documentation
For detailed insights, refer to the core project artifacts:
*   [**Strategic Storyboard**](StoryBoard_.md): Exploring the "Why" and business impact.
*   [**Model Evaluation Report**](Model_Evaluation_Report_.md): Technical deep-dive into metrics and architecture.
