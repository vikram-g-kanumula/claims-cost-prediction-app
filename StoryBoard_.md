
# Storyboard: Claims Cost Prediction Analysis

## 1. Executive Summary & Problem Statement
**Objective:** To develop a predictive modeling framework for estimating the `UltimateIncurredClaimCost` of workers' compensation claims.
**Context:** Accurate early-stage cost estimation is critical for reserving, risk management, and financial planning. Traditional actuarial methods often lag in incorporating unstructured data and complex non-linear interactions.
**Solution:** This project implements an **Advanced Stacking Ensemble Model** that integrates Natural Language Processing (NLP) of claim descriptions with domain-specific feature engineering.

## 2. Methodology: Exploratory Data Analysis (EDA)
A comprehensive analysis was conducted to identify key drivers of cost escalation.

### 2.1. Data Quality & Preprocessing
*   **Data Integrity Check:** Identified and corrected a semantic error in column naming (`InitialIncurredClaimsCost`).
*   **Distribution Analysis:** The target variable (`UltimateIncurredClaimCost`) exhibited significant right-skewness. A logarithmic transformation ($\log(1+x)$) was applied to normalize the distribution for regression stability.

### 2.2. Key Findings
*   **Temporal Correlation:** A positive correlation was observed between `ReportLag` (days between accident and reporting) and ultimate cost, supporting the hypothesis that delayed reporting exacerbates claim severity.
*   **Textual Signals:** Analysis of `ClaimDescription` via word frequency and latent semantic analysis indicated that specific injury types (e.g., "Strain", "Lower Back") are associated with systematic variances in cost.

## 3. Feature Engineering Strategy
To enhance predictive power, the following domain-aware features were engineered:

*   **Dimensionality Reduction (NLP):**
    *   **Technique:** Term Frequency-Inverse Document Frequency (TF-IDF) followed by Truncated Singular Value Decomposition (SVD).
    *   **Rationale:** Captured latent semantic structures from unstructured text in 30 orthogonal components, enabling the model to "read" accident details.
*   **Interaction Effects:**
    *   **Feature:** `Age * WeeklyWages`.
    *   **Rationale:** Modeled the compounding financial risk associated with older workers who possess higher replacement wage requirements.
*   **Linearization:**
    *   **Feature:** `LogInitialCost`.
    *   **Rationale:** Linearized the strongest predictor to facilitate gradient descent efficiency in base learners.

## 4. Model Architecture & Performance
A multi-stage stacking ensemble was selected to maximize generalization.

### 4.1. Architecture
*   **Level 1 (Base Learners):**
    *   **XGBoost:** Selected for its gradient boosting framework capable of handling non-linear interactions and missing data.
    *   **LightGBM:** Utilized for its leaf-wise tree growth, providing superior speed and accuracy on larger datasets.
*   **Level 2 (Meta-Learner):**
    *   **Ridge Regression (CV):** Employed to aggregate base predictions while regularizing coefficients to prevent overfitting.

### 4.2. Validation Results
The model was evaluated using 5-Fold Cross-Validation:
*   **Log-RMSE:** **0.68** (Indicates strong predictive capability on the normalized scale).
*   **Raw RMSE:** **~$25,500** (Reflecting the high variance inherent in insurance loss distributions).

## 5. Strategic Implications
The deployment of this model via the specific Streamlit interface offers three core business values:

1.  **Operational Efficiency:** Automated scoring allows claims adjusters to prioritize complex files immediately upon intake.
2.  **Risk Transparency:** Integration of SHAP (SHapley Additive exPlanations) values provides explainability, allowing stakeholders to understand *why* a claim is high-risk (e.g., identifying if "Report Lag" or "Initial Cost" is the driver).
3.  **Financial Accuracy:** improved consistency in case reserving reduces the volatility of the actuarial balance sheet.

## 6. Conclusion
The integration of unstructured text data with traditional structured attributes has resulted in a robust predictive engine. This approach validates the efficacy of machine learning in modernizing actuarial workflows.
