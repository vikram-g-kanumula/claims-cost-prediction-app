
# Storyboard: Claims Cost Prediction Analysis

## 1. Executive Summary & Problem Statement
*   **Objective**: To develop a predictive modeling framework for estimating the `UltimateIncurredClaimCost` of workers' compensation claims.
*   **Context**: Accurate early-stage cost estimation is critical for reserving, risk management, and financial planning. Traditional actuarial methods often lag in incorporating unstructured data and complex non-linear interactions.
*   **Solution**: This project implements an **Advanced Stacking Ensemble Model** that integrates Natural Language Processing (NLP) of claim descriptions with domain-specific feature engineering.

## 2. Methodology: Exploratory Data Analysis (EDA)
A comprehensive analysis was conducted on the training dataset of **54,000 records** to identify key drivers of cost escalation.

### 2.1. Feature Distribution & Linearity
*   **Target Skewness**: The `UltimateIncurredClaimCost` displayed significant right-skewness (Mean: ~$11,003). A logarithmic transformation ($\log(1+x)$) was applied to normalize the distribution, stabilizing the variance for regression modeling.
*   **Primary Predictor**: `InitialIncurredClaimsCost` demonstrated the strongest single-variable correlation ($r \approx 0.40$) with the ultimate cost, validating its use as a baseline regressor.

### 2.2. Complex Interactions
*   **Temporal Dynamics**: The average reporting lag was found to be **38.32 days**. While the linear correlation with cost was weak ($r \approx 0.03$), tree-based models (XGBoost) effectively utilized this feature to split high-risk tail events.
*   **Demographic Interaction**: The interaction term `Age * WeeklyWages` showed a meaningful correlation ($r \approx 0.17$), outperforming age or wage alone. This suggests that the **economic replacement value** of a claimant (older, higher-wage workers) is a robust predictor of claim severity.

## 3. Feature Engineering Strategy
Systematic engineering was applied to bridge the gap between raw data and predictive signals.

*   **Dimensionality Reduction (NLP)**
    *   **Technique**: TF-IDF Vectorization $\rightarrow$ Truncated SVD (30 Components).
    *   **Rationale**: "Claim Description" text holds latent semantic value regarding injury severity. Decomposing this into 30 orthogonal components allowed the model to ingress unstructured text as dense numerical features.
*   **Mathematical Linearization**
    *   **Feature**: `LogInitialCost`.
    *   **Rationale**: Linearizing the initial cost provided a consistent gradient for the base learners, enhancing convergence speed.

## 4. Model Architecture & Performance
A multi-stage stacking ensemble was selected to maximize generalization.

### 4.1. Architecture
*   **Level 1 (Base Learners)**
    *   **XGBoost & LightGBM**: Selected for their ability to handle non-linearities and potential missing values natively.
*   **Level 2 (Meta-Learner)**
    *   **Ridge Regression (CV)**: Aggregated base predictions with L2 regularization to prevent overfitting.

### 4.2. Validation Results
The model was evaluated using 5-Fold Cross-Validation:
*   **Log-RMSE**: **0.5636** (Stable performance on the log scale).
*   **Raw RMSE**: **~$25,578** (Reflecting the inherent variance in high-cost claims).

## 5. Strategic Implications
The deployment of this model via the Streamlit interface offers tangible business value:

1.  **Prioritization**: Automated scoring allows claims adjusters to triage high-value claims immediately.
2.  **Explainability**: Integration of SHAP values isolates the contribution of specific features (e.g., whether a high prediction is driven by the *Initial Cost* or the *Injury Description*), enhancing trust in automated decisions.
3.  **Efficiency**: Reducing the time-to-estimate for complex claims supports more accurate financial reserving.

## 6. Conclusion
The integration of unstructured text data with traditional structured attributes—specifically the interaction of age and wage—has resulted in a robust predictive engine. This approach validates the efficacy of machine learning in modernizing actuarial workflows.
