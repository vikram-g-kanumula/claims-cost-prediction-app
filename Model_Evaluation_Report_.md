
# Model Evaluation Report

## 1. Executive Summary
This report details the evaluation and performance metrics of the **Advanced Stacking Ensemble Model** designed to predict the `UltimateIncurredClaimCost`. The model successfully integrates unstructured text analytics (NLP), domain-specific feature engineering, and robust ensemble learning techniques.

## 2. Model Architecture
The final model employs a multi-tier stacking architecture to minimize variance and bias:

*   **Base Learners (Level 1)**:
    *   **XGBoost Regressor**: Selected for its gradient boosting framework, capable of modeling complex non-linear interactions.
    *   **LightGBM Regressor**: Utilized for its efficiency with large-scale data and leaf-wise tree growth strategy.
*   **Meta-Learner (Level 2)**:
    *   **Ridge Regression (CV)**: Aggregates base learner predictions with L2 regularization to prevent overfitting and ensure stability.

## 3. Performance Metrics
Performance was evaluated using 5-Fold Cross-Validation on a training set of 54,000 records, followed by validation on a hold-out set.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Validation RMSE (Raw)** | **$25,578** | On average, predictions deviate by ~$25.5k from the actual ultimate cost, reflecting the high variance in the tail-heavy loss distribution. |
| **Log-Scale R²** | **0.9232** | The model explains **92.3%** of the variance in the log-transformed target, indicating a very strong fit for the underlying distribution. |
| **Raw-Scale R²** | **0.4692** | On the original dollar scale, the model captures **46.9%** of the variance, a competitive result given the inherent volatility of insurance claims. |

> **Analyst Note**: The divergence between Log-R² and Raw-R² is expected in actuarial modeling. The model is highly effective at ranking risk (Log-Scale) but faces natural uncertainty in pinpointing the exact dollar value of catastrophic outliers.

## 4. Key Drivers & Feature Importance
Analysis using SHAP (SHapley Additive exPlanations) confirms the following primary risk drivers:

1.  **Initial Incurred Cost**: The strongest single predictor. Higher initial case estimates are strongly correlated with higher ultimate payouts.
2.  **Weekly Wages**: A critical demographic factor; higher wages statistically correlate with higher indemnity costs.
3.  **Claim Description (NLP)**: Latent semantic components (SVD vectors) derived from accident descriptions provided significant lift, identifying high-risk injury types (e.g., "Strain", "Lower Back") encoded in text.
4.  **Report Lag**: Delayed reporting is a verified indicator of cost escalation, likely due to litigation risks or delayed medical intervention.

## 5. Deployment Recommendations
*   **Reserving Strategy**: Use model outputs to baseline initial case reserves, reducing manual estimation variance.
*   **Triage Protocol**: Automatically flag claims in the top decile of predicted cost for immediate review by senior adjusters.
*   **Continuous Monitoring**: Retrain the model quarterly to incorporate new closed claims and adjust for medical inflation trends.

## 6. Conclusion
The Stacking Ensemble meets the business requirements for accuracy and interpretability. By effectively leveraging unstructured data, it provides a measurable competitive advantage over traditional, structured-data-only actuarial models.
