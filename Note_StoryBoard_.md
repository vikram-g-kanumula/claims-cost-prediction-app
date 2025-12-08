
# storyboard.md

## Narrative: Unlocking Insights from Claims Data

### 1. The Challenge
Workers' compensation claims are complex and long-tail. The goal was to predict the `UltimateIncurredClaimCost` to enable better financial planning and early intervention.

### 2. Exploratory Data Analysis (The Discovery)
Our journey began with raw data.
*   **Data Quality**: We identified and fixed a typo in `InitialIncurredClaimsCost`.
*   **Distributions**: Costs are highly right-skewed. A log-transformation was essential to normalize the target for modeling.
*   **Text Insights**: Analysis of `ClaimDescription` revealed that back injuries and lifting accidents are frequent drivers.
*   **Time Matters**: We discovered a clear relationship between `ReportLag` (delay in reporting) and higher costs.

### 3. Feature Engineering (The Strategy)
To improve predictive power beyond basic attributes, we engineered:
*   **NLP Features**: Decomposed text descriptions into 30 semantic components (SVD) to let the model "read" the accident details.
*   **Interactions**: `Age * WeeklyWages` captured the compounding effect of older workers with higher replacement wages.
*   **Transformation**: `LogInitialCost` linearized the strongest predictor.

### 4. Modeling (The Solution)
We moved from simple regressions to an Advanced Stacking Ensemble.
*   **Approach**: Combined the strengths of XGBoost (gradual learning) and LightGBM (leaf-wise growth).
*   **Validation**: Rigorous 5-fold CV ensured the model generalizes well to unseen future claims.
*   **Result**: A robust model achieving ~0.68 Log-RMSE, significantly outperforming the baseline average.

### 5. Business Impact (The Value)
The deployed Streamlit Application puts this power in the hands of adjusters.
*   **Adjusters** can instantly score new claims.
*   **Managers** can visualize portfolio risk in real-time.
*   **Actuaries** gain transparency into what drives cost (via SHAP explanations).

**Conclusion**: By marrying unstructured text data with traditional structured fields, we've created a next-generation predictive engine for claims management.
