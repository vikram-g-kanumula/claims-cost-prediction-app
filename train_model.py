import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
INPUT_DIR = 'data'
OUTPUT_DIR = 'output'
PLOTS_DIR = 'plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TARGET_COL = 'UltimateIncurredClaimCost'
ID_COL = 'ClaimNumber'

def load_data():
    train = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
    return train, test

def preprocess_dates(df):
    date_cols = ['DateTimeOfAccident', 'DateReported']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    
    # Feature Engineering: Lag
    df['ReportLag'] = (df['DateReported'] - df['DateTimeOfAccident']).dt.days
    
    # Cyclical/Seasonal features
    df['AccidentYear'] = df['DateTimeOfAccident'].dt.year
    df['AccidentMonth'] = df['DateTimeOfAccident'].dt.month
    df['AccidentDayOfWeek'] = df['DateTimeOfAccident'].dt.dayofweek
    df['AccidentHour'] = df['DateTimeOfAccident'].dt.hour
    
    # Drop original dates
    df = df.drop(columns=date_cols)
    return df

def preprocess_text(train, test):
    # Combining for consistent vocabulary
    all_text = pd.concat([train['ClaimDescription'], test['ClaimDescription']], axis=0)
    
    # TF-IDF
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = tfidf.fit_transform(all_text)
    
    # Dimensionality Reduction (SVD) to keep features dense and manageable
    svd = TruncatedSVD(n_components=50, random_state=42)
    svd_features = svd.fit_transform(tfidf_matrix)
    
    # Convert to DataFrame
    svd_cols = [f'svd_{i}' for i in range(50)]
    svd_df = pd.DataFrame(svd_features, columns=svd_cols, index=all_text.index)
    
    # Split back
    train_svd = svd_df.iloc[:len(train)]
    test_svd = svd_df.iloc[len(train):]
    
    # Concatenate to original dfs
    train = pd.concat([train.reset_index(drop=True), train_svd.reset_index(drop=True)], axis=1)
    test = pd.concat([test.reset_index(drop=True), test_svd.reset_index(drop=True)], axis=1)
    
    # Drop original text
    train = train.drop(columns=['ClaimDescription'])
    test = test.drop(columns=['ClaimDescription'])
    
    return train, test

def preprocess_categoricals(train, test):
    # Identify categoricals
    cat_cols = train.select_dtypes(include=['object']).columns.drop([ID_COL], errors='ignore')
    
    # Label Encoding for Tree Models (often better than OneHot for high cardinality, though we have low here mostly)
    # But let's check unique counts
    print("Categorical Columns:", cat_cols)
    
    for col in cat_cols:
        le = LabelEncoder()
        # Handle new categories in test by fitting on both
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        
    return train, test

def main():
    print("Loading Data...")
    train, test = load_data()
    
    print("Preprocessing Dates...")
    train = preprocess_dates(train)
    test = preprocess_dates(test)
    
    print("Preprocessing Text (NLP)...")
    train, test = preprocess_text(train, test)
    
    print("Preprocessing Categoricals...")
    train, test = preprocess_categoricals(train, test)
    
    # Handle Missing Values (Simple imputation for now, Tree models handle nan usually but being safe)
    train = train.fillna(-999)
    test = test.fillna(-999)
    
    # Target Transformation (Log)
    y = np.log1p(train[TARGET_COL])
    X = train.drop(columns=[TARGET_COL, ID_COL])
    X_test = test.drop(columns=[ID_COL])
    
    print(f"Training Features: {X.shape[1]}")
    
    # XGBoost Model
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )
    
    print("Cross Validation...")
    # 5-Fold CV
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    
    # Convert log-error back to readable scale approximatley? 
    # Actually, RMSE on log scale isn't directly interpretable as dollar error. 
    # But it gives us stability check.
    print(f"CV Log-RMSE: {-scores.mean():.4f} (+/- {scores.std():.4f})")
    
    print("Training Final Model...")
    model.fit(X, y)
    
    # Interpretation with SHAP
    print("Generating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'shap_summary.png'))
    plt.close()
    
    # Feature Importance (Default Gain)
    xgb.plot_importance(model, max_num_features=20, importance_type='gain')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'))
    plt.close()
    
    # Predictions
    print("Generating Predictions...")
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log) # Inverse log transform
    
    submission = pd.DataFrame({
        ID_COL: test[ID_COL],
        TARGET_COL: preds
    })
    
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

if __name__ == "__main__":
    main()
