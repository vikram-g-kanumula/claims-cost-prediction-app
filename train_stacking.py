import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import RidgeCV, HuberRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
import joblib
import os

# Configuration
INPUT_DIR = 'data'
OUTPUT_DIR = 'output'
MODEL_DIR = 'models'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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
    
    df['ReportLag'] = (df['DateReported'] - df['DateTimeOfAccident']).dt.days
    df['AccidentYear'] = df['DateTimeOfAccident'].dt.year
    df['AccidentMonth'] = df['DateTimeOfAccident'].dt.month
    df['AccidentDayOfWeek'] = df['DateTimeOfAccident'].dt.dayofweek
    df = df.drop(columns=date_cols)
    
    # ---------------- NEW FE ----------------
    # Log transform of Initial Cost (handling 0s)
    df['LogInitialCost'] = np.log1p(df['InitialIncurredCalimsCost'])
    
    # Interaction
    df['Age_Wage_Interaction'] = df['Age'] * df['WeeklyWages']
    # ----------------------------------------
    return df

def preprocess_text(train, test):
    all_text = pd.concat([train['ClaimDescription'], test['ClaimDescription']], axis=0)
    
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = tfidf.fit_transform(all_text)
    
    svd = TruncatedSVD(n_components=30, random_state=42)
    svd_features = svd.fit_transform(tfidf_matrix)
    
    # Save NLP artifacts
    joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf.pkl'))
    joblib.dump(svd, os.path.join(MODEL_DIR, 'svd.pkl'))
    
    svd_cols = [f'svd_{i}' for i in range(30)]
    svd_df = pd.DataFrame(svd_features, columns=svd_cols, index=all_text.index)
    
    train_svd = svd_df.iloc[:len(train)]
    test_svd = svd_df.iloc[len(train):]
    
    train = pd.concat([train.reset_index(drop=True), train_svd.reset_index(drop=True)], axis=1)
    test = pd.concat([test.reset_index(drop=True), test_svd.reset_index(drop=True)], axis=1)
    
    train = train.drop(columns=['ClaimDescription'])
    test = test.drop(columns=['ClaimDescription'])
    return train, test

def preprocess_categoricals(train, test):
    cat_cols = train.select_dtypes(include=['object']).columns.drop([ID_COL], errors='ignore')
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        encoders[col] = le
        
    joblib.dump(encoders, os.path.join(MODEL_DIR, 'encoders.pkl'))
    return train, test

def main():
    print("Loading and Preprocessing Data for Stacking...")
    train, test = load_data()
    train = preprocess_dates(train)
    test = preprocess_dates(test)
    train, test = preprocess_text(train, test)
    train, test = preprocess_categoricals(train, test)
    
    train = train.fillna(-999)
    test = test.fillna(-999)
    
    y = np.log1p(train[TARGET_COL])
    X = train.drop(columns=[TARGET_COL, ID_COL])
    X_test = test.drop(columns=[ID_COL])
    
    print("Initializing Models (Best Ensemble with Domain Features)...")
    
    # 1. XGBoost
    xgb_sq = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=6, 
        objective='reg:squarederror', n_jobs=-1, random_state=42
    )
    
    # 2. LightGBM
    lgb_sq = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        objective='regression', n_jobs=-1, random_state=42, verbose=-1
    )

    estimators = [
        ('xgb_sq', xgb_sq),
        ('lgb_sq', lgb_sq)
    ]
    
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        n_jobs=-1,
        cv=5
    )
    
    print("Evaluating Stacking Ensemble (Hold-out)...")
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Validation Fit
    stacking_regressor.fit(X_train, y_train)
    val_preds = stacking_regressor.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Stacking Validation Log-RMSE: {val_rmse:.4f}")

    # Calculate Raw RMSE
    val_rmse_raw = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_preds)))
    print(f"Stacking Validation Raw RMSE: {val_rmse_raw:.2f}")
    
    print("Training Final Stacking Model (Full Data)...")
    stacking_regressor.fit(X, y)
    
    joblib.dump(stacking_regressor, os.path.join(MODEL_DIR, 'stacking_model.pkl'))
    print("Model saved successfully.")
    
    print("Generating Stacking Predictions...")
    preds_log = stacking_regressor.predict(X_test)
    preds = np.expm1(preds_log)
    
    submission_path = os.path.join(OUTPUT_DIR, 'submission_stacking.csv')
    submission = pd.DataFrame({
        ID_COL: test[ID_COL],
        TARGET_COL: preds
    })
    submission.to_csv(submission_path, index=False)
    print(f"Stacked Submission saved to {submission_path}")

if __name__ == "__main__":
    main()
