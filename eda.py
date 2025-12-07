import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for premium aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

DATA_DIR = 'data'
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    train_path = os.path.join(DATA_DIR, 'train.csv')
    test_path = os.path.join(DATA_DIR, 'test.csv')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Train Value Counts: {len(train)}")
    print(f"Test Value Counts: {len(test)}")
    return train, test

def univariate_analysis(df, target_col='UltimateIncurredClaimCost'):
    print("\n--- Univariate Analysis ---")
    
    # Target Variable Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_col], kde=True, color='#4A90E2')
    plt.title(f'Distribution of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(PLOTS_DIR, 'target_distribution.png'))
    plt.close()
    
    print(f"{target_col} Skewness: {df[target_col].skew()}")
    print(f"{target_col} Kurtosis: {df[target_col].kurt()}")

    # Numerical Features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.drop([target_col])
    for col in numerical_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), kde=True, color='#50E3C2')
        plt.title(f'Distribution of {col}')
        plt.savefig(os.path.join(PLOTS_DIR, f'dist_{col}.png'))
        plt.close()
        
        # Boxplot for outliers
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[col].dropna(), color='#F5A623')
        plt.title(f'Boxplot of {col}')
        plt.savefig(os.path.join(PLOTS_DIR, f'boxplot_{col}.png'))
        plt.close()

def bivariate_analysis(df, target_col='UltimateIncurredClaimCost'):
    print("\n--- Bivariate Analysis ---")
    
    # Correlation Matrix
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numerical_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_matrix.png'))
    plt.close()
    
    # Scatter plots for high correlation features
    # (Simple logic: just plot all numericals vs target for now)
    feature_cols = numerical_cols.drop([target_col])
    for col in feature_cols:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[col], y=df[target_col], alpha=0.6, color='#9013FE')
        plt.title(f'{col} vs {target_col}')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'scatter_{col}_vs_target.png'))
        plt.close()

    # Categorical Analysis vs Target
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    # Filter out high cardinality cols like ClaimNumber, ClaimDescription
    low_card_cats = [c for c in categorical_cols if df[c].nunique() < 20]
    
    for col in low_card_cats:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=col, y=target_col, data=df, palette='Set2')
        plt.title(f'{col} vs {target_col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'boxplot_{col}_vs_target.png'))
        plt.close()

def outlier_analysis(df, target_col='UltimateIncurredClaimCost'):
    print("\n--- Outlier Analysis ---")
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]
    print(f"Number of outliers in {target_col}: {len(outliers)}")
    print(f"Percentage of outliers: {len(outliers)/len(df)*100:.2f}%")
    print(f"Outlier boundaries: {lower_bound:.2f}, {upper_bound:.2f}")

if __name__ == "__main__":
    train_df, test_df = load_data()
    
    # Basic info
    print(train_df.info())
    print("\nMissing Values:")
    print(train_df.isnull().sum())

    univariate_analysis(train_df)
    bivariate_analysis(train_df)
    outlier_analysis(train_df)
    print(f"\nEDA Completed. Plots saved to {PLOTS_DIR}")
