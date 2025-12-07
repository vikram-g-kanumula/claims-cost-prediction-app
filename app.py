import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Configuration
st.set_page_config(page_title="Claims Cost Prediction | InsureTech AI", layout="wide", page_icon="🛡️")

# --- Custom CSS for Professional UI ---
st.markdown("""
<style>
    /* Global Settings */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f0f4f8; /* Light Blue-Grey */
        border-right: 1px solid #d1d5db;
    }
    
    /* Main Content Styling */
    .stApp {
        background-color: #ffffff; /* Clean White */
    }
    
    /* Header Styling */
    h1, h2, h3 {
        color: #0d2c54;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        padding-top: 1rem;
    }
    
    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    div[data-testid="stMetric"] > div {
        justify-content: center;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.3rem !important; /* Smaller font to prevent truncation */
        color: #0d2c54;
        font-weight: 700;
    }
    
    /* Info/Alert Styling */
    .stAlert {
        border-left: 5px solid #ffba08; /* Brand Accent */
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #004e92;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #003366;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px;
        color: #0d2c54;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #004e92;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


MODEL_DIR = 'models'
PLOTS_DIR = 'plots'
OUTPUT_DIR = 'output'

# Load Artifacts (Cached)
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, 'stacking_model.pkl'))
    tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf.pkl'))
    svd = joblib.load(os.path.join(MODEL_DIR, 'svd.pkl'))
    encoders = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))
    return model, tfidf, svd, encoders

try:
    model, tfidf, svd, encoders = load_artifacts()
    ARTIFACTS_LOADED = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    ARTIFACTS_LOADED = False

# Sidebar Navigation
st.sidebar.markdown("## 🔍 Navigation")
page = st.sidebar.radio("Navigation", ["Home & Upload", "Training Insights", "Batch Prediction", "Single Prediction"], label_visibility="collapsed")
st.sidebar.markdown("---")

# --- Helper Functions for Preprocessing ---
def preprocess_input(df, tfidf, svd, encoders, feature_names=None):
    # Dates
    date_cols = ['DateTimeOfAccident', 'DateReported']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    if 'DateReported' in df.columns and 'DateTimeOfAccident' in df.columns:
        df['ReportLag'] = (df['DateReported'] - df['DateTimeOfAccident']).dt.days
        df['AccidentYear'] = df['DateTimeOfAccident'].dt.year
        df['AccidentMonth'] = df['DateTimeOfAccident'].dt.month
        df['AccidentDayOfWeek'] = df['DateTimeOfAccident'].dt.dayofweek
        df = df.drop(columns=date_cols)
    
    # New Stats Features (Order Critical: Before NLP in training!)
    if 'InitialIncurredCalimsCost' in df.columns:
        df['LogInitialCost'] = np.log1p(df['InitialIncurredCalimsCost'])
        
    if 'Age' in df.columns and 'WeeklyWages' in df.columns:
        df['Age_Wage_Interaction'] = df['Age'] * df['WeeklyWages']

    # NLP
    if 'ClaimDescription' in df.columns:
        tfidf_matrix = tfidf.transform(df['ClaimDescription'])
        svd_features = svd.transform(tfidf_matrix)
        svd_cols = [f'svd_{i}' for i in range(30)]
        svd_df = pd.DataFrame(svd_features, columns=svd_cols, index=df.index)
        df = pd.concat([df.reset_index(drop=True), svd_df.reset_index(drop=True)], axis=1)
        df = df.drop(columns=['ClaimDescription'])
    
    # Categoricals
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(lambda s: s if s in le.classes_ else 'Unknown')
            known_mask = df[col].isin(le.classes_)
            if not known_mask.all():
                 df.loc[~known_mask, col] = le.classes_[0] 
            df[col] = le.transform(df[col])
            
    # Missing Values
    df = df.fillna(-999)
    
    # Align Columns (drop ID and Target if present)
    cols_to_drop = ['ClaimNumber', 'UltimateIncurredClaimCost']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Enforce Feature Order
    if feature_names is not None:
        missing_cols = [c for c in feature_names if c not in df.columns]
        if missing_cols:
            for c in missing_cols:
                df[c] = -999 
        df = df[feature_names]
        
    return df

# --- Pages ---

if page == "Home & Upload":
    st.title("🛡️ Claims Cost Prediction")
    st.markdown("""
    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h4 style='color: #0d2c54; margin-top: 0;'>🚀 Enterprise Analytics Portal</h4>
        <p>Welcome to the advanced claims forecasting engine. Upload your claims data to instantly generate cost predictions and analyze risk drivers.</p>
        <p><b>Capabilities:</b> Automated Batches • Real-time Scoring • Actuarial Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("1. Data Upload")
    uploaded_file = st.file_uploader("Upload CSV (Schema must match training data)", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state['uploaded_df'] = df
        st.success(f"Successfully uploaded {len(df)} records.")
        
        st.subheader("2. Data Visualization (Interactive)")
        
        # Interactive Plotly Charts
        viz_type = st.selectbox("Choose Visualization", ["Distribution (Histogram)", "Scatter Relationship"])
        
        if viz_type == "Distribution (Histogram)":
            col = st.selectbox("Select Column", df.select_dtypes(include=np.number).columns)
            fig = px.histogram(df, x=col, title=f"Distribution of {col}", nbins=50, template="plotly_white", color_discrete_sequence=['#004e92'])
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Scatter Relationship":
            x_col = st.selectbox("X Axis", df.select_dtypes(include=np.number).columns, index=0)
            y_col = st.selectbox("Y Axis (Target)", df.select_dtypes(include=np.number).columns, index=min(1, len(df.select_dtypes(include=np.number).columns)-1))
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}", template="plotly_white", color_discrete_sequence=['#004e92'])
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("### 📄 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
    
    elif 'uploaded_df' in st.session_state:
        st.info("Using previously uploaded data.")
        st.dataframe(st.session_state['uploaded_df'].head(), use_container_width=True)

elif page == "Training Insights":
    st.title("📊 Training Intelligence & Insights")
    
    # Executive Summary Card
    st.markdown("### 📋 Executive Summary")
    with st.container():
        st.info("""
        **Performance**: **$25,534 RMSE** (Validation).  
        **Actuarial Insight**: `Initial Incurred Cost`, `Weekly Wages` are dominant drivers.  
        **Recommendation**: Suitable for reserves & triage.
        """)
    
    # KPIs Row
    st.markdown("### 📈 Key Performance Indicators (KPIs)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Validation RMSE", "$25,534", help="Lower is better")
    kpi2.metric("R-Squared", "0.68", help="Variance explained")
    kpi3.metric("Key Feature", "Initial Cost", help="Top Predictor")
    kpi4.metric("Algorithm", "Stack Ensemble", help="XGB+LGBM+Ridge")
    
    st.divider()
    
    # 1. Model Explainability Section
    st.subheader("1. Model Explainability & Drivers")
    st.caption("Attributes driving claim cost predictions.")
    
    exp_tab1, exp_tab2 = st.tabs(["Feature Importance", "SHAP Values"])
    
    with exp_tab1:
        try:
            xgb_model = model.estimators_[0]
            importances = xgb_model.feature_importances_
            
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
                fi_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False).head(15)
                
                # FIXED COLOR: Solid color for visibility
                fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                               title="Top 15 Features (XGBoost Component)",
                               color_discrete_sequence=['#004e92']) # Solid Corporate Blue
                
                fig_fi.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white", height=500)
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.warning("Feature names not found in model artifact.")
                st.bar_chart(importances)
        except Exception as e:
            st.error(f"Could not extract feature importance: {str(e)}")

    with exp_tab2:
        st.markdown("**SHAP (Impact Analysis)**")
        if os.path.exists(os.path.join(PLOTS_DIR, 'shap_summary.png')):
            st.image(os.path.join(PLOTS_DIR, 'shap_summary.png'), caption="Global SHAP Summary", use_container_width=True)
        else:
            st.warning("SHAP plot not found. Run training logic to generate it.")

    st.markdown("---")
    st.subheader("2. Target Variable Analysis")
    st.caption("Understanding the risk profile via claim cost distribution.")
    
    # Load Training Data (Cache this!)
    @st.cache_data
    def load_train_data():
        if os.path.exists('data/train.csv'):
            return pd.read_csv('data/train.csv')
        return None

    train_df = load_train_data()
    
    if train_df is not None:
        dist_tab1, dist_tab2 = st.tabs(["Raw Cost Distribution", "Log-Transformed Profile"])
        
        with dist_tab1:
            fig_dist = px.histogram(train_df, x='UltimateIncurredClaimCost', nbins=100, 
                                   title="Distribution of Ultimate Cost (Right-Skewed)",
                                   color_discrete_sequence=['#1f77b4'], template="plotly_white")
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with dist_tab2:
            train_df['LogCost'] = np.log1p(train_df['UltimateIncurredClaimCost'])
            fig_log = px.histogram(train_df, x='LogCost', nbins=100, 
                                  title="Log-Distribution (Normal Approximation)",
                                  color_discrete_sequence=['#2ca02c'], template="plotly_white")
            st.plotly_chart(fig_log, use_container_width=True)
    else:
        st.warning("Training data 'data/train.csv' not found for visualization.")


elif page == "Batch Prediction":
    st.title("🚀 Batch Prediction")
    
    if 'uploaded_df' not in st.session_state:
        st.warning("⚠️ Please upload a dataset on the **Home** page first.")
    else:
        df = st.session_state['uploaded_df'].copy()
        st.write(f"Ready to predict on {len(df)} records.")
        
        if st.button("Generate Predictions"):
            with st.spinner("Running Stacking Ensemble Inference..."):
                try:
                    X_pred = preprocess_input(df.copy(), tfidf, svd, encoders, feature_names=model.feature_names_in_)
                    preds_log = model.predict(X_pred)
                    preds = np.expm1(preds_log)
                    
                    df['Predicted_UltimateIncurredClaimCost'] = preds
                    
                    st.success("✅ Predictions Generated!")
                    st.dataframe(df[['ClaimNumber', 'Predicted_UltimateIncurredClaimCost']].head(10))
                    
                    # Download
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions (CSV)",
                        data=csv,
                        file_name="predictions_stacking.csv",
                        mime="text/csv"
                    )
                    
                    # Interactive Prediction Dist
                    st.subheader("Prediction Distribution")
                    fig = px.histogram(df, x='Predicted_UltimateIncurredClaimCost', 
                                     title="Distribution of Predicted Costs", 
                                     nbins=50, template="plotly_white",
                                     color_discrete_sequence=['#00CC96'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

elif page == "Single Prediction":
    st.title("👤 Single Record Prediction")
    st.markdown("Enter claim details to get an instant cost estimate.")
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            claim_desc = st.text_area("Claim Description", "LIFTING BOX INJURY TO LOWER BACK")
            acc_date = st.date_input("Date of Accident")
            rep_date = st.date_input("Date Reported")
            age = st.number_input("Age", 18, 100, 30)
            gender = st.selectbox("Gender", ["M", "F", "U"])
        
        with col2:
            marital = st.selectbox("MaritalStatus", ["M", "S", "U", "D", "W"])
            wages = st.number_input("Weekly Wages", 0.0, 5000.0, 500.0)
            initial_cost = st.number_input("Initial Incurred Cost", 0.0, 100000.0, 1000.0)
            part_time = st.selectbox("PartTimeFullTime", ["F", "P"])
        
        submitted = st.form_submit_button("Predict Cost")
        
        if submitted:
            # Create DF
            record = {
                'ClaimNumber': 'DUMMY',
                'ClaimDescription': claim_desc,
                'DateTimeOfAccident': pd.to_datetime(acc_date),
                'DateReported': pd.to_datetime(rep_date),
                'Age': age,
                'Gender': gender,
                'MaritalStatus': marital,
                'WeeklyWages': wages,
                'InitialIncurredCalimsCost': initial_cost,
                'PartTimeFullTime': part_time,
                # Defaults for unexposed fields
                'DependentChildren': 0, 
                'DependentsOther': 0, 
                'HoursWorkedPerWeek': 40.0, 
                'DaysWorkedPerWeek': 5
            }
            
            input_df = pd.DataFrame([record])
            
            try:
                X_single = preprocess_input(input_df, tfidf, svd, encoders, feature_names=model.feature_names_in_)
                pred_log = model.predict(X_single)
                pred = np.expm1(pred_log)[0]
                
                st.metric("Predicted Estimate", f"${pred:,.2f}")
                st.info(f"Log-Prediction: {pred_log[0]:.4f}")
            except Exception as e:
                st.error(f"Error: {e}")
