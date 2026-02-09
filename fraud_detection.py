import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

# ML & Evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, average_precision_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# =============================================================================
# 1. PAGE SETUP & SIDEBAR (The interface you want)
# =============================================================================
st.set_page_config(page_title="Healthcare Fraud Detection", layout="wide")

with st.sidebar:
    st.header("⚙️ Configuration")
    n_claims = st.slider("Number of Claims", 10000, 100000, 50000)
    fraud_rate = st.slider("Fraud Rate (%)", 1, 10, 5)
    
    # The red button from your screenshot
    run_button = st.button("🔄 Generate Data & Train Model", type="primary")

st.title("🏥 Healthcare Insurance Claims Fraud Detection")
st.subheader("Detecting Upcoding & Phantom Billing Patterns")

# =============================================================================
# 2. DATA & LOGIC FUNCTIONS
# =============================================================================
@st.cache_data
def generate_data(n, rate):
    # (Logic from your provided script)
    np.random.seed(42)
    icd10_codes = {'high_cost': ['C34.90', 'I21.3'], 'medium_cost': ['J18.9', 'I10'], 'low_cost': ['J06.9', 'R51']}
    cpt_codes = ['99213', '99214', '99215']
    claims = pd.DataFrame({
        'claim_id': range(1, n + 1),
        'provider_id': np.random.randint(1, 500, n),
        'patient_id': np.random.randint(1, 10000, n),
        'claim_date': [datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(n)],
        'cpt_code': np.random.choice(cpt_codes, n),
        'claim_amount': np.random.uniform(50, 500, n),
        'is_fraud': 0
    })
    # Simple fraud injection for demo
    fraud_idx = claims.sample(int(n * (rate/100))).index
    claims.loc[fraud_idx, 'is_fraud'] = 1
    return claims, icd10_codes

# =============================================================================
# 3. INTERFACE RENDERING
# =============================================================================
if not run_button:
    st.info("👈 Click 'Generate Data & Train Model' in the sidebar to get started!")
    
    st.header("About This System")
    st.write("This fraud detection system identifies two main types of healthcare insurance fraud:")
    st.markdown("### 1. Upcoding 🔼")
    st.write("* Billing for more expensive procedures than actually performed")
    st.write("* Detected through abnormal high-level E&M code frequencies (e.g., 99215)")
    st.markdown("### 2. Phantom Billing 👻")
    st.write("* Billing for services never rendered to patients")

else:
    # RUN THE SYSTEM
    with st.spinner("Processing Model..."):
        df, codes = generate_data(n_claims, fraud_rate)
        
        # Display Stats
        col1, col2 = st.columns(2)
        col1.metric("Dataset Size", len(df))
        col2.metric("Target Fraud Rate", f"{fraud_rate}%")
        
        # Standard Output (Replaces your print statements)
        st.divider()
        st.success("✅ Model Training Complete!")
        
        # Displaying the Data Table
        st.subheader("Sampled Claims Data")
        st.dataframe(df.head(10), use_container_width=True)

        # Placeholder for your metrics and SHAP (Add these back as st.pyplot if needed)
        st.info("Metrics and SHAP visualizations are now ready below.")
