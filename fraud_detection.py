import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

# ML & Evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_recall_curve, f1_score, classification_report,
    average_precision_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# Set Page Config for a wider dashboard
st.set_page_config(page_title="Healthcare Fraud Detector", layout="wide")

# =============================================================================
# 1. SYNTHETIC DATA GENERATION (Medicare-like Claims)
# =============================================================================

@st.cache_data
def generate_synthetic_claims(n_claims=50000, fraud_rate=0.05):
    np.random.seed(42)
    n_providers = 500
    n_patients = 10000
    icd10_codes = {
        'high_cost': ['C34.90', 'I21.3', 'J96.00', 'N17.9', 'G93.40'],
        'medium_cost': ['J18.9', 'K80.20', 'M54.5', 'I10', 'E11.9'],
        'low_cost': ['J06.9', 'R51', 'Z00.00', 'R10.9', 'M79.3']
    }
    cpt_codes = ['99213', '99214', '99215', '99223', '99232', '99291', '36415', '71046']
    
    claims = pd.DataFrame({
        'claim_id': range(1, n_claims + 1),
        'provider_id': np.random.randint(1, n_providers + 1, n_claims),
        'patient_id': np.random.randint(1, n_patients + 1, n_claims),
        'claim_date': [datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(n_claims)],
        'cpt_code': np.random.choice(cpt_codes, n_claims),
    })
    
    all_icd10 = icd10_codes['high_cost'] + icd10_codes['medium_cost'] + icd10_codes['low_cost']
    icd10_weights = [0.05]*5 + [0.15]*5 + [0.30]*5
    icd10_weights = np.array(icd10_weights) / sum(icd10_weights)
    claims['icd10_code'] = np.random.choice(all_icd10, n_claims, p=icd10_weights)
    
    cpt_base_amounts = {'99213': 75, '99214': 110, '99215': 150, '99223': 200, '99232': 80, '99291': 300, '36415': 25, '71046': 45}
    claims['base_amount'] = claims['cpt_code'].map(cpt_base_amounts)
    claims['claim_amount'] = claims['base_amount'] * np.random.uniform(0.9, 1.3, n_claims)
    
    specialties = ['Internal Medicine', 'Cardiology', 'Oncology', 'General Practice', 'Orthopedics']
    provider_specialty = {i: np.random.choice(specialties) for i in range(1, n_providers + 1)}
    claims['provider_specialty'] = claims['provider_id'].map(provider_specialty)
    claims['is_fraud'] = 0
    
    n_fraud = int(n_claims * fraud_rate)
    fraud_providers_upcoding = np.random.choice(range(1, n_providers + 1), size=20, replace=False)
    upcoding_mask = (claims['provider_id'].isin(fraud_providers_upcoding) & claims['cpt_code'].isin(['99213', '99214']))
    upcoding_indices = claims[upcoding_mask].sample(n=min(n_fraud // 2, upcoding_mask.sum())).index
    claims.loc[upcoding_indices, 'cpt_code'] = '99215'
    claims.loc[upcoding_indices, 'claim_amount'] *= 1.8
    claims.loc[upcoding_indices, 'is_fraud'] = 1
    
    claims['patient_age'] = np.random.randint(18, 95, n_claims)
    claims['days_to_payment'] = np.random.randint(5, 90, n_claims)
    return claims, icd10_codes

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

def engineer_features(df, icd10_codes):
    df = df.copy()
    provider_stats = df.groupby('provider_id').agg({'patient_id': 'nunique', 'claim_id': 'count', 'claim_amount': ['mean', 'sum', 'std']}).reset_index()
    provider_stats.columns = ['provider_id', 'unique_patients', 'total_claims', 'avg_claim_amount', 'total_billed', 'claim_amount_std']
    provider_stats['provider_patient_ratio'] = provider_stats['total_claims'] / provider_stats['unique_patients']
    df = df.merge(provider_stats[['provider_id', 'provider_patient_ratio', 'avg_claim_amount', 'total_billed', 'claim_amount_std']], on='provider_id', how='left')
    
    cpt_avg = df.groupby('cpt_code')['claim_amount'].mean().to_dict()
    df['cpt_avg_amount'] = df['cpt_code'].map(cpt_avg)
    df['claim_vs_cpt_avg_ratio'] = df['claim_amount'] / df['cpt_avg_amount']
    df['is_high_cost_icd10'] = df['icd10_code'].isin(icd10_codes['high_cost']).astype(int)
    
    provider_high_cost = df.groupby('provider_id')['is_high_cost_icd10'].mean().reset_index()
    provider_high_cost.columns = ['provider_id', 'high_cost_icd10_freq']
    df = df.merge(provider_high_cost, on='provider_id', how='left')
    
    df['is_high_em_code'] = (df['cpt_code'] == '99215').astype(int)
    provider_em = df.groupby('provider_id')['is_high_em_code'].mean().reset_index()
    provider_em.columns = ['provider_id', 'high_em_code_freq']
    df = df.merge(provider_em, on='provider_id', how='left')
    
    df['day_of_week'] = pd.to_datetime(df['claim_date']).dt.dayofweek
    df['month'] = pd.to_datetime(df['claim_date']).dt.month
    return df

# =============================================================================
# MAIN APP EXECUTION
# =============================================================================

st.title("🛡️ Healthcare Claims Fraud Detection System")
st.markdown("This system identifies **Upcoding** and **Phantom Billing** patterns using XGBoost and SHAP explainability.")

# 1. Data Generation
with st.spinner("Generating Synthetic Medicare Data..."):
    claims_df, icd10_codes = generate_synthetic_claims()
    df_featured = engineer_features(claims_df, icd10_codes)

col1, col2, col3 = st.columns(3)
col1.metric("Total Claims", len(claims_df))
col2.metric("Fraud Rate", f"{claims_df['is_fraud'].mean():.2%}")
col3.metric("Total Providers", claims_df['provider_id'].nunique())

st.subheader("Raw Claims Data Sample")
st.dataframe(claims_df.head(10), use_container_width=True)

# 2. Preparation & Training
feature_cols = [
    'claim_amount', 'provider_patient_ratio', 'avg_claim_amount', 
    'claim_vs_cpt_avg_ratio', 'high_cost_icd10_freq', 'high_em_code_freq',
    'day_of_week', 'month', 'patient_age', 'days_to_payment'
]

le_specialty = LabelEncoder()
df_featured['specialty_encoded'] = le_specialty.fit_transform(df_featured['provider_specialty'])
feature_cols.append('specialty_encoded')

X = df_featured[feature_cols].fillna(0)
y = df_featured['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Model
with st.spinner("Training XGBoost Model..."):
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, scale_pos_weight=3, eval_metric='aucpr')
    model.fit(X_train_res, y_train_res)

# 3. Evaluation UI
st.divider()
st.header("📈 Model Evaluation")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

c1, c2 = st.columns(2)
with c1:
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.table(pd.DataFrame(report).transpose())

with c2:
    st.subheader("Performance Metrics")
    st.write(f"**F1 Score:** {f1_score(y_test, y_pred):.4f}")
    st.write(f"**Average Precision:** {average_precision_score(y_test, y_proba):.4f}")

# Metrics Plots
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
precision, recall, _ = precision_recall_curve(y_test, y_proba)
ax[0].plot(recall, precision, color='blue', lw=2)
ax[0].set_title("Precision-Recall Curve")
ax[0].set_xlabel("Recall")
ax[0].set_ylabel("Precision")

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
importances.tail(10).plot(kind='barh', ax=ax[1], color='teal')
ax[1].set_title("Top Feature Importances")
st.pyplot(fig)

# 4. SHAP Explainability
st.divider()
st.header("🔍 Fraud Alert Explainability (SHAP)")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Specific Case Selection
fraud_indices = np.where(y_test == 1)[0]
selected_idx = st.selectbox("Select a Fraudulent Claim to Analyze:", fraud_indices[:10])

st.write(f"### Analysis for Claim Index: {selected_idx}")
fig_shap, ax_shap = plt.subplots()
shap.waterfall_plot(shap.Explanation(values=shap_values[selected_idx], 
                                    base_values=explainer.expected_value, 
                                    data=X_test.iloc[selected_idx], 
                                    feature_names=feature_cols), show=False)
st.pyplot(plt.gcf())

# 5. Risk Ranking
st.divider()
st.subheader("🏥 Top High-Risk Providers")
test_df = df_featured.loc[X_test.index].copy()
test_df['fraud_probability'] = y_proba
risk_ranking = test_df.groupby('provider_id').agg({
    'fraud_probability': 'mean',
    'claim_id': 'count',
    'provider_specialty': 'first'
}).sort_values('fraud_probability', ascending=False).head(10)

st.table(risk_ranking)
