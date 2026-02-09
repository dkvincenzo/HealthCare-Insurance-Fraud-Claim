"""
Healthcare Insurance Claims Fraud Detection System
Targets: Upcoding and Phantom Billing patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

# ML & Evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    precision_recall_curve, f1_score, classification_report,
    average_precision_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# =============================================================================
# 1. SYNTHETIC DATA GENERATION (Medicare-like Claims)
# =============================================================================

def generate_synthetic_claims(n_claims=50000, fraud_rate=0.05):
    """Generate synthetic healthcare claims with fraud patterns."""
    np.random.seed(42)
    
    n_providers = 500
    n_patients = 10000
    
    # ICD-10 codes (some high-cost, some routine)
    icd10_codes = {
        'high_cost': ['C34.90', 'I21.3', 'J96.00', 'N17.9', 'G93.40'],  # Cancer, MI, Resp failure
        'medium_cost': ['J18.9', 'K80.20', 'M54.5', 'I10', 'E11.9'],    # Pneumonia, stones, back pain
        'low_cost': ['J06.9', 'R51', 'Z00.00', 'R10.9', 'M79.3']        # Cold, headache, checkup
    }
    
    # CPT procedure codes
    cpt_codes = ['99213', '99214', '99215', '99223', '99232', '99291', '36415', '71046']
    
    # Generate base claims
    claims = pd.DataFrame({
        'claim_id': range(1, n_claims + 1),
        'provider_id': np.random.randint(1, n_providers + 1, n_claims),
        'patient_id': np.random.randint(1, n_patients + 1, n_claims),
        'claim_date': [datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365)) 
                       for _ in range(n_claims)],
        'cpt_code': np.random.choice(cpt_codes, n_claims),
    })
    
    # Assign ICD-10 codes with weighted probability
    all_icd10 = icd10_codes['high_cost'] + icd10_codes['medium_cost'] + icd10_codes['low_cost']
    icd10_weights = [0.05]*5 + [0.15]*5 + [0.30]*5  # Low prob for high-cost
    icd10_weights = np.array(icd10_weights) / sum(icd10_weights)
    claims['icd10_code'] = np.random.choice(all_icd10, n_claims, p=icd10_weights)
    
    # Base claim amounts by CPT code
    cpt_base_amounts = {
        '99213': 75, '99214': 110, '99215': 150, '99223': 200,
        '99232': 80, '99291': 300, '36415': 25, '71046': 45
    }
    claims['base_amount'] = claims['cpt_code'].map(cpt_base_amounts)
    claims['claim_amount'] = claims['base_amount'] * np.random.uniform(0.9, 1.3, n_claims)
    
    # Provider specialty
    specialties = ['Internal Medicine', 'Cardiology', 'Oncology', 'General Practice', 'Orthopedics']
    provider_specialty = {i: np.random.choice(specialties) for i in range(1, n_providers + 1)}
    claims['provider_specialty'] = claims['provider_id'].map(provider_specialty)
    
    # Initialize fraud label
    claims['is_fraud'] = 0
    
    # ----- INJECT FRAUD PATTERNS -----
    n_fraud = int(n_claims * fraud_rate)
    
    # Pattern 1: UPCODING - Billing higher-level codes than warranted
    fraud_providers_upcoding = np.random.choice(range(1, n_providers + 1), size=20, replace=False)
    upcoding_mask = (
        claims['provider_id'].isin(fraud_providers_upcoding) & 
        claims['cpt_code'].isin(['99213', '99214'])
    )
    upcoding_indices = claims[upcoding_mask].sample(n=min(n_fraud // 2, upcoding_mask.sum())).index
    claims.loc[upcoding_indices, 'cpt_code'] = '99215'
    claims.loc[upcoding_indices, 'claim_amount'] *= 1.8  # Inflated amount
    claims.loc[upcoding_indices, 'is_fraud'] = 1
    
    # Pattern 2: PHANTOM BILLING - Billing for services not rendered
    fraud_providers_phantom = np.random.choice(
        [p for p in range(1, n_providers + 1) if p not in fraud_providers_upcoding],
        size=15, replace=False
    )
    phantom_indices = []
    for provider in fraud_providers_phantom:
        provider_claims = claims[claims['provider_id'] == provider]
        if len(provider_claims) > 10:
            sample_claims = provider_claims.sample(n=min(50, len(provider_claims)))
            phantom_indices.extend(sample_claims.index.tolist())
    
    phantom_indices = phantom_indices[:n_fraud // 2]
    claims.loc[phantom_indices, 'is_fraud'] = 1
    
    # Add noise features
    claims['patient_age'] = np.random.randint(18, 95, n_claims)
    claims['patient_gender'] = np.random.choice(['M', 'F'], n_claims)
    claims['days_to_payment'] = np.random.randint(5, 90, n_claims)
    
    return claims, icd10_codes


# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

def engineer_features(df, icd10_codes):
    """Create fraud detection features."""
    df = df.copy()
    
    # --- Provider-level Features ---
    
    # 1. Provider-to-Patient Ratio
    provider_stats = df.groupby('provider_id').agg({
        'patient_id': 'nunique',
        'claim_id': 'count',
        'claim_amount': ['mean', 'sum', 'std']
    }).reset_index()
    provider_stats.columns = ['provider_id', 'unique_patients', 'total_claims', 
                               'avg_claim_amount', 'total_billed', 'claim_amount_std']
    provider_stats['provider_patient_ratio'] = provider_stats['total_claims'] / provider_stats['unique_patients']
    
    df = df.merge(provider_stats[['provider_id', 'provider_patient_ratio', 
                                   'avg_claim_amount', 'total_billed', 'claim_amount_std']], 
                  on='provider_id', how='left')
    
    # 2. Average Claim Value per Procedure (CPT Code)
    cpt_avg = df.groupby('cpt_code')['claim_amount'].mean().to_dict()
    df['cpt_avg_amount'] = df['cpt_code'].map(cpt_avg)
    df['claim_vs_cpt_avg_ratio'] = df['claim_amount'] / df['cpt_avg_amount']
    
    # 3. Frequency of High-Cost ICD-10 Codes per Provider
    df['is_high_cost_icd10'] = df['icd10_code'].isin(icd10_codes['high_cost']).astype(int)
    
    provider_high_cost = df.groupby('provider_id')['is_high_cost_icd10'].mean().reset_index()
    provider_high_cost.columns = ['provider_id', 'high_cost_icd10_freq']
    df = df.merge(provider_high_cost, on='provider_id', how='left')
    
    # --- Upcoding Detection Features ---
    
    # 4. High-level E&M code frequency (99215 is often upcoded)
    df['is_high_em_code'] = (df['cpt_code'] == '99215').astype(int)
    provider_em = df.groupby('provider_id')['is_high_em_code'].mean().reset_index()
    provider_em.columns = ['provider_id', 'high_em_code_freq']
    df = df.merge(provider_em, on='provider_id', how='left')
    
    # 5. Claim amount deviation from provider's mean
    df['claim_deviation'] = (df['claim_amount'] - df['avg_claim_amount']) / (df['claim_amount_std'] + 1)
    
    # --- Phantom Billing Detection Features ---
    
    # 6. Same-day claims per patient-provider pair
    df['claim_date_str'] = df['claim_date'].astype(str)
    same_day_claims = df.groupby(['provider_id', 'patient_id', 'claim_date_str']).size().reset_index(name='same_day_claim_count')
    df = df.merge(same_day_claims, on=['provider_id', 'patient_id', 'claim_date_str'], how='left')
    
    # 7. Claims per patient (from same provider)
    patient_provider_claims = df.groupby(['provider_id', 'patient_id']).size().reset_index(name='patient_provider_claim_count')
    df = df.merge(patient_provider_claims, on=['provider_id', 'patient_id'], how='left')
    
    # --- Temporal Features ---
    
    # 8. Day of week / Month patterns
    df['day_of_week'] = pd.to_datetime(df['claim_date']).dt.dayofweek
    df['month'] = pd.to_datetime(df['claim_date']).dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 9. Claims velocity (provider's claims in rolling window)
    df['claim_week'] = pd.to_datetime(df['claim_date']).dt.isocalendar().week
    weekly_claims = df.groupby(['provider_id', 'claim_week']).size().reset_index(name='weekly_claim_volume')
    df = df.merge(weekly_claims, on=['provider_id', 'claim_week'], how='left')
    
    return df


# =============================================================================
# 3. DATA PREPARATION & IMBALANCE HANDLING
# =============================================================================

def prepare_data(df):
    """Prepare features for modeling."""
    
    feature_cols = [
        'claim_amount', 'provider_patient_ratio', 'avg_claim_amount', 
        'claim_vs_cpt_avg_ratio', 'high_cost_icd10_freq', 'high_em_code_freq',
        'claim_deviation', 'same_day_claim_count', 'patient_provider_claim_count',
        'day_of_week', 'month', 'is_weekend', 'weekly_claim_volume',
        'patient_age', 'days_to_payment', 'is_high_cost_icd10', 'is_high_em_code'
    ]
    
    # Encode categorical variables
    le_cpt = LabelEncoder()
    le_specialty = LabelEncoder()
    
    df['cpt_code_encoded'] = le_cpt.fit_transform(df['cpt_code'])
    df['specialty_encoded'] = le_specialty.fit_transform(df['provider_specialty'])
    
    feature_cols.extend(['cpt_code_encoded', 'specialty_encoded'])
    
    X = df[feature_cols].fillna(0)
    y = df['is_fraud']
    
    return X, y, feature_cols


# =============================================================================
# 4. EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, feature_cols):
    """Comprehensive evaluation with PR focus."""
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1:.4f}")
    
    avg_precision = average_precision_score(y_test, y_pred_proba)
    print(f"Average Precision (AP): {avg_precision:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
    
    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.fill_between(recall, precision, alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AP={avg_precision:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True).tail(10)
    
    plt.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importances')
    
    plt.subplot(1, 3, 3)
    f1_scores = []
    threshold_range = np.arange(0.1, 0.9, 0.05)
    for thresh in threshold_range:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        f1_scores.append(f1_score(y_test, y_pred_thresh))
    
    plt.plot(threshold_range, f1_scores, 'g-', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Decision Threshold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fraud_detection_evaluation.png', dpi=150, bbox_inches='tight')
    print("\nSaved: fraud_detection_evaluation.png")
    plt.close()
    
    return y_pred_proba


# =============================================================================
# 5. SHAP EXPLAINABILITY ANALYSIS
# =============================================================================

def shap_analysis(model, X_test, feature_cols, df_featured, y_test):
    """SHAP analysis for fraud alert explanations."""
    
    print("\n" + "="*60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("="*60)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Global Feature Importance
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
    plt.title('SHAP Feature Importance (Global)')
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
    print("Saved: shap_summary.png")
    plt.close()
    
    # Analyze specific fraud cases
    print("\n--- Analyzing Flagged Fraud Cases ---")
    
    fraud_indices = np.where((y_test.values == 1))[0][:5]
    
    for i, idx in enumerate(fraud_indices):
        print(f"\n{'='*40}")
        print(f"FRAUD ALERT #{i+1}")
        print(f"{'='*40}")
        
        feature_values = X_test.iloc[idx]
        shap_contributions = shap_values[idx]
        
        contrib_df = pd.DataFrame({
            'feature': feature_cols,
            'value': feature_values.values,
            'shap_contribution': shap_contributions
        }).sort_values('shap_contribution', key=abs, ascending=False)
        
        print("\nTop Contributing Factors to Fraud Alert:")
        for _, row in contrib_df.head(5).iterrows():
            direction = "↑ INCREASES" if row['shap_contribution'] > 0 else "↓ DECREASES"
            print(f"  * {row['feature']}: {row['value']:.2f}")
            print(f"    SHAP: {row['shap_contribution']:.4f} ({direction} fraud probability)")
        
        if i == 0:
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_contributions,
                    base_values=explainer.expected_value,
                    data=feature_values.values,
                    feature_names=feature_cols
                ),
                show=False
            )
            plt.title(f'SHAP Waterfall - Fraud Case #{i+1}')
            plt.tight_layout()
            plt.savefig('shap_waterfall_example.png', dpi=150, bbox_inches='tight')
            print("\nSaved: shap_waterfall_example.png")
            plt.close()
    
    # Provider-level fraud risk analysis
    print("\n--- High-Risk Provider Analysis ---")
    test_indices = X_test.index
    test_df = df_featured.loc[test_indices].copy()
    test_df['fraud_probability'] = model.predict_proba(X_test)[:, 1]
    
    provider_risk = test_df.groupby('provider_id').agg({
        'fraud_probability': 'mean',
        'claim_id': 'count',
        'high_em_code_freq': 'first',
        'provider_patient_ratio': 'first'
    }).sort_values('fraud_probability', ascending=False).head(10)
    
    print("\nTop 10 High-Risk Providers:")
    print(provider_risk.to_string())
    
    return explainer, shap_values


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("HEALTHCARE CLAIMS FRAUD DETECTION SYSTEM")
    print("="*60)
    
    # 1. Generate Data
    print("\n[1/5] Generating synthetic claims data...")
    claims_df, icd10_codes = generate_synthetic_claims(n_claims=50000, fraud_rate=0.05)
    print(f"Dataset shape: {claims_df.shape}")
    print(f"Fraud rate: {claims_df['is_fraud'].mean():.2%}")
    
    # 2. Feature Engineering
    print("\n[2/5] Engineering features...")
    df_featured = engineer_features(claims_df, icd10_codes)
    print(f"Features created: {df_featured.shape[1]} columns")
    
    # 3. Prepare Data
    print("\n[3/5] Preparing data and handling imbalance...")
    X, y, feature_cols = prepare_data(df_featured)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Fraud in training: {y_train.sum()} ({y_train.mean():.2%})")
    
    # Apply SMOTE
    print("Applying SMOTE for class balancing...")
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE - Training: {len(X_train_resampled)}, Fraud: {y_train_resampled.sum()}")
    
    # 4. Train Model
    print("\n[4/5] Training XGBoost classifier...")
    scale_pos_weight = (y_train_resampled == 0).sum() / (y_train_resampled == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train_resampled, y_train_resampled,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    print("Model training complete!")
    
    # 5. Evaluate
    print("\n[5/5] Evaluating model...")
    y_pred_proba = evaluate_model(model, X_test, y_test, feature_cols)
    
    # 6. SHAP Analysis
    explainer, shap_values = shap_analysis(model, X_test, feature_cols, df_featured, y_test)
    
    print("\n" + "="*60)
    print("FRAUD DETECTION SYSTEM COMPLETE")
    print("="*60)
    print("\nOutput files generated:")
    print("  - fraud_detection_evaluation.png")
    print("  - shap_summary.png")
    print("  - shap_waterfall_example.png")

