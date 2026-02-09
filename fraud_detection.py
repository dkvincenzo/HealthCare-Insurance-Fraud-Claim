# =============================================================================
# MAIN STREAMLIT EXECUTION (Fixed for Web Interface)
# =============================================================================
import streamlit as st

# Setup the interface you want
st.set_page_config(page_title="Healthcare Fraud Detection", layout="wide")

with st.sidebar:
    st.header("⚙️ Configuration")
    n_claims_input = st.slider("Number of Claims", 10000, 100000, 50000)
    fraud_rate_input = st.slider("Fraud Rate (%)", 1, 10, 5)
    run_button = st.button("🔄 Generate Data & Train Model", type="primary")

st.title("🏥 Healthcare Insurance Claims Fraud Detection")
st.subheader("Detecting Upcoding & Phantom Billing Patterns")

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
    # 1. Generate Data
    with st.status("Running Fraud Detection Pipeline...", expanded=True) as status:
        st.write("Generating synthetic claims data...")
        claims_df, icd10_codes = generate_synthetic_claims(n_claims=n_claims_input, fraud_rate=fraud_rate_input/100)
        
        st.write("Engineering features...")
        df_featured = engineer_features(claims_df, icd10_codes)
        
        st.write("Preparing data and handling imbalance...")
        X, y, feature_cols = prepare_data(df_featured)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        smote = SMOTE(sampling_strategy=0.3, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        st.write("Training XGBoost classifier...")
        scale_pos_weight = (y_train_resampled == 0).sum() / (y_train_resampled == 1).sum()
        model = xgb.XGBClassifier(n_estimators=200, max_depth=6, scale_pos_weight=scale_pos_weight, eval_metric='aucpr')
        model.fit(X_train_resampled, y_train_resampled)
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    # 2. Display Results in UI
    st.success("✅ Model training and evaluation complete!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Claims", len(claims_df))
    col2.metric("Actual Fraud Rate", f"{claims_df['is_fraud'].mean():.2%}")
    col3.metric("F1 Score", f"{f1_score(y_test, model.predict(X_test)):.4f}")

    st.divider()
    
    # Show Evaluation Charts (This replaces plt.savefig)
    st.header("📈 Model Performance")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    fig_ev, ax_ev = plt.subplots(1, 2, figsize=(12, 5))
    ax_ev[0].plot(recall, precision, 'b-')
    ax_ev[0].set_title("Precision-Recall Curve")
    
    importance_df = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_}).sort_values('importance').tail(10)
    ax_ev[1].barh(importance_df['feature'], importance_df['importance'], color='steelblue')
    ax_ev[1].set_title("Top 10 Feature Importances")
    st.pyplot(fig_ev)

    # 3. SHAP Analysis in UI
    st.header("🔍 Fraud Explainability (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    fraud_indices = np.where(y_test.values == 1)[0]
    selected_idx = st.selectbox("Select a flagged fraud case to analyze:", fraud_indices[:10])
    
    st.write(f"### SHAP Waterfall for Case Index {selected_idx}")
    fig_sh = plt.figure()
    shap.waterfall_plot(shap.Explanation(values=shap_values[selected_idx], 
                                        base_values=explainer.expected_value, 
                                        data=X_test.iloc[selected_idx].values, 
                                        feature_names=feature_cols), show=False)
    st.pyplot(plt.gcf())
