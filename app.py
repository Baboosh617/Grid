import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ---------- LOAD ARTIFACTS ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load('grid_stability_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    return model, scaler, le, feature_cols

model, scaler, le, feature_cols = load_artifacts()

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Grid Stability Predictor", layout="wide")
st.title(" Electrical Grid Stability Predictor")
st.markdown("""
### What does this tool do?
It predicts whether a **power grid** with four generators will be **stable** or **unstable**  
based on 12 electrical parameters. This helps engineers assess grid reliability under different operating conditions.
""")

# ---------- SIDEBAR WITH INFO ----------
with st.sidebar:
    st.header(" About the Model")
    st.markdown("""
    - **Model type**: XGBoost Classifier  
    - **Accuracy**: **97.6%** on test data  
    - **Inputs**: 12 features (time constants, power outputs, coupling strengths)  
    - **Output**: `stable` / `unstable`  
    - **Training data**: 10,000 synthetic grid states  
    """)
    st.header(" Need help?")
    st.markdown("""
    Hover over any input field for an explanation.  
    Use the **preset examples** below to see typical values.
    """)
    
    # Preset examples
    st.subheader(" Load Example")
    if st.button(" Typical Unstable Grid"):
        st.session_state.tau1 = 5.0
        st.session_state.tau2 = 5.0
        st.session_state.tau3 = 5.0
        st.session_state.tau4 = 5.0
        st.session_state.p1 = 0.0
        st.session_state.p2 = 0.0
        st.session_state.p3 = 0.0
        st.session_state.p4 = 0.0
        st.session_state.g1 = 0.5
        st.session_state.g2 = 0.5
        st.session_state.g3 = 0.5
        st.session_state.g4 = 0.5
        st.success("Example loaded! Click 'Predict'.")
    if st.button(" Typical Stable Grid"):
        st.session_state.tau1 = 2.0
        st.session_state.tau2 = 2.0
        st.session_state.tau3 = 2.0
        st.session_state.tau4 = 2.0
        st.session_state.p1 = 3.0
        st.session_state.p2 = 3.0
        st.session_state.p3 = 3.0
        st.session_state.p4 = 3.0
        st.session_state.g1 = 0.1
        st.session_state.g2 = 0.1
        st.session_state.g3 = 0.1
        st.session_state.g4 = 0.1
        st.success("Example loaded! Click 'Predict'.")

# ---------- MAIN INPUT AREA ----------
st.markdown("###  Enter the 12 grid parameters")

# Use three columns for grouping
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader(" Time Constants (τ) – seconds")
    tau1 = st.number_input("Generator 1 τ", value=5.0, step=0.5,
                           help="Response time of generator 1 (higher = slower response)", key="tau1")
    tau2 = st.number_input("Generator 2 τ", value=5.0, step=0.5,
                           help="Response time of generator 2", key="tau2")
    tau3 = st.number_input("Generator 3 τ", value=5.0, step=0.5,
                           help="Response time of generator 3", key="tau3")
    tau4 = st.number_input("Generator 4 τ", value=5.0, step=0.5,
                           help="Response time of generator 4", key="tau4")

with col2:
    st.subheader(" Power Outputs (p) – per unit")
    p1 = st.number_input("Generator 1 power", value=0.0, step=0.5,
                         help="Active power output (negative = absorbing power)", key="p1")
    p2 = st.number_input("Generator 2 power", value=0.0, step=0.5,
                         help="Active power output", key="p2")
    p3 = st.number_input("Generator 3 power", value=0.0, step=0.5,
                         help="Active power output", key="p3")
    p4 = st.number_input("Generator 4 power", value=0.0, step=0.5,
                         help="Active power output", key="p4")

with col3:
    st.subheader(" Grid Coupling (g) – per unit")
    g1 = st.number_input("Coupling 1", value=0.5, step=0.1,
                         help="Coupling strength between generator 1 and the grid", key="g1")
    g2 = st.number_input("Coupling 2", value=0.5, step=0.1,
                         help="Coupling strength for generator 2", key="g2")
    g3 = st.number_input("Coupling 3", value=0.5, step=0.1,
                         help="Coupling strength for generator 3", key="g3")
    g4 = st.number_input("Coupling 4", value=0.5, step=0.1,
                         help="Coupling strength for generator 4", key="g4")

# Build input DataFrame
input_data = pd.DataFrame([{
    'tau1': tau1, 'tau2': tau2, 'tau3': tau3, 'tau4': tau4,
    'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4,
    'g1': g1, 'g2': g2, 'g3': g3, 'g4': g4
}])

# Prediction button
if st.button(" Predict Stability", type="primary", use_container_width=True):
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    pred_encoded = model.predict(input_scaled)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    
    # Display result with visual feedback
    st.markdown("---")
    st.subheader(" Prediction Result")
    
    # Confidence (probability)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_scaled)[0]
        prob_stable = probs[le.transform(['stable'])[0]]
        prob_unstable = probs[le.transform(['unstable'])[0]]
        
        # Show confidence as a progress bar
        if pred_label == "stable":
            st.success(f"###  The grid is **STABLE**")
            st.progress(float(prob_stable), text=f"Confidence: {prob_stable*100:.1f}%")
        else:
            st.error(f"###  The grid is **UNSTABLE**")
            st.progress(float(prob_unstable), text=f"Confidence: {prob_unstable*100:.1f}%")
        
        # Show detailed probabilities
        st.markdown("**Model confidence breakdown:**")
        prob_df = pd.DataFrame({
            "Outcome": le.classes_,
            "Probability": probs
        }).sort_values("Probability", ascending=False)
        st.dataframe(prob_df, use_container_width=True)
    
    # Explain what the prediction means
    st.markdown("---")
    st.markdown("####  What does this mean?")
    if pred_label == "stable":
        st.markdown("""
        - The grid is **operating within safe limits**.
        - No immediate risk of blackout or equipment damage.
        - The combination of time constants, power outputs, and coupling strengths keeps the system synchronized.
        """)
    else:
        st.markdown("""
        - The grid is **at risk of instability**.
        - Possible causes: high power imbalances, weak coupling, or slow generator responses.
        - Action may be required (e.g., re‑dispatching power, adjusting controls).
        """)
    
    # Optional: Show a simple gauge chart (using plotly)
    if hasattr(model, "predict_proba"):
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob_stable*100 if pred_label=="stable" else prob_unstable*100,
            title = {'text': "Prediction Confidence (%)"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {'axis': {'range': [0, 100]},
                     'bar': {'color': "darkgreen" if pred_label=="stable" else "darkred"},
                     'steps' : [
                         {'range': [0, 50], 'color': "lightgray"},
                         {'range': [50, 100], 'color': "gray"}],
                     'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)