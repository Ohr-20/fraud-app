import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd

st.set_page_config(page_title="Fraud Detector", page_icon="🔍", layout="wide")

# ---------- DARK THEME ----------
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e6e6e6; }
    .stButton>button {
        background-color: #2563eb; color: white; border: none;
        border-radius: 6px; padding: 0.6rem 1.5rem; font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover { background-color: #1d4ed8; }
    .stNumberInput input { background-color: #1a1f2b; color: #e6e6e6; }
    h1, h2, h3 { color: #f1f5f9; }
    .result-box {
        padding: 1.5rem; border-radius: 8px; text-align: center;
        font-size: 1.4rem; font-weight: 700; margin-top: 1rem;
    }
    .fraud { background-color: #3b0d0d; color: #ff6b6b; border: 1px solid #ff6b6b; }
    .legit { background-color: #0d3b1d; color: #4ade80; border: 1px solid #4ade80; }
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "avg_min_between_sent_tnx", "avg_min_between_received_tnx",
    "time_diff_between_first_and_last_mins", "sent_tnx", "received_tnx",
    "number_of_created_contracts", "unique_received_from_addresses",
    "unique_sent_to_addresses", "min_value_received", "max_value_received",
    "avg_val_received", "min_val_sent", "max_val_sent", "avg_val_sent",
    "min_value_sent_to_contract", "max_val_sent_to_contract",
    "avg_value_sent_to_contract",
    "total_transactions_including_tnx_to_create_contract",
    "total_ether_sent", "total_ether_received", "total_ether_sent_contracts",
    "total_ether_balance", "total_erc20_tnxs", "erc20_total_ether_received",
    "erc20_total_ether_sent", "erc20_total_ether_sent_contract",
    "erc20_uniq_sent_addr", "erc20_uniq_rec_addr", "erc20_uniq_sent_addr_1",
    "erc20_uniq_rec_contract_addr", "erc20_avg_time_between_sent_tnx",
    "erc20_avg_time_between_rec_tnx", "erc20_avg_time_between_rec_2_tnx",
    "erc20_avg_time_between_contract_tnx", "erc20_min_val_rec",
    "erc20_max_val_rec", "erc20_avg_val_rec", "erc20_min_val_sent",
    "erc20_max_val_sent", "erc20_avg_val_sent", "erc20_min_val_sent_contract",
    "erc20_max_val_sent_contract", "erc20_avg_val_sent_contract",
    "erc20_uniq_sent_token_name", "erc20_uniq_rec_token_name",
]


@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("model.json")
    return model


model = load_model()

st.title("Ethereum Wallet Fraud Detector")
st.write("Enter wallet transaction stats below. The model predicts fraud risk.")

tab1, tab2 = st.tabs(["Manual Input", "Upload CSV"])

with tab1:
    with st.form("predict_form"):
        cols = st.columns(3)
        values = {}
        for i, feat in enumerate(FEATURES):
            col = cols[i % 3]
            label = feat.replace("_", " ").title()
            values[feat] = col.number_input(label, value=0.0, format="%.4f", key=feat)

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = pd.DataFrame([values])[FEATURES]
        prob = model.predict_proba(row)[0][1]
        pred = int(prob >= 0.5)

        if pred == 1:
            st.markdown(
                f'<div class="result-box fraud">FRAUD DETECTED — Risk score: {prob:.2%}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-box legit">LEGITIMATE — Risk score: {prob:.2%}</div>',
                unsafe_allow_html=True,
            )

with tab2:
    st.write("Upload a CSV with the 45 required columns. Each row is one wallet.")
    file = st.file_uploader("Choose CSV file", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        missing = [f for f in FEATURES if f not in df.columns]

        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            df_pred = df[FEATURES].copy()
            probs = model.predict_proba(df_pred)[:, 1]
            df["fraud_probability"] = probs
            df["prediction"] = np.where(probs >= 0.5, "Fraud", "Legitimate")

            st.dataframe(df, use_container_width=True)

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                csv_out,
                "predictions.csv",
                "text/csv",
            )
