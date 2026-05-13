from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import sqlite3
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "bundle_predictions.db"
MODEL_PATH = BASE_DIR / "bundle_model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"


st.set_page_config(page_title="Olist AI Bundle System", layout="wide")


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS logs
           (timestamp TEXT, product_a TEXT, product_b TEXT,
            total_price REAL, prediction TEXT, confidence REAL, freight_ratio REAL)"""
    )
    conn.commit()
    conn.close()


def save_log(product_a, product_b, price, prediction, confidence, freight_ratio) -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            product_a,
            product_b,
            price,
            prediction,
            confidence,
            freight_ratio,
        ),
    )
    conn.commit()
    conn.close()


@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, label_encoder


init_db()

try:
    model, le = load_assets()
    category_list = list(le.classes_)
except Exception:
    st.error("⚠️ Error: Ensure 'bundle_model.pkl' and 'label_encoder.pkl' are in this folder.")
    st.stop()


st.title("🛍️ Smart Bundle Effectiveness System")
st.markdown("Predicting marketplace success using the Refined Random Forest Model.")

with st.sidebar:
    st.header("🛒 Input Product Details")
    cat_a = st.selectbox("Category Product A", category_list)
    cat_b = st.selectbox("Category Product B", category_list)
    price_a = st.number_input("Price Product A ($)", min_value=1.0, value=50.0)
    price_b = st.number_input("Price Product B ($)", min_value=1.0, value=50.0)
    avg_freight = st.number_input("Average Freight Value ($)", min_value=0.0, value=15.0)


col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Analyze Bundle Effectiveness", use_container_width=True):
        total_bundle_cost = price_a + price_b
        price_ratio = price_a / (price_b + 0.001)
        freight_ratio = avg_freight / (total_bundle_cost + 0.001)
        similarity = 1 if cat_a == cat_b else 0

        enc_a = le.transform([cat_a])[0]
        enc_b = le.transform([cat_b])[0]

        features = pd.DataFrame(
            [[price_ratio, total_bundle_cost, freight_ratio, similarity, enc_a, enc_b]],
            columns=[
                "price_ratio",
                "total_bundle_cost",
                "freight_to_price_ratio",
                "product_similarity",
                "category_A_encoded",
                "category_B_encoded",
            ],
        )

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        if prediction == 1:
            st.success(f"### ✅ EFFECTIVE BUNDLE\n**Confidence Score:** {probability:.2%}")
            status = "Effective"
        else:
            st.error(f"### ❌ INEFFECTIVE BUNDLE\n**Confidence Score:** {1 - probability:.2%}")
            status = "Ineffective"

        save_log(cat_a, cat_b, total_bundle_cost, status, float(probability), float(freight_ratio))

        st.subheader("🔍 Key Business Drivers")
        if freight_ratio > 0.20:
            st.warning(
                f"⚠️ **High Shipping Impact:** Freight accounts for {freight_ratio:.2%} of the bundle cost. "
                "Consider subsidizing shipping to improve conversion."
            )
        else:
            st.info(
                f"✅ **Healthy Freight Ratio:** Shipping is only {freight_ratio:.2%} of total cost, which is ideal for e-commerce success."
            )

with col2:
    st.info(
        "**AI Model Details:**\n- Algorithm: Random Forest (Refined)\n- Optimization: Balanced Class Weights\n- Feature Priority: Freight-to-Price Ratio"
    )


st.divider()
if st.checkbox("📈 View System Analytics & History"):
    conn = sqlite3.connect(DB_PATH)
    df_logs = pd.read_sql_query("SELECT * FROM logs ORDER BY timestamp DESC", conn)
    conn.close()

    if not df_logs.empty:
        dash1, dash2, dash3 = st.columns(3)
        dash1.metric("Total Predictions", len(df_logs))
        dash2.metric("Effective Bundles Found", len(df_logs[df_logs["prediction"] == "Effective"]))
        dash3.metric("Avg. Bundle Price", f"${df_logs['total_price'].mean():.2f}")

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Prediction Distribution**")
            st.bar_chart(df_logs["prediction"].value_counts())
        with c2:
            st.write("**Price Trends Over Time**")
            st.line_chart(df_logs.set_index("timestamp")["total_price"])

        st.dataframe(df_logs, use_container_width=True)

        csv = df_logs.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Database as CSV", csv, "bundle_report.csv", "text/csv")
    else:
        st.write("Database is currently empty. Run a prediction to see analytics!")