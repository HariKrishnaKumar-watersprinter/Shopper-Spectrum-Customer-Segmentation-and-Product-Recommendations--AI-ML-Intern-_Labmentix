import streamlit as st
import pandas as pd, numpy as np, joblib
from sklearn.metrics.pairwise import cosine_similarity
from similarity.cousine import similarity
import os
st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="wide")
st.title("🛒 Shopper Spectrum — Customer Segmentation & Product Recommendations")

tab1, tab2 = st.tabs(["🎯 Product Recommendation", "👤 Customer Segmentation"])
model_path = os.getcwd()
# ---------- Load artifacts ----------

kmeans_path = os.path.join(model_path, "kmeans.pkl")
scaler_path = os.path.join(model_path, "scaler.pkl")

kmeans   = joblib.load(kmeans_path)
scaler   = joblib.load(scaler_path)

# ---------- TAB 1: Recommendation ----------
with tab1:
    st.subheader("Get 5 Similar Products")
    product = st.text_input("Enter product name:")
    if st.button("Get Recommendations") and product:
        product = product.strip().upper()
        item_sim = similarity()
        if product in item_sim.index:
            recs = item_sim[product].sort_values(ascending=False).drop(product).head(5)
            for i, (p, s) in enumerate(recs.items(), 1):
                st.success(f"{i}. {p}   (similarity = {s:.3f})")
        else:
            st.error("Product not found in catalogue.")

# ---------- TAB 2: Customer Segmentation ----------
with tab2:
    st.subheader("Predict Customer Segment")
    recency   = st.number_input("Recency (days since last purchase)", 0, 1000, 30)
    frequency = st.number_input("Frequency (number of purchases)", 1, 5000, 5)
    monetary  = st.number_input("Monetary (total spend £)", 0.0, 1_000_000.0, 500.0)

    if st.button("Predict Cluster"):
        x = np.array([[np.log1p(recency), np.log1p(frequency), np.log1p(monetary)]])
        x_scaled = scaler.transform(x)
        cluster  = kmeans.predict(x_scaled)[0]
        label_map = {0:'Regular',1:'High-Value',2:'At-Risk',3:'Occasional'}
        st.markdown(f"### 🏷️ Predicted Segment: **{label_map[cluster]}**")
        st.write("Cluster description:")
        desc = {
            'High-Value': "Frequent, recent and big spenders — loyalty perks.",
            'Regular'   : "Steady buyers — upsell premium products.",
            'Occasional': "Rare purchasers — re-engagement offers.",
            'At-Risk'   : "Inactive customers — win-back campaigns."
        }
        st.info(desc[label_map[cluster]])
