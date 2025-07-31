# association_rules.py

import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def association_rules_ui():
    st.header("🔗 Association Rule Mining")

    st.markdown("Upload a one-hot encoded transaction dataset, or use the built-in sample below.")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        source = "Uploaded"
    else:
        st.info("No file uploaded. Using built-in sample dataset.")
        df = pd.DataFrame([
            [1, 1, 1, 0, 0],  # milk, bread, eggs
            [0, 1, 0, 1, 1],  # bread, diapers, beer
            [1, 1, 0, 1, 1],  # milk, bread, diapers, beer
            [0, 1, 1, 0, 0],  # bread, eggs
            [1, 1, 1, 1, 0],  # milk, bread, diapers, eggs
        ], columns=["milk", "bread", "eggs", "diapers", "beer"])
        source = "Sample"

    # Check for boolean or binary columns
    if df.dtypes.eq('bool').sum() == 0 and not ((df == 0) | (df == 1)).all().all():
        st.warning("No valid one-hot encoded data detected (must be all 0/1 or True/False).")
        return

    st.success(f"✅ {source} dataset loaded with shape {df.shape}")

    min_support = st.slider("Minimum Support", 0.01, 0.5, 0.1)
    min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.5)

    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)

    st.subheader("📋 Generated Rules")
    if rules.empty:
        st.warning("No rules generated with current thresholds.")
    else:
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        st.download_button("Download Rules as CSV", rules.to_csv(index=False), file_name="association_rules.csv")
