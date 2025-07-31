# association_rules.py

import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def association_rules_ui():
    st.header("🔗 Association Rule Mining")

    uploaded_file = st.file_uploader("Upload a transaction dataset (one-hot encoded or basket format)", type="csv")
    if not uploaded_file:
        return

    df = pd.read_csv(uploaded_file)
    if df.dtypes.eq('bool').sum() == 0:
        st.warning("No boolean (True/False) columns detected. Please upload one-hot encoded data.")
        return

    min_support = st.slider("Minimum support", 0.01, 0.5, 0.1)
    min_conf = st.slider("Minimum confidence", 0.1, 1.0, 0.5)

    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)

    st.subheader("Generated Rules")
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    st.download_button("Download Rules as CSV", rules.to_csv(index=False), file_name="association_rules.csv")
