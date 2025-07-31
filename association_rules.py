# association_rules.py

import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import altair as alt

def association_rules_ui():
    st.header("🔗 Association Rule Mining")

    # 🔹 Explanatory panel
    with st.expander("ℹ️ How Association Rule Mining Works"):
        st.markdown("""
**Association Rule Mining** helps find interesting relationships between items in transaction data.

#### 📚 Key Concepts:
- **Support**: How often items appear together  
  E.g., Support(Bread → Milk) = 60% means 60% of transactions contain both.
- **Confidence**: Likelihood of buying consequent given antecedent  
  E.g., Confidence(Bread → Milk) = 80% means 80% of bread buyers also buy milk.
- **Lift**: Strength of correlation (Lift > 1 = positive association)

#### 🛒 Example:
If many customers buy **diapers** and **beer** together, the rule:  
`{diapers} → {beer}` could be discovered, helping marketing decisions.
""")

    # 🔹 File uploader or fallback sample
    uploaded_file = st.file_uploader("Upload a transaction dataset (one-hot encoded or basket format)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Uploaded dataset loaded with shape {df.shape}")
    else:
        st.info("📄 No file uploaded. Using built-in sample dataset.")
        df = pd.DataFrame({
            'milk':     [1, 0, 1, 1, 0],
            'bread':    [1, 1, 0, 1, 0],
            'eggs':     [0, 1, 1, 1, 1],
            'diapers':  [0, 1, 0, 1, 1],
            'beer':     [0, 1, 0, 0, 1]
        }).astype(bool)
        st.success(f"✅ Sample dataset loaded with shape {df.shape}")

    if df.dtypes.eq('bool').sum() == 0:
        st.warning("⚠️ No boolean (True/False) columns detected. Please upload one-hot encoded data.")
        return

    # 🔹 Sliders for thresholds
    min_support = st.slider("📉 Minimum Support", 0.01, 0.5, 0.1)
    min_conf = st.slider("📈 Minimum Confidence", 0.1, 1.0, 0.5)

    # 🔹 Apriori and rule generation
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)

    # 🔍 Optional item filter
    selected_item = st.text_input("🔍 Filter rules containing item (optional):")
    if selected_item:
        rules = rules[rules['antecedents'].astype(str).str.contains(selected_item) |
                      rules['consequents'].astype(str).str.contains(selected_item)]

    # 📊 Visualization: Support vs Confidence
    if not rules.empty:
        st.subheader("📊 Rule Strength Visualization")
        chart = alt.Chart(rules).mark_circle(size=100).encode(
            x='support:Q',
            y='confidence:Q',
            color='lift:Q',
            tooltip=['antecedents', 'consequents', 'support', 'confidence', 'lift']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

    # 📈 Item frequency bar chart
    st.subheader("📦 Item Frequencies")
    item_freq = df.sum().sort_values(ascending=False)
    st.bar_chart(item_freq)

    # 📋 Show rules
    st.subheader("📋 Generated Rules")
    if rules.empty:
        st.warning("No rules found with the current thresholds.")
    else:
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

        # 💾 Download buttons
        st.download_button("📥 Download Rules as CSV", rules.to_csv(index=False), file_name="association_rules.csv")

    # 💾 Sample dataset download
    if not uploaded_file:
        st.download_button("📥 Download Sample Dataset", df.astype(int).to_csv(index=False), file_name="sample_transactions.csv")

