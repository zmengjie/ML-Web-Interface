# association_rules.py

import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO

def basket_to_onehot(df):
    basket = df.stack().reset_index(level=1, drop=True).to_frame('item')
    basket['value'] = True
    onehot = basket.pivot_table(index=basket.index, columns='item', values='value', fill_value=False)
    return onehot.astype(bool)

def plot_network_graph(rules_df):
    G = nx.DiGraph()
    for _, row in rules_df.iterrows():
        for antecedent in row['antecedents']:
            for consequent in row['consequents']:
                G.add_edge(antecedent, consequent, weight=row['lift'])

    pos = nx.spring_layout(G, seed=42)
    node_colors = ['dodgerblue' if G.out_degree(n) > 0 else 'mediumseagreen' for n in G.nodes()]
    edge_colors = ['#555' for _ in G.edges()]

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors,
            node_size=2500, font_size=10, width=2, arrowsize=20, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
    st.pyplot(plt.gcf())

    buf = BytesIO()
    plt.savefig(buf, format="png")
    st.download_button("🖼️ Download Graph as PNG", buf.getvalue(), file_name="association_graph.png", mime="image/png")
    buf.close()
    plt.clf()

def association_rules_ui():
    st.header("🔗 Association Rule Mining")
    st.markdown("Upload a transaction dataset (one-hot encoded or basket format).")

    with st.expander("🧭 What is this module doing?", expanded=False):
        st.markdown("""
        - **Apriori Algorithm** finds frequent itemsets and builds association rules.
        - **Support**: How often a rule appears in the dataset.
        - **Confidence**: How often the rule is correct.
        - **Lift**: How much more likely the consequent is, given the antecedent.
        - A **lift > 1** means a strong association.
        """)

    with st.expander("🧺 Example Basket Format"):
        st.markdown("""
        If your data is in this format:
        ```
        transaction_id,item
        1,milk
        1,bread
        2,diapers
        ...
        ```
        It will be automatically converted to one-hot format.
        """)

    uploaded_file = st.file_uploader("📤 Upload CSV (optional)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.info(f"✅ Uploaded dataset shape: {df.shape}")
    else:
        sample_data = {
            'milk': [1, 1, 1, 0, 1],
            'bread': [1, 1, 0, 1, 0],
            'eggs': [0, 1, 1, 1, 0],
            'diapers': [0, 0, 1, 1, 1],
            'beer': [0, 0, 0, 1, 1],
        }
        df = pd.DataFrame(sample_data).astype(bool)
        st.info("🧪 No file uploaded. Using built-in sample dataset.")
        st.success(f"✅ Sample dataset loaded with shape {df.shape}")

    if df.dtypes.eq('bool').sum() == 0:
        st.warning("No boolean (True/False) columns detected. Converting basket format to one-hot.")
        df = basket_to_onehot(df)
        st.success(f"Converted to one-hot with shape: {df.shape}")

    min_support = st.slider("📊 Minimum Support", 0.01, 0.5, 0.1)
    min_conf = st.slider("✅ Minimum Confidence", 0.1, 1.0, 0.5)
    min_lift = st.slider("📈 Minimum Lift", 0.5, 5.0, 1.0)

    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
    rules = rules[rules['lift'] >= min_lift]

    st.subheader("📋 Generated Rules")
    st.caption(f"📊 Showing rules with support ≥ {min_support}, confidence ≥ {min_conf}, lift ≥ {min_lift}")
    if rules.empty:
        st.warning("No rules found with current thresholds.")
    else:
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        st.download_button("⬇️ Download Rules as CSV", rules.to_csv(index=False), file_name="association_rules.csv")

        st.subheader("🌐 Rule Network Graph")
        plot_network_graph(rules)

    with st.expander("📚 Guided Tour (Optional Teaching Aid)"):
        st.markdown("### Step 1: Dataset Structure\nEach row is a transaction. Each column is an item (True if purchased).")
        st.markdown("### Step 2: Support & Confidence\n- Support: Proportion of transactions containing itemset.\n- Confidence: P(consequent | antecedent).")
        st.markdown("### Step 3: Lift\n- Lift > 1 implies positive association between items.")
        st.markdown("### Step 4: Network Graph\nNodes are items. Arrows indicate strong rules with lift above your threshold.")


