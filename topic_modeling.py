# topic_modeling.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Sample data
BUILTIN_DATA = pd.DataFrame({
    "text": [
        "The weather is hot and dry in the summer.",
        "Rainfall has increased due to climate change.",
        "He installed solar panels to reduce electricity cost.",
        "Stock markets show strong growth this quarter.",
        "Economic forecasts predict a recession.",
        "The company announced its earnings report.",
        "Artificial intelligence is transforming technology.",
        "Machine learning and deep learning are core AI areas.",
        "Neural networks outperform traditional models.",
        "Vaccines have helped reduce the spread of disease.",
        "The new drug shows promise in cancer treatment.",
        "Hospitals are overwhelmed with COVID-19 cases."
    ]
})

def topic_modeling_ui():
    st.header("🧠 Topic Modeling")

    st.markdown("""
    This module identifies hidden **topics** in a collection of documents using **Latent Dirichlet Allocation (LDA)**.
    
    - Topics are clusters of frequently co-occurring words.
    - Each document can belong to multiple topics with different proportions.
    - Useful for exploring themes in news, feedback, papers, etc.
    """)

    uploaded_file = st.file_uploader("📤 Upload a CSV with a text column", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded file with shape {df.shape}")
    else:
        df = BUILTIN_DATA.copy()
        st.info("🧪 No file uploaded. Using built-in dataset.")
        st.dataframe(df)

    text_columns = df.select_dtypes(include="object").columns.tolist()
    if not text_columns:
        st.warning("No text columns found.")
        return

    col = st.selectbox("📝 Select text column", text_columns)
    num_topics = st.slider("📊 Number of topics", 2, 10, 4)

    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df[col].fillna(""))

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    topic_words = []
    words = vectorizer.get_feature_names_out()
    for i, topic in enumerate(lda.components_):
        top_words = [words[j] for j in topic.argsort()[-8:][::-1]]
        topic_words.append(", ".join(top_words))

    st.subheader("🔑 Top Words per Topic")
    for i, terms in enumerate(topic_words):
        st.markdown(f"**Topic {i+1}:** {terms}")

    st.subheader("📊 Topic Distribution per Document")
    doc_topic = lda.transform(X)
    topic_df = pd.DataFrame(doc_topic, columns=[f"Topic {i+1}" for i in range(num_topics)])
    st.dataframe(topic_df)

    st.subheader("📈 Topic Share Bar Chart")
    topic_avg = topic_df.mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=topic_avg.values, y=topic_avg.index, ax=ax)
    ax.set_xlabel("Average Topic Share")
    ax.set_ylabel("Topic")
    st.pyplot(fig)

    csv = topic_df.copy()
    csv['original_text'] = df[col].values
    st.download_button("⬇️ Download Topic Distribution", csv.to_csv(index=False), file_name="topic_modeling_output.csv")


