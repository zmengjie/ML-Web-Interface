# topic_modeling.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

def topic_modeling_ui():
    st.header("🧠 Topic Modeling")
    st.markdown("Upload your own text dataset or use our built-in example to explore topics.")

    with st.expander("📘 What is Topic Modeling?", expanded=False):
        st.markdown("""
        - **Topic modeling** is an **unsupervised** technique to discover abstract topics in a collection of documents.
        - We use **Latent Dirichlet Allocation (LDA)** to uncover these topics based on word co-occurrence.
        - Each topic is represented by a set of **keywords**, and each document is a mix of topics.
        """)

    uploaded_file = st.file_uploader("📤 Upload CSV (with one text column)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File loaded successfully!")
    else:
        # Built-in sample
        st.info("🧪 No file uploaded. Using built-in sample dataset.")
        df = pd.DataFrame({
            'text': [
                "The stock market crashed due to inflation and economic downturn.",
                "Climate change is causing severe weather conditions globally.",
                "Artificial intelligence is revolutionizing many industries.",
                "Healthcare advances are improving life expectancy.",
                "Education is becoming more accessible through online platforms."
            ]
        })

    # Select a text column
    text_cols = df.select_dtypes(include='object').columns.tolist()
    if not text_cols:
        st.warning("No text columns found.")
        return
    col = st.selectbox("📄 Select text column", text_cols)

    # Preprocessing and LDA
    num_topics = st.slider("🔢 Number of Topics", 2, 10, 3)
    max_features = st.slider("🔤 Max Vocabulary Size", 100, 2000, 1000)
    n_top_words = st.slider("📌 Top Words per Topic", 5, 20, 10)

    vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    X = vectorizer.fit_transform(df[col].fillna(""))
    words = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    # Display topics
    st.subheader("📋 Topics and Top Keywords")
    for idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-n_top_words:][::-1]]
        st.markdown(f"**Topic {idx+1}:** " + ", ".join(top_words))

    # Optional visualization
    st.subheader("📊 Topic-Word Distribution (Bar Chart)")
    selected_topic = st.selectbox("Select a topic to visualize", list(range(num_topics)))
    topic_words = lda.components_[selected_topic]
    top_indices = topic_words.argsort()[-n_top_words:][::-1]
    top_keywords = [words[i] for i in top_indices]
    top_scores = topic_words[top_indices]

    fig, ax = plt.subplots()
    ax.barh(top_keywords[::-1], top_scores[::-1], color='skyblue')
    ax.set_xlabel("Word Importance")
    ax.set_title(f"Topic {selected_topic + 1} Keyword Distribution")
    st.pyplot(fig)

