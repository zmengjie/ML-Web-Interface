# topic_modeling.py


import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.graph_objects as go

def topic_modeling_ui():
    st.header("🧠 Topic Modeling")

    uploaded_file = st.file_uploader("Upload a CSV with a text column", type="csv")
    if not uploaded_file:
        return

    df = pd.read_csv(uploaded_file)
    text_columns = df.select_dtypes(include="object").columns.tolist()
    if not text_columns:
        st.warning("No text columns found.")
        return

    col = st.selectbox("Select text column", text_columns)
    num_topics = st.slider("Number of topics", 2, 15, 5)

    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df[col].fillna(""))

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    words = vectorizer.get_feature_names_out()

    st.subheader("📊 Top Words per Topic")
    for i, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-10:][::-1]
        top_words = [words[j] for j in top_indices]
        top_weights = topic[top_indices]

        fig = go.Figure(go.Bar(
            x=top_weights,
            y=top_words,
            orientation='h',
            marker_color='indianred'
        ))
        fig.update_layout(
            title=f"Topic {i+1}",
            xaxis_title="Word Importance",
            yaxis_title="Word",
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig, use_container_width=True)

